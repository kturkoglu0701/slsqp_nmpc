"""Exact-derivative SLSQP for condensed nonlinear model predictive control.

The SLSQP algorithm is unchanged.  The implementation supplies an exact
discrete adjoint, exact condensed path-constraint sensitivities, and a
byte-exact trajectory cache to the compiled SciPy SLSQP core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize

from nmpc_runtime import (
    NONFINITE_DIAGNOSTIC_STATUS,
    SYMMETRY_TOLERANCE,
    bound_violation,
    finalize_result,
    readonly_view,
    resolve_start,
    validate_solver_options,
    violation_from_values,
)

Array = np.ndarray
Dynamics = Callable[[Array, Array], tuple[Array, Array, Array]]
PathConstraints = Callable[[Array, Array], tuple[Array, Array, Array]]

def _immutable_array(value: Array, name: str) -> Array:
    array = np.array(value, dtype=float, copy=True, order="C")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class NMPCProblem:
    """Condensed, finite-horizon NMPC problem data.

    ``dynamics(x, u)`` returns ``(next_state, df_dx, df_du)``.
    ``path_constraints(x, u)`` returns ``(c, dc_dx, dc_du)`` for ``c >= 0``.
    Bounds, weights, dynamics, target, and constraint type are stage invariant.
    Bounds are specified for one stage and repeated over the horizon.  Only
    path inequalities are supported.  Both callbacks must be pure functions
    of their read-only array arguments.  Returned arrays are copied before
    they enter the evaluator cache.
    """

    horizon: int
    initial_state: Array
    target_state: Array
    state_weight: Array
    input_weight: Array
    terminal_weight: Array
    input_lower: Array
    input_upper: Array
    dynamics: Dynamics
    path_constraints: PathConstraints | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool):
            raise TypeError("horizon must be an integer")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not callable(self.dynamics):
            raise TypeError("dynamics must be callable")
        if self.path_constraints is not None and not callable(self.path_constraints):
            raise TypeError("path_constraints must be callable or None")

        names = (
            "initial_state",
            "target_state",
            "state_weight",
            "input_weight",
            "terminal_weight",
            "input_lower",
            "input_upper",
        )
        for name in names:
            object.__setattr__(self, name, _immutable_array(getattr(self, name), name))

        if self.initial_state.ndim != 1 or self.initial_state.size == 0:
            raise ValueError("initial_state must be a nonempty vector")
        if self.target_state.shape != self.initial_state.shape:
            raise ValueError("target_state must match initial_state")
        if self.input_lower.ndim != 1 or self.input_lower.size == 0:
            raise ValueError("input_lower must be a nonempty vector")
        if self.input_upper.shape != self.input_lower.shape:
            raise ValueError("input_upper must match input_lower")
        if np.any(self.input_lower > self.input_upper):
            raise ValueError("input_lower must not exceed input_upper")

        state_shape = (self.state_size, self.state_size)
        input_shape = (self.input_size, self.input_size)
        for name in ("state_weight", "terminal_weight"):
            matrix = getattr(self, name)
            if matrix.shape != state_shape:
                raise ValueError(f"{name} must have shape {state_shape}")
        if self.input_weight.shape != input_shape:
            raise ValueError(f"input_weight must have shape {input_shape}")

        for name in ("state_weight", "terminal_weight", "input_weight"):
            matrix = getattr(self, name)
            asymmetry = float(np.max(np.abs(matrix - matrix.T)))
            # The tolerance scales with the matrix, so a legitimately built
            # weight (a Gram matrix, a Lyapunov or Riccati solution) whose
            # asymmetry is pure rounding is accepted at any magnitude.
            scale = max(1.0, float(np.max(np.abs(matrix))))
            if asymmetry > SYMMETRY_TOLERANCE * scale:
                raise ValueError(
                    f"{name} must be symmetric; maximum asymmetry is "
                    f"{asymmetry:.3e} against a tolerance of "
                    f"{SYMMETRY_TOLERANCE * scale:.3e}"
                )
            object.__setattr__(
                self, name, _immutable_array(0.5 * (matrix + matrix.T), name)
            )

        for name in ("state_weight", "terminal_weight"):
            matrix = getattr(self, name)
            if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-12:
                raise ValueError(f"{name} must be positive semidefinite")
        if float(np.min(np.linalg.eigvalsh(self.input_weight))) <= 0.0:
            raise ValueError("input_weight must be positive definite")

    @property
    def state_size(self) -> int:
        return int(self.initial_state.size)

    @property
    def input_size(self) -> int:
        return int(self.input_lower.size)

    @property
    def decision_size(self) -> int:
        return self.horizon * self.input_size


class _CondensedEvaluator:
    """Objective and constraint evaluations owned by one synchronous solve."""

    def __init__(self, problem: NMPCProblem) -> None:
        self.problem = problem
        self.rollout_count = 0
        self.nonfinite_evaluations = 0
        self._raise_on_nonfinite = True
        self._decision: Array | None = None
        self._trajectory: tuple[Array, Array, Array] | None = None
        self._path_decision: Array | None = None
        self._path_data: tuple[Array, Array, Array] | None = None
        self._path_rows: int | None = None

    def allow_nonfinite_trials(self) -> None:
        """Let non-finite trial values reach SLSQP after a valid initial point."""

        self._raise_on_nonfinite = False

    def _handle_nonfinite(self, name: str, block: Array) -> bool:
        finite_by_stage = np.isfinite(block).reshape(self.problem.horizon, -1).all(axis=1)
        if np.all(finite_by_stage):
            return False
        stage = int(np.flatnonzero(~finite_by_stage)[0])
        if self._raise_on_nonfinite:
            raise ValueError(
                f"{name} must contain only finite values at stage {stage}"
            )
        return True

    def _controls(self, decision: Array) -> Array:
        return np.asarray(decision, dtype=float).reshape(
            self.problem.horizon, self.problem.input_size
        )

    def _rollout(self, decision: Array) -> tuple[Array, Array, Array]:
        point = np.ascontiguousarray(decision, dtype=float)
        if (
            self._decision is not None
            and self._trajectory is not None
            and point.shape == self._decision.shape
            and np.array_equal(point.view(np.uint8), self._decision.view(np.uint8))
        ):
            return self._trajectory

        problem = self.problem
        controls = self._controls(point)
        states = np.empty((problem.horizon + 1, problem.state_size))
        states[0] = problem.initial_state
        state_jacobians = np.empty(
            (problem.horizon, problem.state_size, problem.state_size)
        )
        input_jacobians = np.empty(
            (problem.horizon, problem.state_size, problem.input_size)
        )

        # The read-only handles are built once per rollout rather than twice per
        # stage.  Indexing a non-writeable array already yields a non-writeable
        # view, so the callback contract is unchanged while 2H view objects per
        # rollout become two.  The writeable arrays are retained for filling.
        states_view = readonly_view(states)
        controls_view = readonly_view(controls)
        shapes = (
            ("next state", (problem.state_size,)),
            ("state Jacobian", (problem.state_size, problem.state_size)),
            ("input Jacobian", (problem.state_size, problem.input_size)),
        )
        for stage in range(problem.horizon):
            returned = problem.dynamics(states_view[stage], controls_view[stage])
            state, state_jacobian, input_jacobian = returned
            # Every stage of every rollout is checked.  Assignment alone is not a
            # shape gate: a scalar or a row vector broadcasts silently into the
            # destination, so a callback that is correct on the validating
            # rollout and wrong afterwards would store a matrix of repeated rows.
            for (name, shape), value in zip(shapes, returned):
                if getattr(value, "shape", None) != shape:
                    if np.asarray(value, dtype=float).shape != shape:
                        raise ValueError(
                            f"dynamics {name} must have shape {shape} at stage {stage}"
                        )
            states[stage + 1] = state
            state_jacobians[stage] = state_jacobian
            input_jacobians[stage] = input_jacobian

        nonfinite = any(
            (
                self._handle_nonfinite("dynamics next state", states[1:]),
                self._handle_nonfinite(
                    "dynamics state Jacobian", state_jacobians
                ),
                self._handle_nonfinite(
                    "dynamics input Jacobian", input_jacobians
                ),
            )
        )
        self.nonfinite_evaluations += int(nonfinite)

        self.rollout_count += 1
        states.setflags(write=False)
        state_jacobians.setflags(write=False)
        input_jacobians.setflags(write=False)
        self._decision = point.copy()
        self._trajectory = (
            states,
            state_jacobians,
            input_jacobians,
        )
        return self._trajectory

    def _evaluate_path(self, decision: Array) -> tuple[Array, Array, Array]:
        problem = self.problem
        states, _, _ = self._rollout(decision)
        assert self._decision is not None
        if (
            self._path_decision is not None
            and self._path_data is not None
            and np.array_equal(
                self._decision.view(np.uint8), self._path_decision.view(np.uint8)
            )
        ):
            return self._path_data
        controls = self._controls(decision)
        expected_rows = self._path_rows
        value_matrix: Array | None = None
        state_tensor: Array | None = None
        input_tensor: Array | None = None
        # ``states`` is already frozen by ``_rollout``, so ``states[stage + 1]``
        # is a non-writeable view and needs no second one.  ``controls`` is a
        # fresh reshape of the caller's vector and does need one.
        controls_view = readonly_view(controls)
        for stage in range(problem.horizon):
            state_view = states[stage + 1]
            control_view = controls_view[stage]
            assert problem.path_constraints is not None
            values, state_jacobian, input_jacobian = problem.path_constraints(
                state_view, control_view
            )
            values = np.atleast_1d(np.asarray(values, dtype=float))
            state_jacobian = np.asarray(state_jacobian, dtype=float)
            input_jacobian = np.asarray(input_jacobian, dtype=float)
            rows = int(values.size)
            if rows == 0:
                raise ValueError(
                    "path_constraints must return at least one row"
                )
            if expected_rows is None:
                expected_rows = rows
            if rows != expected_rows:
                raise ValueError(
                    "path_constraints must return the same number of rows at "
                    f"every stage; expected {expected_rows}, received {rows} "
                    f"at stage {stage}"
                )
            expected = (
                ("values", values, (rows,)),
                ("state Jacobian", state_jacobian, (rows, problem.state_size)),
                ("input Jacobian", input_jacobian, (rows, problem.input_size)),
            )
            for name, value, shape in expected:
                if value.shape != shape:
                    raise ValueError(
                        f"path constraint {name} must have shape {shape} "
                        f"at stage {stage}"
                    )
            if value_matrix is None:
                value_matrix = np.empty((problem.horizon, rows))
                state_tensor = np.empty((problem.horizon, rows, problem.state_size))
                input_tensor = np.empty((problem.horizon, rows, problem.input_size))
            value_matrix[stage] = values
            assert state_tensor is not None and input_tensor is not None
            state_tensor[stage] = state_jacobian
            input_tensor[stage] = input_jacobian
        assert value_matrix is not None
        assert state_tensor is not None
        assert input_tensor is not None
        nonfinite = any(
            (
                self._handle_nonfinite("path constraint values", value_matrix),
                self._handle_nonfinite(
                    "path constraint state Jacobian", state_tensor
                ),
                self._handle_nonfinite(
                    "path constraint input Jacobian", input_tensor
                ),
            )
        )
        self.nonfinite_evaluations += int(nonfinite)
        value_matrix.setflags(write=False)
        state_tensor.setflags(write=False)
        input_tensor.setflags(write=False)
        self._path_rows = expected_rows
        data = value_matrix, state_tensor, input_tensor
        self._path_decision = self._decision.copy()
        self._path_data = data
        return data

    def objective(self, decision: Array) -> float:
        problem = self.problem
        states, _, _ = self._rollout(decision)
        controls = self._controls(decision)
        errors = states[:-1] - problem.target_state
        terminal_error = states[-1] - problem.target_state
        return float(
            0.5 * np.sum((errors @ problem.state_weight) * errors)
            + 0.5 * np.sum((controls @ problem.input_weight) * controls)
            + 0.5 * terminal_error @ problem.terminal_weight @ terminal_error
        )

    def objective_jacobian(self, decision: Array) -> Array:
        problem = self.problem
        states, state_jacobians, input_jacobians = self._rollout(decision)
        controls = self._controls(decision)
        gradient = np.empty_like(controls)
        costate = problem.terminal_weight @ (states[-1] - problem.target_state)

        for stage in range(problem.horizon - 1, -1, -1):
            gradient[stage] = (
                problem.input_weight @ controls[stage]
                + input_jacobians[stage].T @ costate
            )
            costate = (
                problem.state_weight @ (states[stage] - problem.target_state)
                + state_jacobians[stage].T @ costate
            )
        return gradient.ravel()

    def inequality(self, decision: Array) -> Array:
        problem = self.problem
        if problem.path_constraints is None:
            return np.empty(0)
        values, _, _ = self._evaluate_path(decision)
        return values.reshape(-1)

    def inequality_jacobian(self, decision: Array) -> Array:
        problem = self.problem
        if problem.path_constraints is None:
            return np.empty((0, problem.decision_size))

        _, state_jacobians, input_jacobians = self._rollout(decision)
        _, constraint_state_jacobians, constraint_input_jacobians = (
            self._evaluate_path(decision)
        )
        sensitivity = np.zeros((problem.state_size, problem.decision_size))
        rows: list[Array] = []

        for stage in range(problem.horizon):
            sensitivity = state_jacobians[stage] @ sensitivity
            block = slice(stage * problem.input_size, (stage + 1) * problem.input_size)
            sensitivity[:, block] += input_jacobians[stage]
            constraint_state_jacobian = constraint_state_jacobians[stage]
            constraint_input_jacobian = constraint_input_jacobians[stage]
            row = constraint_state_jacobian @ sensitivity
            row[:, block] += constraint_input_jacobian
            rows.append(row)
        return np.vstack(rows)


def _attach_final_point_diagnostics(
    result: OptimizeResult,
    problem: NMPCProblem,
    evaluator: _CondensedEvaluator,
    lower: Array,
    upper: Array,
) -> None:
    """Re-establish finiteness and feasibility at the returned point.

    Every array the caller is told about is recomputed and tested: the controls,
    the objective, its gradient, the trajectory, both dynamics Jacobians, and
    when path constraints are present their values and both Jacobians.  A single
    non-finite entry overrides ``success``, because ``max(0.0, -np.min(values))``
    evaluates to ``0.0`` for a NaN row and would otherwise report an invalid
    point as feasible.  Checking only the constraint values would miss a
    non-finite final state on an unconstrained problem, where ``inequality``
    returns an empty array without rolling the trajectory out at all.
    """

    controls = np.asarray(result.x, dtype=float)
    blocks: list[tuple[str, Array]] = [("controls", controls)]
    objective = result.get("fun", None)
    if objective is not None:
        blocks.append(("objective", np.asarray(objective, dtype=float)))

    violation = float("nan")
    diagnostic_error: str | None = None
    try:
        states, state_jacobians, input_jacobians = evaluator._rollout(controls)
        blocks += [
            ("states", states),
            ("dynamics state Jacobians", state_jacobians),
            ("dynamics input Jacobians", input_jacobians),
            ("objective gradient", evaluator.objective_jacobian(controls)),
        ]
        if problem.path_constraints is None:
            violation = 0.0
        else:
            values, path_state, path_input = evaluator._evaluate_path(controls)
            blocks += [
                ("path values", values),
                ("path state Jacobians", path_state),
                ("path input Jacobians", path_input),
                ("path Jacobian", evaluator.inequality_jacobian(controls)),
            ]
            violation = violation_from_values(values)
    except (MemoryError, RecursionError):
        # Resource exhaustion is not a model defect and must not be recorded as
        # one, so it is re-raised rather than folded into a diagnostic string.
        raise
    except Exception as error:
        diagnostic_error = f"{type(error).__name__}: {error}"

    finalize_result(
        result,
        blocks,
        violation,
        bound_violation(controls, lower, upper),
        evaluator.nonfinite_evaluations,
        evaluator.rollout_count,
        diagnostic_error,
    )


def solve_slsqp_nmpc(
    problem: NMPCProblem,
    initial_controls: Array | None = None,
    *,
    ftol: float = 1e-9,
    maxiter: int = 600,
) -> OptimizeResult:
    """Solve a condensed NMPC problem with exact-derivative SLSQP."""

    validate_solver_options(ftol, maxiter)

    evaluator = _CondensedEvaluator(problem)
    lower = np.tile(problem.input_lower, problem.horizon)
    upper = np.tile(problem.input_upper, problem.horizon)
    decision0 = resolve_start(
        initial_controls, problem.decision_size, lower, upper
    )

    # A non-finite initial trajectory is a specification error.  Later trial
    # evaluations are returned to SLSQP so its line search can retreat.
    evaluator._rollout(decision0)
    if problem.path_constraints is not None:
        evaluator._evaluate_path(decision0)
    evaluator.allow_nonfinite_trials()

    bounds = Bounds(lower, upper)
    constraints: list[dict[str, object]] = []
    if problem.path_constraints is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": evaluator.inequality,
                "jac": evaluator.inequality_jacobian,
            }
        )

    # Non-finite line-search trials are an expected, explicitly counted
    # condition, so NumPy's own floating-point warnings are suppressed for the
    # duration: under a filter set to "error" they would otherwise raise after
    # the optimisation finished and discard it.  The state is thread-local.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        result = minimize(
            evaluator.objective,
            decision0,
            method="SLSQP",
            jac=evaluator.objective_jacobian,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": ftol, "maxiter": maxiter, "disp": False},
        )
        _attach_final_point_diagnostics(result, problem, evaluator, lower, upper)
    return result


__all__ = [
    "NONFINITE_DIAGNOSTIC_STATUS",
    "SYMMETRY_TOLERANCE",
    "NMPCProblem",
    "solve_slsqp_nmpc",
]
