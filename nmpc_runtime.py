"""Shared runtime contract for the condensed NMPC engines.

Both the exact-derivative engine and the value-only finite-difference control
must enforce the same interface rules, because a measured ratio between them is
only a statement about derivative work if everything else is identical.  The
rules that were previously copied into each engine, and in two cases had
drifted, live here once:

* ``readonly_view`` — callback arguments are handed out non-writeable;
* ``violation_from_values`` — a non-finite constraint row can never be reported
  as zero violation;
* ``validate_solver_options`` — ``ftol`` and ``maxiter`` never reach the
  compiled core unvalidated;
* ``resolve_start`` — an omitted start is the projection of the origin onto the
  input box, and a supplied start must satisfy the bounds;
* ``nonfinite_blocks`` — the finiteness test applied to a returned point;
* ``emit_nonfinite_warning`` — a warning that cannot discard a solve.
"""

from __future__ import annotations

import warnings
from numbers import Integral, Real
from typing import Iterable, Sequence

import numpy as np

Array = np.ndarray

# Reported in ``result.status`` when a post-solve array is not finite.  The
# compiled SLSQP core uses 0 through 9, so this value cannot collide with one.
NONFINITE_DIAGNOSTIC_STATUS = 100
INFEASIBLE_DIAGNOSTIC_STATUS = 101

# Relative tolerance on weight-matrix asymmetry, applied as
# ``atol = SYMMETRY_TOLERANCE * max(1, max|M|)``.
SYMMETRY_TOLERANCE = 1e-10

# Backward-error allowance for a symmetric eigenvalue solve.  This is a small
# multiple of machine epsilon, not a modelling tolerance: a larger fixed
# relative number would admit a genuinely negative mode in an ill-scaled
# matrix and could reject a resolvable positive mode in the input weight.
DEFINITENESS_TOLERANCE = 100.0 * np.finfo(float).eps

# Independent feasibility contract for a returned control.  SLSQP's ``ftol``
# also controls objective and step tests, so a large ``ftol`` is not a safe
# substitute for an explicit feasibility decision.
DEFAULT_FEASIBILITY_TOLERANCE = 1e-9


def readonly_view(value: Array) -> Array:
    """Return a non-writeable view of ``value``.

    The flag is a guard rail rather than a boundary: NumPy lets a holder of the
    view reach the writeable owner through ``.base``.  It converts the common
    accident -- a callback normalising an angle in place -- into an immediate
    error, and callbacks are required by contract to be pure.
    """

    view = value.view()
    view.setflags(write=False)
    return view


def violation_from_values(values: Array) -> float:
    """Largest inequality violation, or NaN when a row is not finite.

    ``max(0.0, -np.min(values))`` evaluates to ``0.0`` for a NaN row, because a
    NaN comparison never wins, so an invalid point would be reported as exactly
    feasible.  The finiteness test must therefore precede the arithmetic.
    """

    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size == 0:
        return 0.0
    if not np.all(np.isfinite(flat)):
        return float("nan")
    return float(max(0.0, -np.min(flat)))


def validate_solver_options(ftol: float, maxiter: int) -> None:
    """Reject options that the compiled core cannot interpret.

    A negative ``ftol`` does not terminate inside the Fortran core at horizons
    above five, and the loop cannot be interrupted from Python, so this is
    validated rather than forwarded.
    """

    if (
        not isinstance(ftol, Real)
        or isinstance(ftol, (bool, np.bool_))
        or not np.isfinite(ftol)
        or ftol <= 0.0
    ):
        raise ValueError("ftol must be a positive finite float")
    if (
        not isinstance(maxiter, Integral)
        or isinstance(maxiter, (bool, np.bool_))
        or maxiter < 1
    ):
        raise ValueError("maxiter must be a positive integer")


def validate_feasibility_tolerance(value: float) -> None:
    """Require a nonnegative finite post-solve feasibility tolerance."""

    if (
        not isinstance(value, Real)
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or value < 0.0
    ):
        raise ValueError("feasibility_tolerance must be a nonnegative finite float")


def resolve_start(
    initial_controls: Array | None,
    decision_size: int,
    lower: Array,
    upper: Array,
) -> Array:
    """Return the decision vector the solver starts from.

    An omitted start is the projection of the origin onto the box, so a
    positive-only actuator does not reject a call that supplied nothing.  A box
    containing zero keeps the origin exactly.  A supplied start is validated
    rather than projected, because silently moving a caller's warm start would
    hide a specification error.
    """

    if initial_controls is None:
        return np.clip(np.zeros(decision_size), lower, upper)

    supplied = np.asarray(initial_controls, dtype=float)
    if supplied.shape != (decision_size,):
        raise ValueError(
            "initial_controls must be a one-dimensional vector with "
            f"shape ({decision_size},)"
        )
    start = supplied.copy()
    if not np.all(np.isfinite(start)):
        raise ValueError("initial_controls must contain only finite values")
    if np.any(start < lower) or np.any(start > upper):
        raise ValueError("initial_controls must satisfy the input bounds")
    return start


def nonfinite_blocks(blocks: Iterable[tuple[str, Array]]) -> tuple[str, ...]:
    """Names of the supplied arrays that contain a non-finite entry."""

    return tuple(
        name
        for name, block in blocks
        if not np.all(np.isfinite(np.asarray(block, dtype=float)))
    )


def bound_violation(controls: Array, lower: Array, upper: Array) -> float:
    """Largest box violation at ``controls``, or NaN if it is not finite."""

    point = np.asarray(controls, dtype=float)
    if not np.all(np.isfinite(point)):
        return float("nan")
    return float(
        max(
            0.0,
            float(np.max(lower - point, initial=0.0)),
            float(np.max(point - upper, initial=0.0)),
        )
    )


def emit_nonfinite_warning(count: int, rollouts: int) -> str:
    """Warn that trials left the model domain, without discarding the solve.

    ``warnings.filters`` is process-global and ``catch_warnings`` is not
    thread-safe, so a filter set to ``error`` -- by ``-W error``, by
    ``pytest.warns``, or by any library in another thread -- would turn this
    into an exception raised after the optimisation had finished.  The message
    is therefore carried on the result and the warning itself cannot propagate.
    """

    message = (
        f"{count} non-finite trajectory/path evaluations occurred across "
        f"{rollouts} rollouts; the input "
        "bounds and the model domain may disagree, and the returned point may "
        "not be optimal"
    )
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    except Exception:
        # A third-party ``warnings.showwarning`` hook may raise a non-Warning
        # exception.  Reporting a diagnostic must never discard a completed
        # optimization; the same text remains available on the result.
        pass
    return message


def finalize_result(
    result,
    blocks: Sequence[tuple[str, Array]],
    violation: float,
    box_violation: float,
    nonfinite_evaluations: int,
    rollouts: int,
    diagnostic_error: str | None,
    feasibility_tolerance: float = DEFAULT_FEASIBILITY_TOLERANCE,
) -> None:
    """Attach the verification fields and withdraw an unverifiable success.

    A caller that gates on ``success`` must not accept a point whose
    feasibility was never established, so both a non-finite array and a failed
    diagnostic clear the flag while every computed field is preserved.
    """

    unverified = nonfinite_blocks(blocks)
    result.rollouts = rollouts
    result.max_constraint_violation = violation
    result.bound_violation = box_violation
    result.nonfinite_evaluations = nonfinite_evaluations
    result.diagnostic_error = diagnostic_error
    result.nonfinite_blocks = unverified
    result.nonfinite_warning = None
    result.feasibility_tolerance = feasibility_tolerance

    if unverified or diagnostic_error is not None:
        result.success = False
        result.status = NONFINITE_DIAGNOSTIC_STATUS
        result.message = (
            f"non-finite {', '.join(unverified)} at the returned point"
            if unverified
            else f"post-solve verification failed: {diagnostic_error}"
        )
    elif (
        not np.isfinite(violation)
        or not np.isfinite(box_violation)
        or violation > feasibility_tolerance
        or box_violation > feasibility_tolerance
    ):
        result.success = False
        result.status = INFEASIBLE_DIAGNOSTIC_STATUS
        result.message = (
            "returned point failed independent feasibility verification: "
            f"path violation {violation:.3e}, box violation {box_violation:.3e}, "
            f"tolerance {feasibility_tolerance:.3e}"
        )

    if nonfinite_evaluations:
        result.nonfinite_warning = emit_nonfinite_warning(
            nonfinite_evaluations, rollouts
        )


__all__ = [
    "DEFAULT_FEASIBILITY_TOLERANCE",
    "DEFINITENESS_TOLERANCE",
    "INFEASIBLE_DIAGNOSTIC_STATUS",
    "NONFINITE_DIAGNOSTIC_STATUS",
    "SYMMETRY_TOLERANCE",
    "bound_violation",
    "emit_nonfinite_warning",
    "finalize_result",
    "nonfinite_blocks",
    "readonly_view",
    "resolve_start",
    "validate_solver_options",
    "validate_feasibility_tolerance",
    "violation_from_values",
]
