# slsqp_nmpc

Exact-derivative SLSQP for condensed nonlinear model predictive control.

The optimizer is the compiled Kraft SLSQP core that ships inside SciPy, and it
is unchanged: the same major iteration, BFGS approximation, merit line search,
stopping rules and status codes. What changes is the mathematical information
that core is given, and what it costs to produce it:

- an **exact discrete adjoint** for the objective gradient, so dynamics work is
  linear in the horizon rather than quadratic;
- **exact condensed sensitivities** for the path-constraint Jacobian;
- a **byte-exact trajectory cache**, so the four callbacks a major iteration
  issues at one control vector share a single rollout.

Both mechanisms matter. An ablation attributes roughly 2.6x to the exact
derivatives and the larger remaining share to the trajectory reuse.

## Install

```
pip install numpy scipy
```

Python 3.9 or newer. Tested against NumPy 1.26 and SciPy 1.13.

## Use

```python
import numpy as np
from slsqp_nmpc import NMPCProblem, solve_slsqp_nmpc

def dynamics(state, control):
    """Return (next_state, df_dx, df_du).  Must be a pure function."""
    next_state = np.array([state[0] + 0.1 * state[1], state[1] + 0.1 * control[0]])
    df_dx = np.array([[1.0, 0.1], [0.0, 1.0]])
    df_du = np.array([[0.0], [0.1]])
    return next_state, df_dx, df_du

def path_constraints(state, control):
    """Return (c, dc_dx, dc_du) for c >= 0.  Optional."""
    return (
        np.array([1.5 - state[0]]),
        np.array([[-1.0, 0.0]]),
        np.array([[0.0]]),
    )

problem = NMPCProblem(
    horizon=20,
    initial_state=np.array([1.0, 0.0]),
    target_state=np.zeros(2),
    state_weight=np.diag([1.0, 0.1]),
    input_weight=np.array([[0.05]]),
    terminal_weight=np.diag([5.0, 0.5]),
    input_lower=np.array([-4.0]),
    input_upper=np.array([4.0]),
    dynamics=dynamics,
    path_constraints=path_constraints,
)

result = solve_slsqp_nmpc(problem, ftol=1e-9, maxiter=300)
print(result.success, result.x[:1], result.rollouts)
```

`result` is a SciPy `OptimizeResult` with additional fields that are present on
every return path:

| Field | Meaning |
|---|---|
| `rollouts` | full-horizon rollouts the solve required, excluding the post-solve diagnostic |
| `max_constraint_violation` | largest path violation, or NaN if a row was not finite |
| `bound_violation` | largest input-box violation |
| `nonfinite_evaluations` | trials that left the model domain |
| `nonfinite_blocks` | which arrays at the returned point were not finite |
| `nonfinite_warning` | the warning text, also carried here so a strict warning filter cannot discard a solve |
| `diagnostic_error` | why post-solve verification could not be completed, if it could not |

## Contract

The callbacks must be **pure functions of their arguments**. They receive
read-only views, and their returned arrays are copied before they enter the
cache. The trajectory cache keys on the exact bytes of the decision vector, so
a callback that reads a clock, an RNG or mutable external state breaks the
reuse.

The problem class is **stage invariant**: one dynamics callable, one constraint
callable, one weight set and one pair of bounds, applied at every stage.
`target_state` is a single setpoint rather than a reference trajectory. Only
inequality path constraints are supported; an equality must be supplied as two
opposed inequalities.

Before reporting success the solver recomputes the returned point in full --
controls, objective, gradient, trajectory, both dynamics Jacobians and, where
present, path values and both Jacobians. A single non-finite entry withdraws
`success` while every computed field is preserved. Checking only the constraint
values would miss a non-finite final state on a problem without path
constraints, where the constraint call returns an empty array without rolling
the trajectory out at all.

## Verification

The derivatives are checked against complex-step differentiation on
multi-state, multi-input systems with non-square `B` and stage-varying
Jacobians:

| Quantity | Agreement |
|---|---|
| Objective gradient (discrete adjoint) | `1.1e-16` |
| Path-constraint Jacobian (condensed sensitivity) | `2.2e-16` |
| Runge-Kutta stage Jacobians, two plants | `4.4e-16`, `7.1e-15` |

These checks have power, not just precision: four plausible index errors in the
adjoint produce errors from `1.1e-3` to `2.3e-1`.

Disabling the trajectory cache raises the exact path from 27 rollouts to 133
and returns a **bit-identical** solution, which is what establishes that the
reuse introduces no approximation rather than merely a small one.

The engine is re-entrant. Four hundred concurrent solves across threads and
processes were compared bitwise against serial with zero mismatches, and serial
determinism was verified separately so that the comparison could fail.

## Scope and limits

Stated plainly, because the derivative result and the timing result are
different kinds of claim.

The gradient and the sensitivity are **exact** for this problem class, and that
is verified symbolically and numerically. Nothing here establishes convergence
for an arbitrary differentiable callback, global optimality for a nonconvex
plant, a universal speedup, or a worst-case execution time.

Reported speedups are desktop measurements under a stated contract against one
value-only finite-difference control, run-dependent rather than
machine-independent. The technical note gives the contract, the per-repeat
dispersion and the configuration history.

The accompanying acceptance certificate scores the four first-order KKT
conditions and carries **no second-order content**, so a passing residual
certifies an approximate first-order point and makes no local-optimality claim.

## Contents

| File | |
|---|---|
| `slsqp_nmpc.py` | the solver: `NMPCProblem`, `solve_slsqp_nmpc`, and the condensed evaluator |
| `nmpc_runtime.py` | the interface contract the solver enforces |
| `ARCHITECTURE.md` | the derivations, the caching, and what the engine refuses |
| `slsqp_nmpc_technical_note.pdf` | the technical note, with the benchmark contract and two closed-loop case studies |

## Citation

Kamran Turkoglu, *SLSQP-NMPC: Exact Derivatives and Exact Trajectory Reuse for
Condensed Nonlinear Model Predictive Control*, technical note. See the PDF in
this repository.

The SLSQP core is that of D. Kraft, *A Software Package for Sequential
Quadratic Programming*, DFVLR-FB 88-28, 1988, as distributed in SciPy.
