# Architecture

`slsqp_nmpc` solves condensed, finite-horizon nonlinear model predictive
control problems. The optimizer is the compiled Kraft SLSQP core that ships
inside SciPy. This package does not reimplement it. It changes only the
mathematical information that core is given, and the cost of producing that
information.

## The problem

For a horizon `N`, a fixed initial state `x0`, and a decision vector
`U = (u_0, ..., u_{N-1})`:

```
minimize    J(U) = 1/2 sum_{k=0}^{N-1} [ (x_k - x*)' Q (x_k - x*) + u_k' R u_k ]
                 + 1/2 (x_N - x*)' P (x_N - x*)

subject to  x_{k+1} = f(x_k, u_k),      x_0 fixed
            lo <= u_k <= hi
            c(x_{k+1}, u_k) >= 0        (optional path inequalities)
```

The dynamics are eliminated by single shooting, so the only decision variables
are the controls. `Q`, `R`, `P`, the bounds, the target and both callbacks are
stage invariant: one `f`, one `c`, one weight set applied at every stage. Only
inequality path constraints are supported; an equality must be supplied as two
opposed inequalities.

## Where the cost goes

A finite-difference SLSQP call perturbs every component of `U` and repeats the
whole nonlinear prediction for each derivative column. With `d = N * n_u` and
separately differenced objective and constraint callbacks, one derivative point
costs about `(1 + chi_c) * (d + 1)` full-horizon rollouts. Measured on the
double integrator, `d = 20`: 42 rollouts per derivative point, against a
predicted 42.

This package supplies the derivatives in closed form instead, and reuses the
trajectory when the decision vector has not changed. Both mechanisms matter,
and an ablation attributes roughly 2.6x to the exact derivatives and the larger
remaining share to the trajectory reuse. The `r_exact ~ 1` figure is a property
of the reuse, not of the derivatives alone.

## Module layout

```
slsqp_nmpc.py     NMPCProblem, solve_slsqp_nmpc, and the condensed evaluator
nmpc_runtime.py   the interface contract: read-only views, guarded violation
                  arithmetic, option validation, start resolution, and the
                  result finaliser
```

`nmpc_runtime.py` exists so that an engine and any comparator enforce one copy
of the rules rather than two that can drift. A measured ratio between two
engines is only a statement about derivative work if everything else -- the
callback contract, the caching, the start, the counting convention -- is
identical on both sides.

## The exact discrete adjoint

One forward rollout stores `x_k`, `A_k = df/dx` and `B_k = df/du` at every
stage. One reverse sweep then produces the whole gradient:

```
lambda_N     = P (x_N - x*)
grad_{u_k} J = R u_k + B_k' lambda_{k+1}          <- formed before the update
lambda_k     = Q (x_k - x*) + A_k' lambda_{k+1}
```

The gradient at stage `k` uses `lambda_{k+1}`, so it is formed before the
costate is advanced. `lambda_0` is computed and discarded, because `x_0` is
fixed and contributes no gradient. Dynamics work is linear in `N`, against
`N^2 n_u` for forward differences.

Verified against complex-step differentiation at `1.1e-16` on a three-state,
two-input system with a non-square `B` and stage-varying Jacobians. Four
plausible index errors -- using `lambda_k`, dropping the terminal weight,
shifting the state index, shifting the Jacobian index -- produce errors from
`1.1e-3` to `2.3e-1`, so the check has power.

## The condensed path sensitivity

The same rollout carries the sensitivity of the state to the whole decision
vector:

```
S_0     = 0
S_{k+1} = A_k S_k + B_k e_k'
row_k   = c_x S_{k+1}  (+ c_u in the columns of stage k)
```

Columns beyond stage `k` are structurally zero and remain exactly zero. The
stacked Jacobian is in the same stage-then-component order as the flattened
constraint vector, which is load-bearing: SLSQP pairs them positionally.
Verified against complex step at `2.2e-16`, with the row order agreeing
exactly.

The propagation is dense, so it multiplies through known-zero columns and the
flop count is quadratic in the horizon. At the horizons measured, per-stage
Python overhead dominates and the fitted exponent is 1.10, so this is left as a
documented trade rather than optimised.

## The byte-exact trajectory cache

Within one major iteration the core asks for the objective, the gradient, the
constraint values and the constraint Jacobian, often at the same point. The
cache key is the shape together with the exact bytes of the contiguous
float64 point, taken after dtype and contiguity are normalised.

That key is exact, not tolerant: there is no approximation and no
tolerance-dependent branch. Byte equality treats two NaN payloads as equal and
`-0.0` as distinct from `+0.0`, the second of which is a conservative miss.
Shape collision, dtype aliasing and stride aliasing are all closed. The cache
assumes the callbacks are pure functions of `(x, u)`; that is the stated
precondition of the whole mechanism.

Disabling the cache raises the exact path from 27 rollouts to 133 and returns a
bit-identical solution, which is what establishes that the reuse introduces no
approximation.

## What the engine refuses

Validation is not decoration here. Three of the defects found in review were
cases where a guard existed but did not hold.

Shapes of the dynamics return are checked at **every stage of every rollout**.
An assignment is not a shape gate: a scalar or a row vector broadcasts silently
into the destination, so a callback that is correct on the first rollout and
wrong afterwards would otherwise store a matrix of repeated rows.

Callback arguments are handed out as read-only views. The flag is a guard rail
rather than a boundary -- NumPy lets a holder reach the writeable owner through
`.base` -- but it converts the common accident, a callback normalising a state
in place, into an immediate error.

Weight matrices are tested for symmetry against a tolerance that **scales with
the matrix**, then symmetrised, then tested for definiteness. An absolute
tolerance rejects a Lyapunov or congruence-built weight whose asymmetry is pure
rounding at large magnitude.

`ftol` and `maxiter` are validated. A negative `ftol` does not terminate inside
the Fortran core above horizon five, and that loop cannot be interrupted from
Python.

An omitted start is the projection of the origin onto the input box, so a
positive-only actuator does not reject a call that supplied nothing. A supplied
start is validated, not projected, because silently moving a caller's warm
start would hide a specification error.

## What the result guarantees

Before reporting success the engine recomputes the returned point in full: the
controls, the objective, its gradient, the trajectory, both dynamics Jacobians
and, where path constraints are present, their values and both Jacobians. A
single non-finite entry, or a diagnostic that cannot be completed, withdraws
`success` while every computed field is preserved.

Checking only the constraint values is not enough, and that was a real defect:
on a problem without path constraints the constraint call returns an empty
array without rolling the trajectory out at all, so a non-finite final state
was invisible.

Violation arithmetic goes through one guarded helper. `max(0.0, -min(values))`
returns `0.0` for a NaN row, because a NaN comparison never wins, so the
finiteness test must precede the arithmetic.

Non-finite line-search trials are expected and are counted rather than raised,
so the core can retreat. The count is on the result as
`nonfinite_evaluations`, and a non-zero count means the declared box and the
model domain disagree and the returned point may not be optimal.

The result carries `rollouts`, `max_constraint_violation`, `bound_violation`,
`nonfinite_evaluations`, `nonfinite_blocks`, `nonfinite_warning` and
`diagnostic_error` on every path, so the schema is fixed rather than
conditional.

## Concurrency

Each public solve constructs its own evaluator, so two solves share no cache
state, and the problem's arrays are copied and frozen at construction. Four
hundred concurrent solves across threads and processes were compared bitwise
against serial with zero mismatches, and serial determinism was verified
separately so the comparison could fail.

Two caveats are real. The reentrancy of the compiled core is supplied by SciPy,
which in the tested version threads the formerly saved Fortran state through
each call; the declared floor is wider than the version verified. And callback
thread safety remains the caller's obligation, since the contract requires pure
callbacks.

The non-finite warning is emitted through a path that cannot raise, because
`warnings.filters` is process-global and `catch_warnings` is not thread-safe: a
filter set to `error` in any thread would otherwise turn a finished
optimisation into an exception.

## What this package does not claim

The gradient and the sensitivity are exact for the stated problem class, and
that is verified. Nothing here establishes convergence for an arbitrary
differentiable callback, global optimality for a nonconvex plant, a universal
speedup, or a worst-case execution time. Reported speedups are desktop
measurements under a stated contract against one value-only finite-difference
control, and they are run-dependent observations rather than machine-independent
bounds.

The acceptance certificate that accompanies the benchmark scores the four
first-order KKT conditions and carries no second-order content, so a passing
residual certifies an approximate first-order point and makes no local
optimality claim.
