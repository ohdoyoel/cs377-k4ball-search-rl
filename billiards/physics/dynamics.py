"""Free-flight dynamics for a billiard ball on a flat cloth.

State per ball: (x, y, vx, vy, wx, wy, wz). Coordinate frame:
+x = long axis, +y = short axis, +z = up out of cloth.

Three regimes:

1. Slipping. The contact point on the cloth moves with velocity
       u = v - R·(wy, -wx)
   When |u| > 0, kinetic friction acts at the contact:
       f      = -μ_s·m·g·û        (linear deceleration)
       τ      = r_contact × f      (with r_contact = -R·ẑ)
       I·ω̇   = τ                  (I = 2/5·m·R²)
   leading to
       dv/dt   = -μ_s·g·û
       dω⊥/dt = -(5μ_s·g)/(2R) · (ûy, -ûx)
   and the standard billiard result du/dt = -(7/2)·μ_s·g·û, so û is
   conserved and the ball reaches rolling at exactly t_roll = |u|/((7/2)·μ_s·g).

2. Rolling. The contact velocity vanishes; v and (wx, wy) are bound by
       wy = vx/R,  wx = -vy/R.
   Rolling friction decays v slowly:
       dv/dt = -μ_r·g·v̂
   ω⊥ is recomputed from v at every step.

3. Vertical-axis spin (wz). Decoupled from translation in the idealized
   point-contact model — spin about the vertical axis doesn't add to the
   contact-point velocity. We damp it linearly:
       d|wz|/dt = -μ_spin
   wz only matters at cushion bounces (collisions.py).

This means a ball with pure side spin and zero translation just spins
in place — exactly the desired behavior. Side spin only changes
trajectories when the ball hits a cushion or another ball.
"""

from __future__ import annotations

import numpy as np

from .state import BallState, TableSpec, TableState

# Threshold below which slip and velocity are treated as exactly zero.
SLIP_EPS = 1e-9
SPEED_EPS = 1e-9


def slip_velocity(ball: BallState, R: float) -> tuple[float, float]:
    """Velocity of the cloth-contact point relative to ground."""
    return ball.vx - R * ball.wy, ball.vy + R * ball.wx


# ---------------------------------------------------------------------------
# Per-ball substep
# ---------------------------------------------------------------------------


def _slip_substep(b: BallState, dt: float, ux: float, uy: float, u_mag: float,
                  spec: TableSpec) -> None:
    """Advance a slipping ball by dt. Slip direction is held constant
    (analytically valid since du/dt ∥ û). Caller must already have
    verified |u| > SLIP_EPS, and that the slip will not collapse to
    rolling within dt — otherwise split the step.
    """
    R = spec.ball_radius
    g = spec.g
    mu_s = spec.mu_slip

    uxh = ux / u_mag
    uyh = uy / u_mag

    ax = -mu_s * g * uxh
    ay = -mu_s * g * uyh

    # Position with constant acceleration
    b.x += b.vx * dt + 0.5 * ax * dt * dt
    b.y += b.vy * dt + 0.5 * ay * dt * dt
    b.vx += ax * dt
    b.vy += ay * dt

    spin_alpha = (5.0 * mu_s * g) / (2.0 * R)
    b.wx += -spin_alpha * uyh * dt
    b.wy += spin_alpha * uxh * dt


def _roll_substep(b: BallState, dt: float, spec: TableSpec) -> float:
    """Advance a rolling ball by dt; clamps to rest if it would stop
    within the step. Returns the time actually consumed (≤ dt).
    Maintains wy = vx/R, wx = -vy/R exactly at the end.
    """
    R = spec.ball_radius
    g = spec.g
    mu_r = spec.mu_roll

    v_mag = float(np.hypot(b.vx, b.vy))
    if v_mag < SPEED_EPS:
        b.vx = b.vy = 0.0
        b.wx = b.wy = 0.0
        return 0.0

    a_mag = mu_r * g
    t_stop = v_mag / a_mag if a_mag > 0 else float("inf")
    t_use = min(dt, t_stop)

    vxh = b.vx / v_mag
    vyh = b.vy / v_mag
    ax = -a_mag * vxh
    ay = -a_mag * vyh

    b.x += b.vx * t_use + 0.5 * ax * t_use * t_use
    b.y += b.vy * t_use + 0.5 * ay * t_use * t_use
    b.vx += ax * t_use
    b.vy += ay * t_use

    if t_use >= t_stop - 1e-15:
        b.vx = b.vy = 0.0
        b.wx = b.wy = 0.0
    else:
        # enforce rolling constraint
        b.wy = b.vx / R
        b.wx = -b.vy / R
    return t_use


def _wz_substep(b: BallState, dt: float, spec: TableSpec) -> None:
    """Linear decay of vertical-axis spin."""
    if b.wz == 0.0:
        return
    dwz = spec.mu_spin * dt
    if dwz >= abs(b.wz):
        b.wz = 0.0
    else:
        b.wz -= np.sign(b.wz) * dwz


# ---------------------------------------------------------------------------
# Public step
# ---------------------------------------------------------------------------


def advance_ball(b: BallState, dt: float, spec: TableSpec) -> None:
    """Advance a single ball by dt under free-flight dynamics.

    Handles the slipping → rolling transition exactly within the step.
    Does NOT detect collisions — that's the simulator's job.
    """
    R = spec.ball_radius

    ux, uy = slip_velocity(b, R)
    u_mag = float(np.hypot(ux, uy))

    if u_mag > SLIP_EPS:
        # |u| decays at rate (7/2)·μ_s·g; we know exactly when it hits zero.
        decay_rate = 3.5 * spec.mu_slip * spec.g
        t_to_roll = u_mag / decay_rate if decay_rate > 0 else float("inf")

        if t_to_roll <= dt:
            _slip_substep(b, t_to_roll, ux, uy, u_mag, spec)
            # Snap to rolling constraint to absorb any FP drift
            b.wy = b.vx / R
            b.wx = -b.vy / R
            remaining = dt - t_to_roll
            if remaining > 0:
                _roll_substep(b, remaining, spec)
        else:
            _slip_substep(b, dt, ux, uy, u_mag, spec)
    else:
        # Already rolling (or stationary)
        _roll_substep(b, dt, spec)

    _wz_substep(b, dt, spec)


def step_free(state: TableState, dt: float) -> None:
    """Advance every ball by dt under free-flight dynamics, in place."""
    for b in state.balls:
        advance_ball(b, dt, state.spec)
    state.t += dt


# ---------------------------------------------------------------------------
# Run-to-rest helper (no collisions yet — useful for unit tests)
# ---------------------------------------------------------------------------


def _clamp_to_rest(state: TableState) -> None:
    """Zero out sub-threshold residuals so a 'rest' state is exactly zero."""
    for b in state.balls:
        if b.is_at_rest(state.spec):
            b.vx = b.vy = 0.0
            b.wx = b.wy = 0.0
            b.wz = 0.0


def integrate_until_rest(
    state: TableState,
    dt: float = 2e-3,
    t_max: float = 30.0,
    record: bool = False,
) -> list[np.ndarray] | None:
    """Step state until all balls are at rest or t_max is reached.

    If `record=True`, returns a list of (N, 7) snapshots, one per dt.
    Useful for plotting trajectories. NO collision handling.

    Once `all_at_rest()` is satisfied, sub-threshold residual velocities
    are clamped to exactly zero so callers can reason about a clean
    fully-stopped state.
    """
    snaps: list[np.ndarray] | None = [] if record else None
    t = 0.0
    if snaps is not None:
        snaps.append(state.to_array())
    while t < t_max and not state.all_at_rest():
        step_free(state, dt)
        t += dt
        if snaps is not None:
            snaps.append(state.to_array())
    _clamp_to_rest(state)
    if snaps is not None:
        snaps[-1] = state.to_array()
    return snaps
