"""Cue impact (Marlow / Dr. Dave instantaneous-point model).

Closed-form impulse solution for a level cue striking a stationary ball
at contact offset (a, b) on the rear hemisphere. Adapted from PoolTool's
``physics/resolve/stick_ball/instantaneous_point`` with the cue tilt fixed
at θ = 0 (no jump / massé).

Quantities:
    V0       cue tip speed at impact (m/s)        = action.power * V_MAX_DEFAULT
    a, b     normalized contact offset, |a²+b²| ≤ 1
    m, M     ball / cue masses                    (spec.ball_mass, spec.cue_mass)
    R        ball radius                          (spec.ball_radius)
    I/m      = (2/5) R²

Marlow energy-loss factor (level cue):
    v_eff = 2·V0 / (1 + m/M + (5/2)·(a² + b²))

Important:
- Hitting off-center now correctly *reduces* both the resulting linear
  speed AND the spin (because both come from the same ``v_eff``).
- A center hit (a=b=0) gives the maximum ball speed, ``2·V0/(1+m/M)``,
  which for M=0.55, m=0.21 is ≈ 1.45·V0.
- A miscue-limit english (a²+b²=1) cuts that to ``2·V0/(1+m/M+2.5) ≈ 0.51·V0``.

Linear and angular velocity in the *table* frame:
    v   =  v_eff · (cos θ_aim, sin θ_aim, 0)
    ω   = (5·v_eff / 2R) · ( -b sin θ_aim,  b cos θ_aim,  -a )

Sign conventions (right-hand-rule, +z up):
    a > 0  → contact on the right side of the ball   → ωz < 0  (right english)
    b > 0  → tip strikes ABOVE center                → top / follow spin
    b < 0  → tip strikes BELOW center                → back / draw spin
"""

from __future__ import annotations

import math

from .state import CueAction, TableState

# Default *cue tip* speed at full power. Center-hit ball speed is then
# 2·V_MAX_DEFAULT / (1 + m/M) ≈ 6.3 m/s for the carom defaults — a strong
# but not extreme break shot.
V_MAX_DEFAULT = 4.5   # m/s


def cue_impulse(
    action: CueAction,
    ball_radius: float,
    ball_mass: float = 0.210,
    cue_mass: float = 0.55,
    v_max: float = V_MAX_DEFAULT,
) -> tuple[float, float, float, float, float]:
    """Return (vx, vy, wx, wy, wz) imparted to a stationary cue ball.

    Pure function — no state mutation. ``v_max`` parameterizes the *cue
    tip* speed at full power, not the ball speed. Use ``apply_cue`` for
    the convenience wrapper that pulls m and M from a TableState.spec.
    """
    if not action.is_valid():
        raise ValueError(f"invalid CueAction: {action}")

    V0 = action.power * v_max
    a, b = action.a, action.b
    theta = action.theta

    # Marlow impulse: scalar magnitude of the post-impact ball speed.
    energy_factor = 2.0 / (1.0 + ball_mass / cue_mass + 2.5 * (a * a + b * b))
    v_eff = V0 * energy_factor

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    vx = v_eff * cos_t
    vy = v_eff * sin_t

    spin_factor = (5.0 * v_eff) / (2.0 * ball_radius)
    wx = -spin_factor * b * sin_t
    wy = spin_factor * b * cos_t
    wz = -spin_factor * a

    return vx, vy, wx, wy, wz


def apply_cue(
    state: TableState,
    action: CueAction,
    v_max: float = V_MAX_DEFAULT,
) -> TableState:
    """Apply cue impulse to ``state.balls[cue_id]`` in place; return state.

    Reads ``ball_mass`` and ``cue_mass`` from ``state.spec`` so the energy-
    loss factor stays consistent with the equipment. Raises if the cue
    ball is not at rest.
    """
    cue = state.balls[state.cue_id]
    if not cue.is_at_rest(state.spec):
        raise ValueError("cue ball must be at rest before applying a new shot")

    vx, vy, wx, wy, wz = cue_impulse(
        action,
        ball_radius=state.spec.ball_radius,
        ball_mass=state.spec.ball_mass,
        cue_mass=state.spec.cue_mass,
        v_max=v_max,
    )
    cue.vx, cue.vy = vx, vy
    cue.wx, cue.wy, cue.wz = wx, wy, wz
    state.t = 0.0
    return state
