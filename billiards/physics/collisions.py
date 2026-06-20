"""Collision resolution and time-of-impact (TOI) helpers.

Two contact types are handled:

1. Ball ↔ cushion (rail). Decompose the velocity into normal/tangential
   components relative to the rail normal. Apply restitution to v_n and a
   coupled friction-with-side-spin update to v_t / wz.
2. Ball ↔ ball. Treat as elastic normal-only along the line of centers,
   with restitution e_b. Tangential velocity and angular velocity are
   carried through unchanged (a "throw" extension is gated by an
   un-implemented flag).

Time-of-impact helpers project each ball forward under free-flight
dynamics. The cushion solver accounts for sliding/rolling deceleration
exactly within a substep (constant acceleration, quadratic in t). The
ball-ball solver uses a first-order linearization (constant velocity) to
keep the algebra to a univariate quadratic — that approximation is fine
because TOIs at game speeds are short (~10 ms) compared to the deceleration
timescale (~1 s).
"""

from __future__ import annotations

import math

from .state import BallState, TableSpec

INF = float("inf")
TOI_EPS = 1e-9        # ignore numerical contact "now" when balls are already touching
NUMERIC_EPS = 1e-12


# ---------------------------------------------------------------------------
# Ball ↔ cushion
# ---------------------------------------------------------------------------


def resolve_ball_cushion(
    ball: BallState,
    normal: tuple[float, float],
    spec: TableSpec,
) -> None:
    """Han 2005 ball-cushion collision (port of PoolTool's han_2005 model).

    Reference: Han, I. (2005). "Dynamics in carom and three cushion billiards."
    Journal of Mechanical Science and Technology, 19(4), 976-984.
    DOI: 10.1007/BF02919180. Two regimes are handled exactly:
      (a) sliding-and-sticking — slip is killed at the contact patch
      (b) forward sliding     — Coulomb friction caps the impulse

    Cushion geometry: the rail nose contacts the ball at height ``h``
    (``spec.cushion_height``). The contact-normal makes angle
    ``θ_a = arcsin(h/R − 1)`` with horizontal, so the rebound transfers
    impulse not only in the table-plane normal direction but also
    vertically — coupling all three angular-velocity components and the
    tangential velocity.

    ``normal`` is the inward unit normal of the rail (pointing INTO the
    play area). Only axis-aligned cushions are exercised today, but the
    math handles arbitrary planar normals via ψ = atan2(out_y, out_x).
    """
    # Han uses the *outward* rail normal (table → rail). Our convention
    # is the inward play-area normal, so flip.
    out_nx, out_ny = -normal[0], -normal[1]
    psi = math.atan2(out_ny, out_nx)
    cp, sp = math.cos(psi), math.sin(psi)

    # --- rotate state into rail (cushion) frame: outward normal → +x_R ---
    vx_R = cp * ball.vx + sp * ball.vy
    vy_R = -sp * ball.vx + cp * ball.vy
    wx_R = cp * ball.wx + sp * ball.wy
    wy_R = -sp * ball.wx + cp * ball.wy
    wz_R = ball.wz

    # Ball must be moving INTO the cushion (toward +x_R). If not, the
    # caller scheduled a bounce that has already been resolved — bail.
    if vx_R <= 0.0:
        return

    R = spec.ball_radius
    m = spec.ball_mass
    h = spec.cushion_height
    e = spec.cushion_restitution
    mu = spec.cushion_friction

    # Cushion-contact angle. h/R must be in [0, 2]; clamp for safety.
    arg = max(-1.0, min(1.0, h / R - 1.0))
    theta_a = math.asin(arg)
    sin_t, cos_t = math.sin(theta_a), math.cos(theta_a)

    # Han Eqs (14): tangential surface-slip components and normal "compression".
    # Note: vz_R = 0 in our 2D model.
    sx = vx_R * sin_t + R * wy_R
    sy = -vy_R - R * wz_R * cos_t + R * wx_R * sin_t
    c = -vx_R * cos_t

    II = (2.0 / 5.0) * m * R * R
    A = 7.0 / (2.0 * m)
    B = 1.0 / m

    # Han Eqs (17, 20): normal impulse magnitude PzE; sliding-vs-sticking
    # threshold is whether the friction needed to kill slip exceeds μ·PzE.
    PzE = -(1.0 + e) * c / B
    abs_s = math.hypot(sx, sy)
    PzS = abs_s / A

    if PzS <= mu * PzE:
        # Sliding-and-sticking: friction fully kills the slip.
        PxE = sx / A
        PyE = sy / A
    else:
        # Forward sliding: Coulomb-capped friction.
        denom = abs_s if abs_s > NUMERIC_EPS else NUMERIC_EPS
        PxE = mu * PzE * sx / denom
        PyE = mu * PzE * sy / denom

    # Han Eqs (21, 22): rotate impulse from contact-normal frame to rail frame.
    PX = -PxE * sin_t - PzE * cos_t
    PY = PyE
    PZ = PxE * cos_t - PzE * sin_t

    # Han Eq (23): linear and angular velocity updates in rail frame.
    vx_R += PX / m
    vy_R += PY / m
    wx_R += -R / II * PY * sin_t
    wy_R += R / II * (PX * sin_t - PZ * cos_t)
    wz_R += R / II * PY * cos_t

    # --- rotate back to table frame ---
    ball.vx = cp * vx_R - sp * vy_R
    ball.vy = sp * vx_R + cp * vy_R
    ball.wx = cp * wx_R - sp * wy_R
    ball.wy = sp * wx_R + cp * wy_R
    ball.wz = wz_R


# ---------------------------------------------------------------------------
# Ball ↔ ball
# ---------------------------------------------------------------------------


def resolve_ball_ball(
    b1: BallState,
    b2: BallState,
    spec: TableSpec,
    throw_enabled: bool = False,
) -> None:
    """Resolve a ball-ball collision in place (elastic normal-only).

    Equal masses are assumed (spec.ball_mass is shared). Tangential v and
    angular velocity are unchanged. ``throw_enabled`` is accepted for
    API parity but currently a no-op (see module docstring).
    """
    if throw_enabled:
        # Reserved for a future tangential-impulse / throw model.
        pass

    dx = b2.x - b1.x
    dy = b2.y - b1.y
    d = math.hypot(dx, dy)
    if d < NUMERIC_EPS:
        return  # degenerate; refuse to divide
    nx = dx / d
    ny = dy / d

    # Normal velocities (signed: along +n̂, from b1 toward b2)
    v1n = b1.vx * nx + b1.vy * ny
    v2n = b2.vx * nx + b2.vy * ny
    rel_n = v1n - v2n  # > 0 means closing

    if rel_n <= 0.0:
        return  # already separating; nothing to do

    e_b = spec.ball_restitution
    # Equal-mass elastic-with-restitution exchange:
    #   v1n' = ((1 - e) v1n + (1 + e) v2n) / 2
    #   v2n' = ((1 + e) v1n + (1 - e) v2n) / 2
    v1n_new = 0.5 * ((1.0 - e_b) * v1n + (1.0 + e_b) * v2n)
    v2n_new = 0.5 * ((1.0 + e_b) * v1n + (1.0 - e_b) * v2n)

    dv1 = v1n_new - v1n
    dv2 = v2n_new - v2n
    b1.vx += dv1 * nx
    b1.vy += dv1 * ny
    b2.vx += dv2 * nx
    b2.vy += dv2 * ny


# ---------------------------------------------------------------------------
# Free-flight acceleration (for TOI integration)
# ---------------------------------------------------------------------------


def _free_flight_accel(b: BallState, spec: TableSpec) -> tuple[float, float]:
    """Acceleration vector for a TOI projection (constant over the substep).

    Approximation: use the *rolling* friction coefficient regardless of
    the current slip state. The rolling regime dominates the time the
    ball spends in flight (slip lasts only ~|u|/((7/2)μ_s g) ≈ tens of
    ms for a normal shot), so using μ_slip would massively under-predict
    travel and miss far walls. Using μ_roll over-predicts slightly during
    the brief slip phase, but the simulator's event loop re-projects each
    substep so the error never compounds beyond one event window.
    Direction is taken from the current velocity.
    """
    v_mag = math.hypot(b.vx, b.vy)
    if v_mag > 1e-9:
        a_mag = spec.mu_roll * spec.g
        return -a_mag * b.vx / v_mag, -a_mag * b.vy / v_mag
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Quadratic root utilities
# ---------------------------------------------------------------------------


def _smallest_positive_root(a: float, b: float, c: float, tol: float = TOI_EPS) -> float:
    """Smallest t > tol satisfying a t² + b t + c = 0, or +inf if none."""
    if abs(a) < NUMERIC_EPS:
        if abs(b) < NUMERIC_EPS:
            return INF
        t = -c / b
        return t if t > tol else INF
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return INF
    sq = math.sqrt(disc)
    # numerically stable form
    if b >= 0.0:
        t1 = (-b - sq) / (2.0 * a)
        t2 = (2.0 * c) / (-b - sq) if (b + sq) != 0.0 else (-b + sq) / (2.0 * a)
    else:
        t1 = (2.0 * c) / (-b + sq) if (sq - b) != 0.0 else (-b - sq) / (2.0 * a)
        t2 = (-b + sq) / (2.0 * a)
    cands = [t for t in (t1, t2) if t > tol and math.isfinite(t)]
    return min(cands) if cands else INF


# ---------------------------------------------------------------------------
# Ball ↔ cushion TOI
# ---------------------------------------------------------------------------


def toi_ball_cushion(
    ball: BallState,
    spec: TableSpec,
) -> tuple[float, tuple[float, float]] | None:
    """Earliest positive time when this ball center reaches a cushion line.

    Cushion lines (where the ball *center* contacts) are at
        x = R, x = W - R, y = R, y = H - R
    for play-area dimensions (W, H) = (spec.width, spec.height).

    Free-flight is approximated as constant acceleration (Coulomb friction
    in the current regime; see `_free_flight_accel`). The deceleration is
    quadratic in `dt` and we solve each cushion line as a 1-D quadratic.

    Returns ``(t_min, normal)`` where ``normal`` is the inward unit normal
    of the wall the ball will hit, or ``None`` if it never reaches one
    (e.g., it would stop before contact, or speed is zero).
    """
    R = spec.ball_radius
    W = spec.width
    H = spec.height

    ax, ay = _free_flight_accel(ball, spec)

    # Speed at time t: components vx + ax t. Friction must not flip the sign
    # of velocity within the candidate t — clamp against the deceleration
    # stop time of each axis component to keep the solution physical.
    def comp_stop_time(v: float, a: float) -> float:
        if abs(a) < NUMERIC_EPS:
            return INF
        if (v > 0 and a < 0) or (v < 0 and a > 0):
            return -v / a
        return INF

    # For each wall, candidate ts come from solving 0.5*a t² + v t + (p - target) = 0.
    walls = [
        # (target, p, v, a, normal_inward)
        (R,        ball.x, ball.vx, ax, (1.0, 0.0)),
        (W - R,    ball.x, ball.vx, ax, (-1.0, 0.0)),
        (R,        ball.y, ball.vy, ay, (0.0, 1.0)),
        (H - R,    ball.y, ball.vy, ay, (0.0, -1.0)),
    ]

    best_t = INF
    best_n: tuple[float, float] | None = None
    for target, p, v, a, n in walls:
        c = p - target
        # Already past or exactly on the wall and moving outward → ignore.
        # Use the inward normal: motion outward is when (v · -n_outward) > 0,
        # but here we're solving from the "inside"; if the ball is already
        # on the wrong side (within numerical noise) and moving away, skip.
        # The wall is hit when the position equation crosses 0.
        a_q = 0.5 * a
        t = _smallest_positive_root(a_q, v, c)
        if not math.isfinite(t):
            continue
        # Reject solutions past the velocity-component stop time (the ball
        # would have stopped along this axis before reaching the wall).
        t_stop = comp_stop_time(v, a)
        if t > t_stop + 1e-9:
            continue
        if t < best_t:
            best_t = t
            best_n = n

    if best_n is None or not math.isfinite(best_t):
        return None
    return best_t, best_n


# ---------------------------------------------------------------------------
# Ball ↔ ball TOI
# ---------------------------------------------------------------------------


def toi_ball_ball(
    b1: BallState,
    b2: BallState,
    spec: TableSpec,
) -> float:
    """Earliest positive time at which two balls collide.

    Approximation: each ball's velocity is held constant over the candidate
    substep (first-order in dt). Solving |Δp(t)| = 2R then yields a single
    univariate quadratic.

    Returns +inf if the balls never collide (e.g., separating or grazing
    too far). Returns +inf rather than 0 when the balls are already in
    contact and *separating*; if they are in contact and *closing*, the
    smallest positive root is still returned (the simulator's resolver
    will handle them).
    """
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    dvx = b2.vx - b1.vx
    dvy = b2.vy - b1.vy

    a = dvx * dvx + dvy * dvy
    b = 2.0 * (dx * dvx + dy * dvy)
    R = spec.ball_radius
    c = dx * dx + dy * dy - (2.0 * R) ** 2

    # Already overlapping or just touching:
    if c <= 0.0:
        # Closing? then collision time is essentially "now"
        if b < 0.0:
            return TOI_EPS  # immediate
        return INF

    # Separating and not overlapping → no collision.
    if b >= 0.0:
        return INF

    return _smallest_positive_root(a, b, c)
