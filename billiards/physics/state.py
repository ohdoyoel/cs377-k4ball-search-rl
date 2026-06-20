"""Pure data containers for billiards state.

Units are SI throughout: meters, seconds, kilograms, radians/sec.
Coordinate frame: table-fixed, +x = long axis, +y = short axis,
+z = upward (out of the cloth). All ball spins (wx, wy, wz) live in
this global frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


# ---------------------------------------------------------------------------
# Ball roles (4구 / Korean carom)
# ---------------------------------------------------------------------------


class BallRole(IntEnum):
    """Index into TableState.balls."""

    CUE_WHITE = 0   # 흰공 — player A's cue ball candidate
    CUE_YELLOW = 1  # 노란공 — player B's cue ball candidate
    RED_1 = 2       # 빨강 1
    RED_2 = 3       # 빨강 2


# ---------------------------------------------------------------------------
# Table specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableSpec:
    """Geometry + physical constants for the table.

    Defaults follow the Korean 4-ball *medium* table (中臺):
    play area 2540 mm × 1270 mm, ball diameter 65.5 mm.
    """

    # Geometry — play area inside the cushions
    width: float = 2.540        # m — long axis (x)
    height: float = 1.270       # m — short axis (y)

    # Ball
    ball_radius: float = 0.03275   # m (65.5 mm diameter)
    ball_mass: float = 0.210       # kg

    # Cue stick
    cue_mass: float = 0.55         # kg, typical carom/pool cue stick

    # Free-flight friction (PoolTool defaults: u_s=0.2, u_r=0.01).
    # Carom cloth literature: u_r is 0.005–0.015. Bumped to 0.018 here so
    # zero-cushion shots terminate in tractable wall-clock time even when
    # Han 2005 cushion losses don't catch them.
    g: float = 9.81                # m/s²
    mu_slip: float = 0.20          # sliding friction coeff (slip phase)
    mu_roll: float = 0.018         # rolling friction coeff
    mu_spin: float = 50.0          # vertical-axis spin decay (rad/s², linear);
                                   # side spin lifetime ≈ 4–6 s in free flight

    # Ball-cushion collision (Han 2005 model — see collisions.py)
    cushion_height: float = 0.0370  # m, carom rail nose height; gives θ_a ≈ 7.5°
    cushion_restitution: float = 0.85   # PoolTool default; velocity-dependent caps off
    cushion_friction: float = 0.20      # μ_w for tangential / spin coupling

    # Ball-ball collision (phenolic-on-phenolic)
    ball_restitution: float = 0.94
    ball_ball_friction: float = 0.06   # off by default; enables throw effect

    # Termination thresholds for "shot complete"
    rest_speed: float = 1e-3       # m/s
    rest_spin: float = 1e-2        # rad/s


# ---------------------------------------------------------------------------
# Ball state
# ---------------------------------------------------------------------------

BALL_DIM = 7  # x, y, vx, vy, wx, wy, wz


@dataclass
class BallState:
    """7-dim per-ball state."""

    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    wx: float = 0.0   # spin about +x
    wy: float = 0.0   # spin about +y
    wz: float = 0.0   # spin about +z (side spin)

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.vx, self.vy, self.wx, self.wy, self.wz],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, a: np.ndarray) -> BallState:
        return cls(
            x=float(a[0]), y=float(a[1]),
            vx=float(a[2]), vy=float(a[3]),
            wx=float(a[4]), wy=float(a[5]), wz=float(a[6]),
        )

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=np.float64)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    def is_at_rest(self, spec: TableSpec) -> bool:
        return (
            self.speed < spec.rest_speed
            and abs(self.wx) < spec.rest_spin
            and abs(self.wy) < spec.rest_spin
            and abs(self.wz) < spec.rest_spin
        )


# ---------------------------------------------------------------------------
# Cue action (RL action space)
# ---------------------------------------------------------------------------


@dataclass
class CueAction:
    """4-dim action: (theta, power, a, b).

    - theta: cue direction in radians, measured from +x axis
    - power: in [0, 1]; mapped to an initial cue speed by cue_impact
    - a: horizontal contact offset on the cue ball, in units of ball radius,
      ∈ [-1, 1]; positive = right english (side spin)
    - b: vertical contact offset, ∈ [-1, 1]; positive = top, negative = back
      (draw)
    Constraint: a² + b² ≤ 1 (the tip must touch the spherical cap).
    """

    theta: float
    power: float
    a: float = 0.0
    b: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.theta, self.power, self.a, self.b], dtype=np.float64)

    @classmethod
    def from_array(cls, x: np.ndarray) -> CueAction:
        return cls(theta=float(x[0]), power=float(x[1]),
                   a=float(x[2]), b=float(x[3]))

    def is_valid(self) -> bool:
        return (
            0.0 <= self.power <= 1.0
            and -1.0 <= self.a <= 1.0
            and -1.0 <= self.b <= 1.0
            and self.a * self.a + self.b * self.b <= 1.0 + 1e-9
        )


# ---------------------------------------------------------------------------
# Full table state
# ---------------------------------------------------------------------------


@dataclass
class TableState:
    """Full game state for a single instant."""

    balls: list[BallState] = field(default_factory=list)
    cue_id: int = int(BallRole.CUE_WHITE)
    spec: TableSpec = field(default_factory=TableSpec)
    t: float = 0.0   # simulation seconds since shot start

    # ----- factory -------------------------------------------------------

    @classmethod
    def initial_4ball(
        cls,
        cue_id: int = int(BallRole.CUE_WHITE),
        spec: TableSpec | None = None,
    ) -> TableState:
        """Standard Korean 4-ball (사구) opening layout.

        The horizontal play length L (long axis, +x) is split into eighths
        from the left cushion (x = 0). Three balls sit on the long centre
        line y = W/2; the cue sits just above its red:
            - 상대구 (yellow, opponent cue) : x = 1·L/8  (near the left cushion)
            - 목적구 (red 1)                : x = 2·L/8
            - 목적구 (red 2)                : x = 6·L/8
            - 수구  (white, player cue)     : same x as red 2 (6·L/8), placed
              directly above it with centre-to-centre gap = 2 ball-diameters.

        Layout (top view, left cushion on the left):
            ┌─────────────────────────────────────────┐
            │                            W            │
            │   Y      R1              R2             │
            │                                         │
            └─────────────────────────────────────────┘
        """
        spec = spec or TableSpec()
        L, W = spec.width, spec.height          # length, width of play area
        cy = W / 2.0
        unit = L / 8.0                          # horizontal length in eighths
        cue_gap = 2.0 * (2.0 * spec.ball_radius)  # 2 ball-diameters, cue above red 2
        balls = [
            BallState(x=6 * unit, y=cy + cue_gap),   # white (cue/수구) just above red 2
            BallState(x=1 * unit, y=cy),             # yellow (opponent/상대구) near left cushion
            BallState(x=2 * unit, y=cy),             # red 1 (목적구)
            BallState(x=6 * unit, y=cy),             # red 2 (목적구)
        ]
        return cls(balls=balls, cue_id=cue_id, spec=spec)

    # ----- vector / dict views ------------------------------------------

    def to_array(self) -> np.ndarray:
        """Stack ball states into an (N, 7) array."""
        return np.stack([b.to_array() for b in self.balls])

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        cue_id: int = int(BallRole.CUE_WHITE),
        spec: TableSpec | None = None,
        t: float = 0.0,
    ) -> TableState:
        balls = [BallState.from_array(arr[i]) for i in range(arr.shape[0])]
        return cls(balls=balls, cue_id=cue_id, spec=spec or TableSpec(), t=t)

    # ----- queries -------------------------------------------------------

    @property
    def cue_ball(self) -> BallState:
        return self.balls[self.cue_id]

    @property
    def red_balls(self) -> list[BallState]:
        return [self.balls[int(BallRole.RED_1)], self.balls[int(BallRole.RED_2)]]

    @property
    def opponent_ball(self) -> BallState:
        opp = (
            int(BallRole.CUE_YELLOW)
            if self.cue_id == int(BallRole.CUE_WHITE)
            else int(BallRole.CUE_WHITE)
        )
        return self.balls[opp]

    def all_at_rest(self) -> bool:
        return all(b.is_at_rest(self.spec) for b in self.balls)

    def copy(self) -> TableState:
        return TableState(
            balls=[BallState(**b.__dict__) for b in self.balls],
            cue_id=self.cue_id,
            spec=self.spec,
            t=self.t,
        )
