"""Korean 4-ball carom billiards RL environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from billiards.env import Billiards4BallEnv as _Billiards4BallEnv  # noqa: F401
    from billiards.inning_env import Billiards4BallInningEnv as _Inning  # noqa: F401

__all__ = ["Billiards4BallEnv", "Billiards4BallInningEnv"]


def __getattr__(name: str) -> Any:
    if name == "Billiards4BallEnv":
        from billiards.env import Billiards4BallEnv
        return Billiards4BallEnv
    if name == "Billiards4BallInningEnv":
        from billiards.inning_env import Billiards4BallInningEnv
        return Billiards4BallInningEnv
    raise AttributeError(f"module 'billiards' has no attribute {name!r}")
