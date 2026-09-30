"""Gymnasium environment backed by the Haskell Dr. Mario simulator."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised when optional deps are absent
    raise ImportError(
        "DrMarioEnv requires gymnasium; install the Python RL dependencies first"
    ) from exc

from .simulator import SimulatorClient


class DrMarioEnv(gym.Env[np.ndarray, np.int64]):
    """One placement decision per environment step."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        simulator_command: Sequence[str] | None = None,
        level: int = 0,
        max_steps: int = 512,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.level = level
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(8 * 16 * 4)
        self.observation_space = spaces.Box(
            low=0.0, high=255.0, shape=(133,), dtype=np.float32
        )
        self._simulator_command = simulator_command
        self._simulator: SimulatorClient | None = None
        self._steps = 0
        self._last_response: dict[str, Any] | None = None

    def _get_simulator(self) -> SimulatorClient:
        if self._simulator is None:
            self._simulator = SimulatorClient(self._simulator_command)
        return self._simulator

    @staticmethod
    def _observation(response: dict[str, Any]) -> np.ndarray:
        observation = np.asarray(response["observation"], dtype=np.float32)
        if observation.shape != (133,):
            raise ValueError(f"unexpected observation shape: {observation.shape}")
        return observation

    @staticmethod
    def _info(response: dict[str, Any]) -> dict[str, Any]:
        return dict(response.get("info", {}))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        requested_level = self.level if options is None else options.get("level", self.level)
        response = self._get_simulator().reset(
            seed=1 if seed is None else seed,
            level=int(requested_level),
        )
        self._steps = 0
        self._last_response = response
        return self._observation(response), self._info(response)

    def step(
        self, action: np.int64 | int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._last_response is None:
            raise RuntimeError("reset must be called before step")
        response = self._get_simulator().step(int(action))
        self._steps += 1
        truncated = self._steps >= self.max_steps and not response["terminated"]
        self._last_response = response
        return (
            self._observation(response),
            float(response["reward"]),
            bool(response["terminated"]),
            truncated,
            self._info(response),
        )

    def action_masks(self) -> np.ndarray:
        if self._last_response is None:
            return np.ones(self.action_space.n, dtype=bool)
        return np.asarray(self._last_response["action_mask"], dtype=bool)

    def render(self) -> str:
        if self._last_response is None:
            return "<not reset>"
        board = self._last_response["observation"][:128]
        rows = [board[offset : offset + 8] for offset in range(0, 128, 8)]
        rendered = "\n".join(" ".join(f"{cell:02d}" for cell in row) for row in rows)
        if self.render_mode == "ansi":
            return rendered
        print(rendered)
        return rendered

    def close(self) -> None:
        if self._simulator is not None:
            self._simulator.close()
            self._simulator = None
