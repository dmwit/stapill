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
        level_distribution = [],
        reward_weights: dict[str, float] | None = None,
        max_steps: int = 512,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(8 * 16 * 4) # columns x rows x rotations
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1160,), dtype=np.float32
        )
        # not public because we maintain a non-emptiness invariant
        self._level_distribution = level_distribution if level_distribution else [0]
        self._simulator_command = simulator_command
        self._simulator: SimulatorClient | None = None
        self._steps = 0
        self._last_response: dict[str, Any] | None = None

        if reward_weights is not None: self.reward_weights = reward_weights
        else: self.reward_weights = {
            "match/horz": 1.7,
            "match/vert": 1,
            "invalid": -10,
            "chasm": -0.1/16,
            "frames": -0.01/100,
            "viruses": 100,
            "halves": 3,
            "win": 1000,
            "lose": -1000,
            }

    def _get_simulator(self) -> SimulatorClient:
        if self._simulator is None:
            self._simulator = SimulatorClient(self.action_space.n, self._simulator_command)
        return self._simulator

    def choose_level(self, largest):
        return self.np_random.integers(0, int(largest) + 1)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._last_response = self._get_simulator().reset(
            seed=int(self.np_random.integers(-2**63, 2**63)) if seed is None else seed,
            level=int(self.np_random.choice(self._level_distribution)),
        )
        self._steps = 0
        assert self._last_response.observation.shape == (1160,)
        return self._last_response.observation, {}

    def step(
        self, action: np.int64 | int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._last_response is None:
            raise RuntimeError("reset must be called before step")
        response = self._get_simulator().step(int(action))
        self._steps += 1
        truncated = self._steps >= self.max_steps and not response.terminated
        self._last_response = response
        return (
            response.observation,
            self._weight_rewards(response.reward),
            response.terminated,
            truncated,
            response.reward,
        )

    def action_masks(self) -> np.ndarray:
        if self._last_response is None:
            return np.ones(self.action_space.n, dtype=bool)
        return self._last_response.actions

    COLOR_SYMBOLS = {0: '', 1: '\x1b[1;36m', 2: '\x1b[0;31m', 3: '\x1b[1;33m'}
    SHAPE_SYMBOLS = {0: ' ', 1: 'x', 2: 'o', 3: '⊃', 4: '⊂'}
    def render(self) -> str:
        if self._last_response is None:
            return "<not reset>"
        values = self._last_response.raw_observation["named"]
        colors = values["color"]["oneHot"]["indices"]
        shapes = values["shape"]["oneHot"]["indices"]
        lookahead = values["lookahead"]["oneHot"]["indices"]
        rows = []
        for row in range(16):
            cells = []
            for column in range(8):
                offset = row * 8 + column
                color = colors[offset]
                shape = shapes[offset]
                cells.append(self.COLOR_SYMBOLS[color] + self.SHAPE_SYMBOLS[shape])
            rows.append(" ".join(cells))
        rows.append(f"      {self.COLOR_SYMBOLS[lookahead[0]+1]}{self.SHAPE_SYMBOLS[4]} {self.COLOR_SYMBOLS[lookahead[1]+1]}{self.SHAPE_SYMBOLS[3]}")
        rows.reverse()
        rows += ["\x1b[0m"] * 5
        rendered = "\n".join(rows)
        if self.render_mode == "human":
            print(rendered)
        return rendered

    def close(self) -> None:
        if self._simulator is not None:
            self._simulator.close()
            self._simulator = None

    def _weight_rewards(self, rewards):
        reward = 0
        for k, v in self.reward_weights.items():
            reward += rewards[k] * v
        return reward
