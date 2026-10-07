"""Gymnasium environment backed by the Haskell Dr. Mario simulator."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from typing import Any, Sequence
from gymnasium import spaces
from .simulator import SimulatorClient, SimulatorReply


class DrMarioEnv(gym.Env[np.ndarray, np.int64]):
    """One placement decision per environment step."""

    metadata = {"render_modes": ["ansi"]}
    # 14 = 4 colors + 5 shapes + 3 others + 2 positions
    OBSERVATION_SHAPE = (14, 8, 16)

    def __init__(
        self,
        simulator_command: Sequence[str] | None = None,
        level_distribution = [],
        reward_weights: dict[str, float] | None = None,
        reward_max: list[tuple[dict[str, float], dict[str, float]]] | None = None,
        max_steps: int = 512,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(8 * 16 * 4) # columns x rows x rotations
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.OBSERVATION_SHAPE, dtype=np.float32
        )
        # not public because we maintain a non-emptiness invariant
        self._level_distribution = level_distribution if level_distribution else [0]
        self._simulator_command = simulator_command
        self._simulator: SimulatorClient | None = None
        self._steps = 0
        self._last_response: SimulatorReply | None = None

        # wanna write self.rw = rw or {...}
        # but what if the user wants there to be 0 rewards from this term and
        # supplies rw = {}?
        if reward_weights is not None: self.reward_weights = reward_weights
        else: self.reward_weights = {
            "invalid": -1,
            "chasm": -0.0005/16,
            "frames": -0.0005/100,
            "halves": 0.001,
            "viruses": 0.1,
            "win": 1,
            "lose": -0.01,
            }

        if reward_max is not None: self.reward_max = reward_max
        else: self.reward_max = [
            ({"match/bl/horz": 0.00017}, {"match/bl/vert": 0.0001}),
            ({"match/other/horz": 0.00017}, {"match/other/vert": 0.0001}),
            ]

    def _get_simulator(self) -> SimulatorClient:
        if self._simulator is None:
            self._simulator = SimulatorClient(self.action_space.n, self._simulator_command)
        return self._simulator

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._last_response = self._get_simulator().reset(
            seed=int(self.np_random.integers(-2**63, 2**63)) if seed is None else seed,
            level=int(self.np_random.choice(self._level_distribution)),
        )
        self._steps = 0
        return self.observe(), {}

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
            self.observe(),
            self.reward(),
            response.terminated,
            truncated,
            response.reward,
        )

    def action_masks(self) -> np.ndarray:
        if self._last_response is None:
            return np.ones(self.action_space.n, dtype=bool)
        return self._last_response.actions

    COLOR_SYMBOLS = {' ': '', 'b': '\x1b[1;36m', 'r': '\x1b[0;31m', 'y': '\x1b[1;33m'}
    SHAPE_SYMBOLS = {0: ' ', 1: 'x', 2: 'o', 3: '⊃', 4: '⊂'}
    def render(self) -> str:
        if self._last_response is None:
            return "<not reset>"
        raw = self._last_response.raw_observation
        color_order = self._last_response.colors
        colors = raw["color"]["indices"]
        shapes = raw["shape"]["indices"]
        lkl = color_order[1]
        lkr = color_order[1] if raw["other"]["double"] else color_order[2]
        rows = []
        for row in range(16):
            cells = []
            for column in range(8):
                color = color_order[colors[column][row]]
                shape = shapes[column][row]
                cells.append(self.COLOR_SYMBOLS[color] + self.SHAPE_SYMBOLS[shape])
            rows.append(" ".join(cells))
        rows.append(f"      {self.COLOR_SYMBOLS[lkl]}{self.SHAPE_SYMBOLS[4]} {self.COLOR_SYMBOLS[lkr]}{self.SHAPE_SYMBOLS[3]}")
        rows.reverse()
        rows += ["\x1b[0m"]
        rendered = "\n".join(rows)
        if self.render_mode == "human":
            print(rendered)
        return rendered

    def close(self) -> None:
        if self._simulator is not None:
            self._simulator.close()
            self._simulator = None

    def observe(self):
        observation = self._last_response.observation
        board_shape = observation.color.shape[1:]
        other = np.broadcast_to(
            np.array(
                [[[obs]] for _, obs in sorted(observation.other.items())],
                dtype=np.float32,
                ),
            (len(observation.other),) + board_shape,
            )
        xs = _coordinate_plane(board_shape, 0)
        ys = _coordinate_plane(board_shape, 1)
        return np.concatenate([observation.color, observation.shape, other, xs, ys])

    def _reward_for_dict(self, weights):
        return sum(self._last_response.reward[k] * w for k, w in weights.items())

    def reward(self):
        linear_term = self._reward_for_dict(self.reward_weights)
        max_term = sum(max(self._reward_for_dict(d1), self._reward_for_dict(d2)) for d1, d2 in self.reward_max)
        return linear_term + max_term

def _coordinate_plane(board_shape, axis):
    result_shape = [1 for _ in board_shape]
    result_shape[axis] = board_shape[axis]
    result_shape = [1] + result_shape

    return np.broadcast_to(
        np.reshape(
            np.linspace(0, 1, board_shape[axis], endpoint=False, dtype=np.float32),
            result_shape,
            ),
        (1,) + board_shape,
        )
