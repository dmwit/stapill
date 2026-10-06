"""JSON-lines client for the authoritative Haskell simulator."""

from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
import subprocess
from typing import Any, Sequence

#from .observation import encode_observation

@dataclass
class Observation:
    color: np.ndarray # 4x8x16
    shape: np.ndarray # 5x8x16
    other: dict[str, float]

    def __eq__(self, other):
        return (True
            and np.array_equal(self.color, other.color)
            and np.array_equal(self.shape, other.shape)
            and self.other == other.other
            )

@dataclass
class SimulatorReply:
    """Info that the simulator returns."""
    raw_observation: dict[str, Any]
    observation: Observation
    actions: np.ndarray
    reward: dict[str, float]
    terminated: bool
    colors: str

    def __eq__(self, other):
        obs_eq = self.raw_observation == other.raw_observation
        if obs_eq: assert self.observation == other.observation
        return (True
            and obs_eq
            and np.array_equal(self.actions, other.actions)
            and self.reward == other.reward
            and self.terminated == other.terminated
            and self.colors == other.colors
            )

class SimulatorClient:
    """Persistent request/response client for one Haskell simulator process."""

    def __init__(self, action_count: int = 512, command: Sequence[str] | None = None) -> None:
        self.action_count = action_count
        self._process = subprocess.Popen(
            list(command) if command is not None else ['stapill-hiss'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding='utf-8',
            bufsize=1, # line buffered
        )

    def _send(self, kind: str, body: dict[str, Any]):
        self._process.stdin.write(json.dumps({kind: body}) + "\n")
        self._process.stdin.flush()

    def request(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        self._send(kind, body)
        line = self._process.stdout.readline()
        if not line:
            code = self._process.poll()
            raise RuntimeError(f"simulator exited without a response (code {code})")
        response = json.loads(line)
        try: return response["success"]
        except KeyError:
            raise RuntimeError(response.get("error", {}).get("description", "<unknown simulator error>"))

    def reset(self, seed: int = 1, level: int = 0) -> SimulatorReply:
        return self._reply_from_dict(self.request("reset", {"seed": seed, "level": level}))

    def step(self, action: int) -> dict[str, Any]:
        return self._reply_from_dict(self.request("step", {"action": action}))

    def close(self) -> None:
        if self._process.poll() is not None: return
        try: self._send("close", [])
        finally: self._process.wait(timeout=2)

    def __enter__(self) -> "SimulatorClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _reply_from_dict(self, raw: dict[str, Any]) -> SimulatorReply:
        raw_observation = raw["observation"]
        return SimulatorReply(
            raw_observation=raw_observation,
            observation=decode_observation(raw_observation),
            actions=_bool_indices(self.action_count, raw["validActions"]),
            reward=raw["reward"],
            terminated=raw["terminated"],
            colors=' ' + raw["colors"],
            )

def decode_observation(obs: dict[str, Any]) -> Observation:
    return Observation(
        color=_float_indices_from_dict(obs["color"]),
        shape=_float_indices_from_dict(obs["shape"]),
        other=obs["other"],
        )

def _bool_indices(index_count: int, indices: list[int]) -> np.ndarray:
    result = np.zeros((index_count,), dtype=np.bool)
    for i in indices: result[i] = True
    return result

def _float_indices_from_dict(one_hot: dict[str, int | list[list[int]]]) -> np.ndarray:
    return _float_indices(one_hot["indexCount"], one_hot["indices"])

def _float_indices(index_count: int, indices: list[list[int]]) -> np.ndarray:
    indices = np.array(indices, dtype=np.int64)
    result = np.zeros((index_count,) + indices.shape, dtype=np.float32)
    x, y = np.indices(indices.shape)
    result[indices, x, y] = 1
    return result
