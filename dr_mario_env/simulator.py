"""JSON-lines client for the authoritative Haskell simulator."""

from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

@dataclass
class SimulatorReply:
    """Info that the simulator returns."""
    raw_observation: dict[str, Any]
    observation: np.ndarray
    actions: np.ndarray
    reward: dict[str, float]
    terminated: bool

class SimulatorClient:
    """Persistent request/response client for one Haskell simulator process."""

    def __init__(self, action_count: int, command: Sequence[str] | None = None) -> None:
        self.action_count = action_count
        self._process = subprocess.Popen(
            list(command) if command is not None else ['stapill-hiss'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding='utf-8',
            bufsize=1, # line buffered
        )

    def request(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        self._process.stdin.write(json.dumps({kind: body}) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            code = self._process.poll()
            raise RuntimeError(f"simulator exited without a response (code {code})")
        response = json.loads(line)
        try: return response[kind]
        except KeyError:
            raise RuntimeError(response.get("error", {}).get("description", "<unknown simulator error>"))

    def reset(self, seed: int = 1, level: int = 0) -> SimulatorReply:
        return self._reply_from_dict(self.request("reset", {"seed": seed, "level": level}))

    def step(self, action: int) -> dict[str, Any]:
        return self._reply_from_dict(self.request("step", {"action": action}))

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self._process.stdin.write(json.dumps({"close": []}) + "\n")
            self._process.stdin.flush()
        finally:
            self._process.wait(timeout=2)

    def __enter__(self) -> "SimulatorClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _reply_from_dict(self, raw: dict[str, Any]) -> SimulatorReply:
        raw_observation = raw["observation"]
        return SimulatorReply(
            raw_observation=raw_observation,
            observation=_flatten_observation(raw_observation),
            actions=_bool_indices(self.action_count, raw["validActions"]),
            reward=raw.get("reward", {}),
            terminated=raw.get("terminated", False),
            )

def _flatten_observation(raw: dict[str, Any]) -> np.ndarray:
    helpers = {
        "named": _flatten_named,
        "scalars": _flatten_scalars,
        "oneHot": _flatten_one_hot,
        "indices": _flatten_indices,
        "real": _flatten_real,
        }
    assert len(raw) == 1
    for (style, body) in raw.items():
        return helpers[style](body)

def _flatten_named(children: dict[str, Any]) -> np.ndarray:
    return np.concatenate([_flatten_observation(child) for (_, child) in sorted(children.items())], dtype=np.float32)

def _flatten_scalars(scalars: dict[str, float]) -> np.ndarray:
    return np.array([v for (_, v) in sorted(scalars.items())], dtype=np.float32)

def _flatten_one_hot(body: dict[str, int | list[int]]) -> np.ndarray:
    result = np.zeros((body["indexCount"], len(body["indices"])), dtype=np.float32)
    for (i_feature, i_class) in enumerate(body["indices"]):
        result[i_class, i_feature] = 1
    return result.flatten()

def _flatten_indices(body: dict[str, int | list[int]]) -> np.ndarray:
    result = np.zeros((body["indexCount"],), dtype=np.float32)
    for i in body["indices"]: result[i] = 1
    return result

def _flatten_real(body: list[int]) -> np.ndarray:
    return np.array(body, dtype=np.float32)

def _bool_indices(index_count: int, indices: list[int]) -> np.ndarray:
    result = np.zeros((index_count,), dtype=np.bool)
    for i in indices: result[i] = True
    return result
