"""JSON-lines client for the authoritative Haskell simulator."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence


class SimulatorClient:
    """Persistent request/response client for one Haskell simulator process."""

    def __init__(self, command: Sequence[str] | None = None) -> None:
        self._process = subprocess.Popen(
            list(command) if command is not None else ['stapill-hiss'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding='utf-8',
            bufsize=1, # line buffered
        )

    # TODO: this boundary interface needs some work
    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            code = self._process.poll()
            raise RuntimeError(f"simulator exited without a response (code {code})")
        response = json.loads(line)
        if response.get("type") == "error":
            raise RuntimeError(response.get("info", {}).get("error", "simulator error"))
        return response

    def reset(self, seed: int = 1, level: int = 0) -> dict[str, Any]:
        return self.request({"type": "reset", "seed": seed, "level": level})

    def step(self, action: int) -> dict[str, Any]:
        return self.request({"type": "step", "action": action})

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self._process.stdin.write(json.dumps({"type": "close"}) + "\n")
            self._process.stdin.flush()
        finally:
            self._process.wait(timeout=2)

    def __enter__(self) -> "SimulatorClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
