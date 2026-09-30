"""JSON-lines client for the authoritative Haskell simulator."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence


def _default_command() -> list[str]:
    configured = os.environ.get("DR_MARIO_SIMULATOR")
    if configured:
        return [configured]

    root = Path(__file__).resolve().parents[3]
    candidates = [
        path
        for path in root.glob("dist-newstyle/**/dr-mario-simulator/dr-mario-simulator")
        if path.is_file() and os.access(path, os.X_OK)
    ]
    if candidates:
        return [str(sorted(candidates)[-1])]
    raise FileNotFoundError(
        "could not find dr-mario-simulator; build it with "
        "`cabal build exe:dr-mario-simulator` or set DR_MARIO_SIMULATOR"
    )


class SimulatorClient:
    """Persistent request/response client for one Haskell simulator process."""

    def __init__(self, command: Sequence[str] | None = None) -> None:
        self._process = subprocess.Popen(
            list(command) if command is not None else _default_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("simulator process pipes are unavailable")
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
        return self.request({"type": "reset", "seed": int(seed), "level": int(level)})

    def step(self, action: int) -> dict[str, Any]:
        return self.request({"type": "step", "action": int(action)})

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            if self._process.stdin is not None:
                self._process.stdin.write(json.dumps({"type": "close"}) + "\n")
                self._process.stdin.flush()
        finally:
            self._process.wait(timeout=2)

    def __enter__(self) -> "SimulatorClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
