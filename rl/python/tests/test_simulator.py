import json
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]


def simulator_path() -> str:
    configured = os.environ.get("DR_MARIO_SIMULATOR")
    if configured:
        return configured
    matches = sorted(
        path
        for path in ROOT.glob("dist-newstyle/**/dr-mario-simulator/dr-mario-simulator")
        if path.is_file() and os.access(path, os.X_OK)
    )
    if not matches:
        raise unittest.SkipTest("build dr-mario-simulator before running protocol tests")
    return str(matches[-1])


class SimulatorProtocolTests(unittest.TestCase):
    def request(self, messages: list[dict]) -> list[dict]:
        process = subprocess.run(
            [simulator_path()],
            input="".join(json.dumps(message) + "\n" for message in messages),
            text=True,
            capture_output=True,
            check=True,
        )
        return [json.loads(line) for line in process.stdout.splitlines()]

    def test_reset_is_seeded_and_has_fixed_spaces(self) -> None:
        first = self.request([{"type": "reset", "seed": 7, "level": 0}, {"type": "close"}])[0]
        second = self.request([{"type": "reset", "seed": 7, "level": 0}, {"type": "close"}])[0]
        self.assertEqual(first, second)
        self.assertEqual(len(first["observation"]), 133)
        self.assertEqual(len(first["action_mask"]), 512)
        self.assertTrue(any(first["action_mask"]))

    def test_invalid_action_is_replaced(self) -> None:
        reset = self.request([{"type": "reset", "seed": 7, "level": 0}, {"type": "close"}])[0]
        invalid_action = next(index for index, valid in enumerate(reset["action_mask"]) if not valid)
        replies = self.request([
            {"type": "reset", "seed": 7, "level": 0},
            {"type": "step", "action": invalid_action},
            {"type": "close"},
        ])
        info = replies[1]["info"]
        self.assertTrue(info["action_was_invalid"])
        self.assertIn(info["executed_action"], range(512))


if __name__ == "__main__":
    unittest.main()
