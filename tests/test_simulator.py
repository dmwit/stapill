import unittest
import sys
open('/tmp/pytest_path.txt', 'w').write(str(sys.path))
from dr_mario_env import SimulatorClient


class SimulatorProtocolTests(unittest.TestCase):
    RESET_7_0 = {"type": "reset", "seed": 7, "level": 0}

    # TODO: errr... we should just be calling SimulatorClient methods, I think
    def requests(self, messages: list[dict]) -> list[dict]:
        with SimulatorClient() as sim:
            return list(map(sim.request, messages))

    def request(self, message: dict) -> dict:
        return self.requests([message])[0]

    def test_reset_is_seeded_and_has_fixed_spaces(self) -> None:
        first = self.request(self.RESET_7_0)
        second = self.request(self.RESET_7_0)
        self.assertEqual(first, second)
        self.assertEqual(len(first["observation"]), 1160)
        self.assertEqual(len(first["action_mask"]), 512)
        self.assertTrue(any(first["action_mask"]))

    def test_invalid_action_is_replaced(self) -> None:
        reset = self.request(self.RESET_7_0)
        invalid_action = next(index for index, valid in enumerate(reset["action_mask"]) if not valid)
        info = self.requests([
            self.RESET_7_0,
            {"type": "step", "action": invalid_action},
        ])[1]["info"]
        self.assertTrue(info["action_was_invalid"])
        self.assertIn(info["executed_action"], range(512))


if __name__ == "__main__":
    unittest.main()
