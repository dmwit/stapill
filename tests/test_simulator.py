import unittest
import sys
open('/tmp/pytest_path.txt', 'w').write(str(sys.path))
from dr_mario_env import SimulatorClient


class SimulatorProtocolTests(unittest.TestCase):
    def test_reset_is_seeded_and_has_fixed_spaces(self) -> None:
        with SimulatorClient() as sim:
            first = sim.reset(seed=7, level=0)
            second = sim.reset(seed=7, level=0)

        self.assertEqual(first, second)
        self.assertEqual(len(first.observation), 1160)
        self.assertEqual(len(first.actions), 512)
        self.assertTrue(any(first.actions))

    def test_invalid_action_is_replaced(self) -> None:
        with SimulatorClient() as sim:
            reset = sim.reset(seed=7, level=0)
            invalid_action = next(index for index, valid in enumerate(reset.actions) if not valid)
            info = sim.step(invalid_action)
        self.assertEqual(info.reward["invalid"], 1)


if __name__ == "__main__":
    unittest.main()
