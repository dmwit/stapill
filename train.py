"""Minimal MaskablePPO training entry point."""

from __future__ import annotations

import argparse

from dr_mario_env import DrMarioEnv
from sb3_contrib import MaskablePPO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--output", default="dr_mario_maskable_ppo")
    args = parser.parse_args()

    env = DrMarioEnv(level=args.level, max_steps=args.max_steps)
    try:
        model = MaskablePPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=args.timesteps)
        model.save(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
