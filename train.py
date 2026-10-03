"""Minimal MaskablePPO training entry point."""

from __future__ import annotations

import argparse

from dr_mario_env import DrMarioEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--level-distribution", type=int, nargs='+', default=[0])
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--output", default="dr_mario_maskable_ppo")
    args = parser.parse_args()

    env = make_vec_env(lambda: DrMarioEnv(level_distribution=args.level_distribution, max_steps=args.max_steps), 8, vec_env_cls=SubprocVecEnv)
    try:
        model = MaskablePPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=args.timesteps)
        model.save(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
