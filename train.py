"""Minimal MaskablePPO training entry point."""

from __future__ import annotations

import argparse

from dr_mario_env import DrMarioEnv, DrMarioCnnPolicy
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--level-distribution", type=int, nargs='+', default=[0])
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--cnn-channels", type=int, default=16,
                        help="Channel count n for the shared n x 8 x 16 trunk")
    parser.add_argument("--cnn-depth", type=int, default=8,
                        help="Number of padded 3x3 conv layers in the shared trunk")
    parser.add_argument("--output", default="dr_mario_maskable_ppo")
    parser.add_argument("--input")
    parser.add_argument("--disable-cudnn", default=True)
    args = parser.parse_args()

    if args.disable_cudnn:
        import torch
        torch.backends.cudnn.enabled = False

    env = make_vec_env(lambda: DrMarioEnv(level_distribution=args.level_distribution, max_steps=args.max_steps), 8, vec_env_cls=SubprocVecEnv)
    if args.input is None:
        model = MaskablePPO(
            DrMarioCnnPolicy,
            env,
            verbose=1,
            policy_kwargs=dict(
                features_extractor_kwargs=dict(
                    channels=args.cnn_channels,
                    depth=args.cnn_depth,
                ),
            ),
        )
    else: model = MaskablePPO.load(args.input, env)

    try: model.learn(total_timesteps=args.timesteps)
    finally:
        try: model.save(args.output)
        finally: env.close()


if __name__ == "__main__":
    main()
