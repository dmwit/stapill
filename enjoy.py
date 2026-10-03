import argparse
import time

from dr_mario_env import DrMarioEnv
from sb3_contrib import MaskablePPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level-distribution", type=int, nargs='+', default=[0])
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--input", default="dr_mario_maskable_ppo")
    args = parser.parse_args()

    env = DrMarioEnv(level_distribution=args.level_distribution, render_mode="human")
    model = MaskablePPO.load(args.input, env)
    terminated = True
    while True:
        if terminated: obs, _ = env.reset()
        action, _ = model.predict(obs)
        obs, _, terminated, _, _ = env.step(action)
        env.render()
        time.sleep(1/args.fps)
