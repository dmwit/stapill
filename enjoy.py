import argparse
import time

from dr_mario_env import DrMarioEnv
from sb3_contrib import MaskablePPO

def pptime(frames):
    frames = int(frames)
    leftover_frames = frames % 60
    seconds = frames // 60
    leftover_seconds = seconds % 60
    minutes = seconds // 60
    leftover_minutes = minutes % 60
    hours = minutes // 60

    # drop the 0.
    ppframes = f'{leftover_frames / 60:.3f}ms'

    result = f'{hours}h' if hours else ''
    if minutes: result = f'{result}{leftover_minutes:02d}m'
    if seconds: result = f'{result}{leftover_seconds:02d}s{ppframes[2:]}'
    else: result = ppframes

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level-distribution", type=int, nargs='+', default=[0])
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--input", default="dr_mario_maskable_ppo")
    parser.add_argument("--disable-cudnn", default=True)
    args = parser.parse_args()

    if args.disable_cudnn:
        import torch
        torch.backends.cudnn.enabled = False

    env = DrMarioEnv(level_distribution=args.level_distribution, render_mode="human")
    model = MaskablePPO.load(args.input, env)
    terminated = True

    while True:
        if terminated:
            obs, _ = env.reset()

            try:
                print(f"total steps before termination: {steps}")
                print(f"total frames before termination: {frames} ({pptime(frames)})")
                print('='*30)
            except NameError: pass

            cumulative_discounted_reward = 0
            cumulative_absolute_reward = 0
            cumulative_gamma = 1
            steps = 0
            frames = 0

        action, _ = model.predict(obs, action_masks=env.action_masks())
        obs, reward, terminated, _truncated, _info = env.step(action)
        env.render()

        cumulative_discounted_reward += cumulative_gamma * reward
        cumulative_absolute_reward += reward
        cumulative_gamma *= model.gamma
        steps += 1
        frames += env._last_response.reward["frames"]
        print(f"cumulative discounted reward so far this episode: {cumulative_discounted_reward}")
        print(f"cumulative non-discounted reward so far this episode: {cumulative_absolute_reward}")
        print(f"non-discounted reward for this step: {reward}")
        print(f"\t{env._last_response.reward}")
        print('\n'*3)

        time.sleep(1/args.fps)
