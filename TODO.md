# Reinforcement Learning TODO

Build a Gymnasium environment for training a Stable-Baselines3-compatible agent
to play Dr. Mario using the pure Haskell model in `maryodel/haskell`.

## First milestone

- [ ] Add a persistent single-player game simulator around the existing Haskell model.
- [ ] Use a fixed starting level initially.
- [ ] Generate seeded boards and pill sequences with the existing NES-compatible RNG.
- [ ] Advance one environment step per pill placement, including placement, cleanup, gravity, and generation of the next pill.
- [ ] Detect level completion and loss.
- [ ] Track simulated frames for pill movement and cleanup/gravity.

## Action space

- [ ] Expose a fixed discrete action space for `(x, y, orientation/color ordering)`.
- [ ] Start with `8 * 16 * 4 = 512` actions.
- [ ] Use the existing pathfinding/model logic to determine reachable placements.
- [ ] Implement an action mask for legal actions.
- [ ] Use `MaskablePPO` during training.
- [ ] If masking is not used, safely replace an invalid action with a uniformly random valid action.
- [ ] Record the requested action, executed action, and whether replacement occurred in `info`.

## Observation space

- [ ] Use a fixed-shape numeric observation suitable for standard SB3 policies.
- [ ] Include board contents and the current pill's two colors. Do not include the lookahead colors.
- [ ] Include any time/state scalars needed by the reward or value function.
- [ ] Begin with pill information supplied to the network at the input; consider deeper conditioning only if needed.

## Rewards

- [ ] Make reward components independently configurable.
- [ ] Reward newly cleared non-virus pieces.
- [ ] Reward useful horizontal and vertical same-color alignments without repeatedly rewarding unchanged structure.
- [ ] Reward viruses cleared.
- [ ] Give the largest positive reward for clearing the level.
- [ ] Penalize losing heavily.
- [ ] Penalize total simulated transition time, including pill movement and cleanup/gravity frames.
- [ ] Report every reward component and frame count in `info`.
- [ ] Leave an extension point for rewards supplied by an existing Haskell AI/evaluator.

Use ordinary action-based discounting initially. The time penalty is a reward
term, not frame-based discounting.

## Python/SB3 integration

- [ ] Add a Python package/environment implementing the Gymnasium `reset` and `step` APIs.
- [ ] Implement deterministic `reset(seed=...)` behavior.
- [ ] Add `action_masks()` for `MaskablePPO` and vectorized environments.
- [ ] Implement `render()` or a board-debugging representation.
- [ ] Add `close()` and clean shutdown behavior.
- [ ] Add a minimal training script using `MaskablePPO`.
- [ ] Add vectorized environment support after the single environment works.

## Verification

- [ ] Test board transitions against known Haskell model examples.
- [ ] Test all action encodings and legal-action masks.
- [ ] Test invalid-action replacement and seeded reproducibility.
- [ ] Test win, loss, cleanup, gravity, and episode truncation behavior.
- [ ] Run Gymnasium/SB3 environment checks, accounting for checks that sample actions without masks.
- [ ] Confirm that a random agent and a simple heuristic agent can complete environment episodes.
- [ ] Benchmark steps per second before adding multiprocessing.

## Later

- [ ] Add variable virus levels and speed progression.
- [ ] Investigate exact frame-based discounting if action-based discounting is insufficient.
- [ ] Add two-player mechanics, garbage, and self-play.
- [ ] Compare architecture 1 with deeper pill conditioning and color-pair-specific output heads.
