# Reinforcement Learning TODO

Build a Gymnasium environment for training a Stable-Baselines3-compatible agent
to play Dr. Mario using a Haskell simulator based on `maryodel/haskell`.
The Haskell simulator is the authoritative game-transition engine; Python
handles the Gymnasium and Stable-Baselines3 integration.

## First milestone

- [x] Add a persistent single-player Haskell simulator around the existing model.
- [x] Define a stable machine-readable request/response interface between Python and the Haskell simulator.
- [x] Use a fixed starting level initially.
- [x] Generate seeded boards and pill sequences with the existing NES-compatible RNG.
- [x] Advance one environment step per pill placement, including placement, cleanup, gravity, and generation of the next pill.
- [x] Detect level completion and loss.
- [x] Track simulated frames for pill movement and cleanup/gravity.

## Action space

- [x] Expose a fixed discrete action space for `(x, y, orientation/color ordering)`.
- [x] Start with `8 * 16 * 4 = 512` actions.
- [x] Use the existing pathfinding/model logic to determine reachable placements.
- [x] Implement an action mask for legal actions.
- [x] Use `MaskablePPO` during training.
- [x] If masking is not used, safely replace an invalid action with a uniformly random valid action.
- [x] Record the requested action, executed action, and whether replacement occurred in `info`.

## Observation space

- [x] Use a fixed-shape numeric observation suitable for standard SB3 policies.
- [x] Include board contents and the current pill's two colors. Do not include the lookahead colors.
- [x] Include current gravity as normalized frames-per-row and rows-per-frame features.
- [x] Begin with pill information supplied to the network at the input; consider deeper conditioning only if needed.

## Rewards

- [x] Make reward components independently configurable.
- [x] Reward newly cleared non-virus pieces.
- [x] Reward useful horizontal and vertical same-color alignments without repeatedly rewarding unchanged structure.
- [x] Reward viruses cleared.
- [x] Give the largest positive reward for clearing the level.
- [x] Penalize losing heavily.
- [x] Penalize total simulated transition time, including pill movement and cleanup/gravity frames.
- [x] Report every reward component and frame count in `info`.
- [x] Leave an extension point for rewards supplied by an existing Haskell AI/evaluator.

Use ordinary action-based discounting initially. The time penalty is a reward
term, not frame-based discounting.

## Python/SB3 integration

- [x] Add a Python package/environment implementing the Gymnasium `reset` and `step` APIs through the Haskell simulator interface.
- [x] Implement deterministic `reset(seed=...)` behavior.
- [x] Add `action_masks()` for `MaskablePPO` and vectorized environments.
- [x] Implement `render()` or a board-debugging representation.
- [x] Add `close()` and clean shutdown behavior.
- [x] Add a minimal training script using `MaskablePPO`.
- [ ] Add vectorized environment support after the single environment works.

## Verification

- [x] Test the Haskell simulator's transition contract with deterministic fixtures.
- [ ] Test Python/Haskell request and response serialization, including errors and episode termination.
- [ ] Test all action encodings and legal-action masks.
- [x] Test invalid-action replacement and seeded reproducibility.
- [ ] Test win, loss, cleanup, gravity, and episode truncation behavior.
- [x] Run Gymnasium/SB3 environment checks, accounting for checks that sample actions without masks.
- [ ] Confirm that a random agent and a simple heuristic agent can complete environment episodes.
- [ ] Benchmark steps per second before adding multiprocessing.

## Later

- [x] Add variable virus levels and speed progression.
- [ ] Investigate exact frame-based discounting if action-based discounting is insufficient.
- [ ] Add two-player mechanics, garbage, and self-play.
- [ ] Compare architecture 1 with deeper pill conditioning and color-pair-specific output heads.
- [ ] Color-permutation training augmentation.
- [ ] Add a convolutional net in parallel to the existing net architecture.
- [ ] Experiment with different reward weightings.
- [x] Color invariance: make observations be lookahead-relative, with a bit for double pills.
- [ ] CNN-ify the policy/value network.
- [ ] Maybe pathfinding is a speed bottleneck. We could schedule it at action time, but only wait for it to finish at action-mask query time. Needs benchmark.
