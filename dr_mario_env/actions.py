"""Action encoding shared by the Gym and dasyuridia backends."""

from __future__ import annotations
from collections.abc import Iterable
import numpy as np
from .dasyuridia import Placement, Position

ACTION_COUNT = 8 * 16 * 4 # column x row x rotations

def placement_to_action(placement: Placement) -> int:
    """Encode a placement using the simulator's 512-action layout."""

    x = placement.position.x
    y = placement.position.y
    if not 0 <= x < 8 or not 0 <= y < 16:
        raise ValueError("placement position is outside the 8x16 board")
    return y + (x << 4) + (placement.rotations << 7)

def action_to_placement(action: int) -> Placement:
    """Decode a simulator action index into a placement."""

    if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < ACTION_COUNT:
        raise ValueError(f"action must be an integer in [0, {ACTION_COUNT})")

    return Placement(
        rotations=(action >> 7) & 0x3,
        position=Position(x=(action >> 4) & 0x7, y=action & 0xf),
    )

def placement_mask(placements: Iterable[Placement]) -> np.ndarray:
    """Build a boolean mask for the placements reachable in a hi event."""

    mask = np.zeros(ACTION_COUNT, dtype=bool)
    for placement in placements:
        mask[placement_to_action(placement)] = True
    return mask


__all__ = [
    "ACTION_COUNT",
    "action_to_placement",
    "placement_mask",
    "placement_to_action",
]
