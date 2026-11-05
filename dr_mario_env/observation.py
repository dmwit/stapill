"""Policy observation encoding shared by the Gym and dasyuridia backends."""

from __future__ import annotations
from collections.abc import Mapping
import numpy as np

from .dasyuridia import Board, Color, Empty, Event, Occupied, Shape
from .simulator import SimulatorReply

OBSERVATION_SHAPE = (14, 8, 16)
_SHAPE_INDEX = \
    { Shape.VIRUS: 0
    , Shape.DISCONNECTED: 1
    , Shape.NORTH: 1
    , Shape.SOUTH: 1
    , Shape.EAST: 2
    , Shape.WEST: 3
    }

def encode_simulator_reply(reply: SimulatorReply) -> np.ndarray:
    """Encode the existing simulator response exactly as ``DrMarioEnv`` did."""

    observation = reply.observation
    return _assemble(
        observation.color,
        observation.shape,
        observation.other,
    )

def encode_event(event: Event) -> np.ndarray:
    """Encode a dasyuridia-hi event for the trained policy."""

    color_order = list(dict.fromkeys([]
        + [event.lookahead.left, event.lookahead.right]
        + [Color.BLUE, Color.RED, Color.YELLOW]
        ))
    color_index = {color: i for i, color in enumerate(color_order)}
    colors = np.zeros((4, event.board.width, event.board.height), dtype=np.float32)
    shapes = np.zeros((5, event.board.width, event.board.height), dtype=np.float32)

    for y, row in enumerate(event.board.rows):
        for x, cell in enumerate(row):
            if isinstance(cell, Empty):
                colors[0, x, y] = 1
                shapes[0, x, y] = 1
            else:
                assert isinstance(cell, Occupied)
                colors[1 + color_index[cell.color], x, y] = 1
                shapes[1 + _SHAPE_INDEX[cell.shape], x, y] = 1

    other = {
        "double": float(event.lookahead.left == event.lookahead.right),
        "row/frame": 1.0 / event.gravity,
        "superframe/row": event.gravity / 40.0,
    }
    return _assemble(colors, shapes, other)

def encode_none() -> np.ndarray:
    """Make a meaningless but correctly-shaped input to the policy network."""
    colors = np.zeros((4,8,16), dtype=np.float32)
    shapes = np.zeros((5,8,16), dtype=np.float32)
    other = {
        "double": 0,
        "row/frame": 0,
        "superframe/row": 0,
    }
    return _assemble(colors, shapes, other)

def _assemble(
    colors: np.ndarray,
    shapes: np.ndarray,
    other: Mapping[str, float],
) -> np.ndarray:
    board_shape = colors.shape[1:]
    other_planes = np.broadcast_to(
        np.array([[[value]] for _, value in sorted(other.items())], dtype=np.float32),
        (len(other),) + board_shape,
    )
    return np.concatenate([
        colors,
        shapes,
        other_planes,
        _coordinate_plane(board_shape, 0),
        _coordinate_plane(board_shape, 1),
    ])

def _coordinate_plane(board_shape: tuple[int, int], axis: int) -> np.ndarray:
    result_shape = [1 for _ in board_shape]
    result_shape[axis] = board_shape[axis]
    result_shape = [1] + result_shape
    return np.broadcast_to(
        np.reshape(
            np.linspace(0, 1, board_shape[axis], endpoint=False, dtype=np.float32),
            result_shape,
        ),
        (1,) + board_shape,
    )

__all__ = ["OBSERVATION_SHAPE", "encode_event", "encode_simulator_reply"]
