"""Types used by the :program:`dasyuridia-hi` interface.

This module deliberately does not start or communicate with the command-line
program yet. It only gives the wire format's playfield a useful Python shape.
Rows are indexed in game coordinates: ``rows[0]`` is the bottom of the board.
The JSON interface uses the opposite order, so decoding reverses its rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import subprocess
from typing import Any
from collections.abc import Iterator, Mapping, Sequence


class Color(str, Enum):
    """The three colors used by viruses and pill halves."""

    BLUE = "b"
    RED = "r"
    YELLOW = "y"


class Shape(str, Enum):
    """The shape/orientation of an occupied board cell."""

    VIRUS = "x"
    DISCONNECTED = "o"
    NORTH = "∩"
    SOUTH = "∪"
    EAST = "⊃"
    WEST = "⊂"


class Orientation(str, Enum):
    """The orientation of a pill in a placement description."""

    VERTICAL = "↕"
    HORIZONTAL = "↔"


class Confidence(str, Enum):
    """Whether an event is predicted or confirmed by the backend."""

    PREDICTED = "predicted"
    CONFIRMED = "confirmed"


@dataclass(frozen=True, slots=True)
class Empty:
    """An unoccupied board cell."""


@dataclass(frozen=True, slots=True)
class Occupied:
    """A board cell containing a colored virus or pill half."""

    color: Color
    shape: Shape


Cell = Empty | Occupied


@dataclass(frozen=True, slots=True)
class Position:
    """A board coordinate, with ``y=0`` at the bottom row."""

    x: int
    y: int

    @classmethod
    def from_json(cls, source: str | bytes | bytearray | Any) -> Position:
        """Parse a position from ``[x, y]`` JSON."""

        value = _json_value(source)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not _is_int(value[0])
            or not _is_int(value[1])
        ):
            raise ValueError("a position must be a [x, y] JSON array of integers")
        return cls(value[0], value[1])

    def to_json_value(self) -> list[int]:
        """Return the JSON-compatible ``[x, y]`` representation."""

        return [self.x, self.y]

    def to_json(self) -> str:
        """Return the position encoded as JSON."""

        return json.dumps(self.to_json_value())


@dataclass(frozen=True, slots=True)
class Placement:
    """A final pill placement from the high-level interface.

    ``rotations`` is the clockwise rotation count (0 through 3), and
    ``position`` is the final position of the pill's bottom-left half. The
    wire representation is ``[[x, y], rotations]``.
    """

    rotations: int
    position: Position

    def __post_init__(self) -> None:
        if not _is_int(self.rotations) or not 0 <= self.rotations <= 3:
            raise ValueError("rotations must be an integer from 0 through 3")

    @classmethod
    def from_json(cls, source: str | bytes | bytearray | Any) -> Placement:
        """Parse a placement from ``[[x, y], rotations]`` JSON."""

        value = _json_value(source)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("a placement must be a [[x, y], rotations] JSON array")
        position = Position.from_json(value[0])
        return cls(value[1], position)

    def to_json_value(self) -> list[Any]:
        """Return the JSON-compatible ``[[x, y], rotations]`` representation."""

        return [self.position.to_json_value(), self.rotations]

    def to_json(self) -> str:
        """Return the placement encoded as JSON."""

        return json.dumps(self.to_json_value())


@dataclass(frozen=True, slots=True)
class PillContent:
    """The orientation and colors of the two halves of a pill."""

    orientation: Orientation
    bottom_left_color: Color
    other_color: Color

    @classmethod
    def from_json(cls, source: str | bytes | bytearray | Any) -> PillContent:
        value = _json_value(source)
        return cls._from_json_value(value)

    @classmethod
    def _from_json_value(cls, value: Any) -> PillContent:
        if not isinstance(value, str) or len(value) != 3:
            raise ValueError("pill content must be a three-character JSON string")
        try:
            return cls(Orientation(value[0]), Color(value[1]), Color(value[2]))
        except ValueError as error:
            raise ValueError(f"invalid pill content {value!r}") from error

    def to_json_value(self) -> str:
        return self.orientation.value + self.bottom_left_color.value + self.other_color.value

    def to_json(self) -> str:
        return json.dumps(self.to_json_value(), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class PlacementInfo:
    """A reachable pill placement reported by ``dasyuridia-hi``."""

    content: PillContent
    path_cost: int
    placement: Placement
    cleanup_cost: int | None = None
    result: Board | None = None

    def __post_init__(self) -> None:
        if not _is_int(self.path_cost) or self.path_cost < 0:
            raise ValueError("path_cost must be a non-negative integer")
        if self.cleanup_cost is not None and (
            not _is_int(self.cleanup_cost) or self.cleanup_cost < 0
        ):
            raise ValueError("cleanup_cost must be a non-negative integer")

    @classmethod
    def from_json(cls, source: str | bytes | bytearray | Any) -> PlacementInfo:
        value = _json_value(source)
        if not isinstance(value, dict):
            raise ValueError("placement info must be a JSON object")
        required = {"content", "path_cost", "placement"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"placement info is missing {sorted(missing)!r}")
        return cls(
            content=PillContent._from_json_value(value["content"]),
            path_cost=value["path_cost"],
            placement=Placement.from_json(value["placement"]),
            cleanup_cost=value.get("cleanup_cost"),
            result=Board.from_json(value["result"]) if "result" in value else None,
        )

    def to_json_value(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "content": self.content.to_json_value(),
            "path_cost": self.path_cost,
            "placement": self.placement.to_json_value(),
        }
        if self.cleanup_cost is not None:
            value["cleanup_cost"] = self.cleanup_cost
        if self.result is not None:
            value["result"] = self.result.to_json_value()
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_json_value(), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Lookahead:
    """The colors of the next pill, from left/bottom-left to right/other."""

    left_color: Color
    right_color: Color

    @classmethod
    def from_json(cls, source: str | bytes | bytearray | Any) -> Lookahead:
        value = _json_value(source)
        return cls._from_json_value(value)

    @classmethod
    def _from_json_value(cls, value: Any) -> Lookahead:
        if not isinstance(value, str) or len(value) != 2:
            raise ValueError("lookahead must be a two-character JSON string")
        try:
            return cls(Color(value[0]), Color(value[1]))
        except ValueError as error:
            raise ValueError(f"invalid lookahead {value!r}") from error

    def to_json_value(self) -> str:
        return self.left_color.value + self.right_color.value


@dataclass(frozen=True, slots=True)
class Event:
    """A game-state event emitted by ``dasyuridia-hi``."""

    board: Board
    confidence: Confidence
    deadline: datetime
    gravity: int
    lookahead: Lookahead
    microseconds_until_deadline: int
    pills_used: int
    reachable: tuple[PlacementInfo, ...]
    tag: int

    def __post_init__(self) -> None:
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ValueError("deadline must be timezone-aware")
        for name in (
            "gravity",
            "microseconds_until_deadline",
            "pills_used",
            "tag",
        ):
            if not _is_int(getattr(self, name)):
                raise ValueError(f"{name} must be an integer")

    @classmethod
    def from_json(cls, source: str | bytes | bytearray | Any) -> Event:
        value = _json_value(source)
        if not isinstance(value, dict):
            raise ValueError("an event must be a JSON object")
        required = {
            "board",
            "confidence",
            "deadline",
            "gravity",
            "lookahead",
            "microseconds_until_deadline",
            "pills_used",
            "reachable",
            "tag",
        }
        missing = required - value.keys()
        if missing:
            raise ValueError(f"event is missing {sorted(missing)!r}")
        try:
            confidence = Confidence(value["confidence"])
        except ValueError as error:
            raise ValueError(f"invalid confidence {value['confidence']!r}") from error
        try:
            deadline = _parse_deadline(value["deadline"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid event deadline") from error
        reachable = value["reachable"]
        if not isinstance(reachable, list):
            raise ValueError("event reachable must be a JSON array")
        return cls(
            board=Board._from_json_value(value["board"]),
            confidence=confidence,
            deadline=deadline,
            gravity=value["gravity"],
            lookahead=Lookahead._from_json_value(value["lookahead"]),
            microseconds_until_deadline=value["microseconds_until_deadline"],
            pills_used=value["pills_used"],
            reachable=tuple(PlacementInfo.from_json(item) for item in reachable),
            tag=value["tag"],
        )

    def to_json_value(self) -> dict[str, Any]:
        return {
            "board": self.board.to_json_value(),
            "confidence": self.confidence.value,
            "deadline": _format_deadline(self.deadline),
            "gravity": self.gravity,
            "lookahead": self.lookahead.to_json_value(),
            "microseconds_until_deadline": self.microseconds_until_deadline,
            "pills_used": self.pills_used,
            "reachable": [item.to_json_value() for item in self.reachable],
            "tag": self.tag,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_value(), ensure_ascii=False)


class DasyuridiaJSONEncoder(json.JSONEncoder):
    """JSON encoder for dasyuridia protocol value types."""

    def default(self, object: Any) -> Any:
        if isinstance(object, (Position, Placement, PillContent, PlacementInfo, Lookahead, Event)):
            return object.to_json_value()
        if isinstance(object, Board):
            return object.to_json_value()
        return super().default(object)


class Dasyuridia(Iterator[Event]):
    """A JSON-lines client for the ``dasyuridia-hi`` executable.

    Iterating over an instance yields :class:`Event` objects as they arrive.
    The executable is started on construction and can be replaced in tests by
    passing a different command.
    """

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._process = subprocess.Popen(
            list(command) if command is not None else ["dasyuridia-hi"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=dict(env) if env is not None else None,
        )

    def __iter__(self) -> Dasyuridia:
        return self

    def __next__(self) -> Event:
        if self._process.stdout is None:
            raise RuntimeError("dasyuridia-hi stdout is unavailable")
        line = self._process.stdout.readline()
        if line:
            try:
                return Event.from_json(line)
            except ValueError as error:
                raise RuntimeError(f"dasyuridia-hi emitted invalid event: {line!r}") from error

        code = self._process.poll()
        if code is None:
            raise RuntimeError("dasyuridia-hi closed stdout unexpectedly")
        raise StopIteration

    def place(self, tag: int, placement: Placement) -> None:
        """Tell the process which reachable placement to select."""

        if not _is_int(tag):
            raise TypeError("tag must be an integer")
        self._send({"tag": tag, "placement": placement.to_json_value()})

    def include_results(self, include: bool) -> None:
        """Request or suppress speculative cleanup and result fields."""

        if not isinstance(include, bool):
            raise TypeError("include must be a bool")
        self._send({"results": include})

    def _send(self, command: dict[str, Any]) -> None:
        if self._process.stdin is None:
            raise RuntimeError("dasyuridia-hi stdin is unavailable")
        if self._process.poll() is not None:
            raise RuntimeError("dasyuridia-hi has exited")
        self._process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

    def close(self) -> None:
        """Close the subprocess and release its pipes."""

        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    def __enter__(self) -> Dasyuridia:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class Board:
    """A rectangular Dr. Mario playfield.

    ``rows`` are ordered from bottom to top, so ``rows[y]`` uses the game's
    coordinate system. A board is immutable so a received game state can
    safely be passed between an agent and any decision-making code.
    """

    rows: tuple[tuple[Cell, ...], ...]

    def __post_init__(self) -> None:
        if not self.rows:
            return
        width = len(self.rows[0])
        if any(len(row) != width for row in self.rows):
            raise ValueError("board rows must all have the same width")

    @property
    def width(self) -> int:
        """The number of cells in each row."""

        return len(self.rows[0]) if self.rows else 0

    @property
    def height(self) -> int:
        """The number of rows."""

        return len(self.rows)

    @classmethod
    def empty(cls, width: int = 8, height: int = 16) -> Board:
        """Construct an empty board with the standard playfield dimensions."""

        if width < 0 or height < 0:
            raise ValueError("board dimensions must not be negative")
        empty_row = (Empty(),) * width
        return cls((empty_row,) * height)

    def to_json_value(self) -> list[Any]:
        """Return the board in the protocol's top-to-bottom JSON format."""

        rows = [
            "".join(_cell_to_json_text(cell) for cell in row)
            for row in reversed(self.rows)
        ]
        return [self.width, rows]

    def to_json(self) -> str:
        """Return the board encoded as JSON."""

        return json.dumps(self.to_json_value(), ensure_ascii=False)

    @classmethod
    def from_json(cls, source: str | bytes | bytearray | Any) -> Board:
        """Decode a board from the high-level interface's JSON representation.

        ``source`` may be JSON text, bytes, or the value returned by
        :func:`json.loads`. The wire representation is ``[width, rows]``;
        each occupied cell is represented by two characters, its color and
        shape, while an empty cell is represented by two spaces.
        """

        if isinstance(source, (str, bytes, bytearray)):
            try:
                source = json.loads(source)
            except json.JSONDecodeError as error:
                raise ValueError("invalid board JSON") from error
        return cls._from_json_value(source)

    @classmethod
    def _from_json_value(cls, value: Any) -> Board:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("a board must be a [width, rows] JSON array")

        width, rows = value
        if not isinstance(width, int) or isinstance(width, bool) or width < 0:
            raise ValueError("board width must be a non-negative integer")
        if not isinstance(rows, list):
            raise ValueError("board rows must be a JSON array")

        parsed_rows: list[tuple[Cell, ...]] = []
        for row_number, row in enumerate(rows):
            if not isinstance(row, str) or len(row) != width * 2:
                raise ValueError(
                    f"row {row_number} must contain exactly {width * 2} characters"
                )
            parsed_rows.append(
                tuple(
                    _cell_from_json_text(row[column : column + 2], row_number, column // 2)
                    for column in range(0, len(row), 2)
                )
            )

        # JSON rows are top-to-bottom; Board rows are indexed bottom-to-top.
        return cls(tuple(reversed(parsed_rows)))


def _cell_from_json_text(text: str, row: int, column: int) -> Cell:
    if text == "  ":
        return Empty()
    if len(text) != 2:
        raise ValueError(f"invalid cell at ({column}, {row})")
    try:
        return Occupied(Color(text[0]), Shape(text[1]))
    except ValueError as error:
        raise ValueError(f"invalid cell {text!r} at ({column}, {row})") from error


def _cell_to_json_text(cell: Cell) -> str:
    if isinstance(cell, Empty):
        return "  "
    return cell.color.value + cell.shape.value


def _json_value(source: str | bytes | bytearray | Any) -> Any:
    if isinstance(source, (str, bytes, bytearray)):
        try:
            return json.loads(source)
        except json.JSONDecodeError as error:
            raise ValueError("invalid JSON") from error
    return source


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_deadline(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("deadline must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("deadline must include a timezone")
    return parsed


def _format_deadline(value: datetime) -> str:
    formatted = value.astimezone(timezone.utc).isoformat(timespec="microseconds")
    return formatted.replace("+00:00", "Z")


__all__ = [
    "Board",
    "Cell",
    "Color",
    "Confidence",
    "Dasyuridia",
    "Event",
    "Empty",
    "Lookahead",
    "Orientation",
    "Occupied",
    "PillContent",
    "Placement",
    "PlacementInfo",
    "Position",
    "Shape",
    "DasyuridiaJSONEncoder",
]
