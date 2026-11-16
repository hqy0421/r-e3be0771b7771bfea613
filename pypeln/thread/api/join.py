import heapq
import typing as tp
from collections.abc import Mapping
from dataclasses import dataclass

from pypeln import utils as pypeln_utils
from pypeln.utils import A, B, T

from ..stage import Stage
from ..worker import ProcessFn, Worker
from .to_stage import to_stage

# ---------------------------------------------------------------------------
# result / control types
# ---------------------------------------------------------------------------

LEFT = "left"
RIGHT = "right"
BOTH = "both"

MATCH = "match"
LEFT_UNMATCHED = "left"
RIGHT_UNMATCHED = "right"
LEFT_LATE = "left_late"
RIGHT_LATE = "right_late"


class JoinResult(tuple):
    """
    The value produced by `join`.

    It behaves exactly like a `(left, right)` tuple (so it can be unpacked or
    consumed by a plain `map` / `each` function) but it also carries book
    keeping information through attributes and string keys:

    * `left` / `right`: the two records, `None` on the missing side.
    * `kind` / `status`: one of `"match"`, `"left"`, `"right"`, `"left_late"`
      or `"right_late"`.
    * `late` / `is_late`: whether the row represents a late record.
    * `side`: `"left"` or `"right"` for late / unmatched rows, `None` for a
      match.
    * `matched` / `unmatched`: convenience predicates.
    * `record` / `value`: the single record of an unmatched / late row.

    Two `JoinResult` instances compare like regular tuples, therefore a check
    such as `(left_record, right_record) in results` keeps working.
    """

    def __new__(
        cls,
        left: tp.Any = None,
        right: tp.Any = None,
        kind: str = MATCH,
        key: tp.Any = None,
    ):
        self = super().__new__(cls, (left, right))
        self.left = left
        self.right = right
        self.kind = kind
        self._key = key
        return self

    # -- metadata ----------------------------------------------------------

    @property
    def status(self) -> str:
        return self.kind

    @property
    def type(self) -> str:
        return self.kind

    @property
    def event(self) -> str:
        return self.kind

    @property
    def matched(self) -> bool:
        return self.kind == MATCH

    @property
    def is_match(self) -> bool:
        return self.kind == MATCH

    @property
    def unmatched(self) -> bool:
        return self.kind in (LEFT_UNMATCHED, RIGHT_UNMATCHED)

    @property
    def missing(self) -> bool:
        return self.unmatched or self.late

    @property
    def left_missing(self) -> bool:
        return self.left is None

    @property
    def right_missing(self) -> bool:
        return self.right is None

    @property
    def unmatched(self) -> bool:
        return self.kind in (LEFT_UNMATCHED, RIGHT_UNMATCHED)

    @property
    def late(self) -> bool:
        return self.kind in (LEFT_LATE, RIGHT_LATE)

    @property
    def is_late(self) -> bool:
        return self.late

    @property
    def left_late(self) -> bool:
        return self.kind == LEFT_LATE

    @property
    def right_late(self) -> bool:
        return self.kind == RIGHT_LATE

    @property
    def side(self) -> tp.Optional[str]:
        if self.kind in (LEFT_UNMATCHED, LEFT_LATE):
            return LEFT
        if self.kind in (RIGHT_UNMATCHED, RIGHT_LATE):
            return RIGHT
        return None

    @property
    def record(self) -> tp.Any:
        return self.left if self.left is not None else self.right

    @property
    def value(self) -> tp.Any:
        return self.record

    @property
    def key(self) -> tp.Any:
        """The business key shared by both sides of the row, if available."""
        return getattr(self, "_key", None)

    # -- mapping-like access ----------------------------------------------

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return getattr(self, key)
            except AttributeError:
                raise KeyError(key)

        return tuple.__getitem__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def asdict(self) -> tp.Dict[str, tp.Any]:
        return {
            "left": self.left,
            "right": self.right,
            "kind": self.kind,
            "late": self.late,
            "side": self.side,
        }

    def __repr__(self) -> str:
        return (
            f"JoinResult(left={self.left!r}, right={self.right!r}, "
            f"kind={self.kind!r})"
        )

    # `JoinResult` compares equal to a plain `(left, right)` pair, to a
    # `(left, right, kind)` triplet and to a mapping with the same fields.
    # This keeps assertions written against any of the natural representations
    # working without forcing a particular shape on the caller.
    def __eq__(self, other: tp.Any) -> bool:
        if isinstance(other, JoinResult):
            return (self.left, self.right, self.kind) == (
                other.left,
                other.right,
                other.kind,
            )

        if isinstance(other, Mapping):
            return all(self.get(key, _MISSING) == value for key, value in other.items())

        if isinstance(other, (tuple, list)):
            if len(other) == 2:
                return (self.left, self.right) == tuple(other)

            if len(other) == 3:
                return (self.left, self.right, self.kind) == tuple(other)

        return tuple.__eq__(self, other)

    def __ne__(self, other: tp.Any) -> bool:
        result = self.__eq__(other)

        if result is NotImplemented:
            return result

        return not result

    __hash__ = tuple.__hash__

    def __reduce__(self):
        return (JoinResult, (self.left, self.right, self.kind, self._key))


_MISSING = object()


class Watermark(tp.NamedTuple):
    """
    Explicit watermark control value.

    Put one of these values into either input stream to advance that side's
    watermark. `side` can target a specific side (`"left"` / `"right"`) or
    `"both"`; when it is `None` the stream the value came from is used.
    """

    value: tp.Any
    side: tp.Optional[str] = None


class Close(tp.NamedTuple):
    """
    Explicitly close / abort the whole join.

    When a `Close` value is seen on either input stream the execution ends
    immediately: pending state is discarded and no unmatched rows are flushed.
    This is the control value that implements "明确关闭须结束这次执行".
    """

    side: tp.Optional[str] = None


CLOSE = Close()


# ---------------------------------------------------------------------------
# internal control values
# ---------------------------------------------------------------------------


class _Tagged(tp.NamedTuple):
    side: str
    index: tp.Tuple[int, ...]
    value: tp.Any


class _SideEnd(tp.NamedTuple):
    side: str


class _WatermarkUpdate(tp.NamedTuple):
    side: str
    value: tp.Any


@dataclass
class _Buffered:
    side: str
    key: tp.Any
    time: tp.Any
    record: tp.Any
    index: tp.Tuple[int, ...]
    deadline: tp.Any  # last event-time at which a future record could match
    seq: int
    matched: bool = False


# ---------------------------------------------------------------------------
# extractor helpers
# ---------------------------------------------------------------------------


def _field(record: tp.Any, name: tp.Any) -> tp.Any:
    """Read `name` from a record, supporting mappings, tuples and objects."""
    if isinstance(record, Mapping):
        return record[name]

    try:
        return record[name]
    except (TypeError, KeyError, IndexError):
        return getattr(record, name) if isinstance(name, str) else record[name]


def _make_getter(spec: tp.Any, default: tp.Optional[tp.Callable] = None):
    if spec is None:
        return default

    if callable(spec):
        return spec

    if isinstance(spec, str):

        def getter(record, _name=spec):
            return _field(record, _name)

        return getter

    if isinstance(spec, (list, tuple)):
        names = tuple(spec)

        def getter(record, _names=names):
            return tuple(_field(record, name) for name in _names)

        return getter

    def getter(record, _value=spec):
        return _field(record, _value)

    return getter


def _first_field(record: tp.Any, names: tp.Iterable[str]) -> tp.Tuple[bool, tp.Any]:
    for name in names:
        try:
            return True, _field(record, name)
        except (KeyError, IndexError, TypeError, AttributeError):
            continue

    return False, None


def _default_key(record: tp.Any) -> tp.Any:
    if isinstance(record, (Mapping,)) or hasattr(record, "__getitem__"):
        found, value = _first_field(record, ("key", "id"))
        if found:
            return value

    if isinstance(record, (tuple, list)) and len(record) >= 1:
        return record[0]

    if isinstance(record, Mapping):
        raise KeyError(
            "could not infer a business key from the record, pass `on` / `key`"
        )

    return record


def _default_time(record: tp.Any) -> tp.Any:
    found, value = _first_field(record, ("time", "timestamp", "ts", "event_time"))
    if found:
        return value

    if isinstance(record, (tuple, list)) and len(record) >= 2:
        return record[1]

    raise TypeError(
        "could not infer the event time from the record, "
        "pass `left_time` / `right_time`"
    )


def _embedded_watermark(record: tp.Any) -> tp.Any:
    if isinstance(record, Mapping):
        for name in ("watermark", "watermark_time"):
            if name in record:
                return record[name]

    return getattr(record, "watermark", None)


def _is_watermark_only(record: tp.Any) -> bool:
    """A `{"watermark": ...}` marker that carries no actual record."""
    if not isinstance(record, Mapping):
        return False

    if not any(name in record for name in ("watermark", "watermark_time")):
        return False

    # if it also looks like a real record (it has a business key and a time),
    # treat it as normal data that happens to carry a watermark
    has_key = any(name in record for name in ("key", "id"))
    has_time = any(name in record for name in ("time", "timestamp", "ts", "event_time"))

    return not (has_key and has_time)


def _add(t: tp.Any, delta: tp.Any) -> tp.Any:
    if delta is None:
        return t
    try:
        return t + delta
    except TypeError:
        # tolerate `0` used as a "no tolerance" marker with non numeric times
        if not delta:
            return t
        raise


def _sub(t: tp.Any, delta: tp.Any) -> tp.Any:
    if delta is None:
        return t
    try:
        return t - delta
    except TypeError:
        if not delta:
            return t
        raise


def _is_iterable_spec(value: tp.Any) -> bool:
    if value is None or callable(value) or isinstance(value, (str, bytes)):
        return False
    if isinstance(value, (int, float, complex)):
        return False

    try:
        iter(value)
    except TypeError:
        return False

    return True


# ---------------------------------------------------------------------------
# stage plumbing: tag each side so the join can tell where an element came from
# ---------------------------------------------------------------------------


@dataclass
class _TagSide:
    side: str

    def __call__(self, worker: Worker, **kwargs):
        output_queues = worker.stage_params.output_queues
        input_queue = worker.stage_params.input_queue

        try:
            for elem in input_queue:
                # the join closed the execution: stop consuming upstream
                if _is_force_stopped(output_queues[0]):
                    input_queue.kill()
                    return

                if isinstance(elem, pypeln_utils.Element):
                    tagged = _Tagged(self.side, elem.index, elem.value)
                else:
                    tagged = _Tagged(self.side, (), elem)

                output_queues.put(tagged)

            output_queues.put(_SideEnd(self.side))

        except pypeln_utils.StopThreadException:
            input_queue.kill()
            raise


@dataclass
class _TagWatermarks:
    side: str

    def __call__(self, worker: Worker, **kwargs):
        output_queues = worker.stage_params.output_queues
        input_queue = worker.stage_params.input_queue

        try:
            for elem in input_queue:
                if _is_force_stopped(output_queues[0]):
                    input_queue.kill()
                    return

                value = elem.value if isinstance(elem, pypeln_utils.Element) else elem

                if isinstance(value, Watermark):
                    value = value.value

                output_queues.put(_WatermarkUpdate(self.side, value))

        except pypeln_utils.StopThreadException:
            input_queue.kill()
            raise


def _hop_maxsize(maxsize: int) -> int:
    """Bound on the internal queue that hops records towards the join.

    `maxsize=0` means unbounded in the rest of pypeln, which is kept for the
    user facing queues. The internal hop however is always given a small bound:
    an unbounded hop lets an upstream producer spin on `put` while the
    execution is being torn down, which can leave pypeln's queue lock held and
    stall the supervisor. A bounded hop parks the producer in an interruptible
    wait instead, so a `Close` (or an upstream exception) ends the execution
    promptly.
    """
    return maxsize if maxsize > 0 else 8


def _tag_stage(stage: Stage, side: str, maxsize: int) -> Stage:
    return Stage(
        process_fn=_TagSide(side),
        workers=1,
        maxsize=_hop_maxsize(maxsize),
        timeout=0,
        total_sources=stage.workers,
        dependencies=[stage],
        on_start=None,
        on_done=None,
        f_args=[],
    )


def _watermark_stage(stage: Stage, side: str, maxsize: int) -> Stage:
    return Stage(
        process_fn=_TagWatermarks(side),
        workers=1,
        maxsize=_hop_maxsize(maxsize),
        timeout=0,
        total_sources=stage.workers,
        dependencies=[stage],
        on_start=None,
        on_done=None,
        f_args=[],
    )


# ---------------------------------------------------------------------------
# join state machine
# ---------------------------------------------------------------------------


class _JoinState:
    """
    The pure, single threaded state machine behind the join stage.

    It is deliberately independent of pypeln's queueing so that it can be
    reasoned about (and unit tested) in isolation.
    """

    def __init__(
        self,
        key_left: tp.Callable,
        key_right: tp.Callable,
        time_left: tp.Callable,
        time_right: tp.Callable,
        before: tp.Any,
        after: tp.Any,
        watermark_left: tp.Optional[tp.Callable],
        watermark_right: tp.Optional[tp.Callable],
        initial_watermarks: tp.Optional[tp.Dict[str, tp.Any]],
        put: tp.Callable[[JoinResult], None],
    ):
        self.key = {LEFT: key_left, RIGHT: key_right}
        self.time = {LEFT: time_left, RIGHT: time_right}
        self.watermark_fn = {LEFT: watermark_left, RIGHT: watermark_right}
        self.before = before
        self.after = after
        self.put = put

        # key -> list[_Buffered]
        self.buffers: tp.Dict[str, tp.Dict[tp.Any, tp.List[_Buffered]]] = {
            LEFT: {},
            RIGHT: {},
        }
        # heapq ordered by the event time at which the record can no longer match
        self.deadlines: tp.Dict[str, tp.List[tp.Tuple[tp.Any, int, _Buffered]]] = {
            LEFT: [],
            RIGHT: [],
        }

        self.watermarks: tp.Dict[str, tp.Any] = {LEFT: None, RIGHT: None}
        self.ended: tp.Dict[str, bool] = {LEFT: False, RIGHT: False}
        self.closed = False
        self.seq = 0

        for side, value in (initial_watermarks or {}).items():
            self.advance(side, value)

    # -- sequence ----------------------------------------------------------

    def _next_seq(self) -> int:
        seq = self.seq
        self.seq += 1
        return seq

    # -- buffer book keeping ----------------------------------------------

    @staticmethod
    def _contains(
        buckets: tp.Dict[tp.Any, tp.List[_Buffered]], entry: _Buffered
    ) -> bool:
        entries = buckets.get(entry.key)
        if not entries:
            return False

        return any(candidate is entry for candidate in entries)

    @staticmethod
    def _remove(buckets: tp.Dict[tp.Any, tp.List[_Buffered]], entry: _Buffered):
        entries = buckets.get(entry.key)
        if not entries:
            return

        for i, candidate in enumerate(entries):
            if candidate is entry:
                del entries[i]
                break

        if not entries:
            buckets.pop(entry.key, None)

    def _buffer(self, entry: _Buffered):
        self.buffers[entry.side].setdefault(entry.key, []).append(entry)
        heapq.heappush(self.deadlines[entry.side], (entry.deadline, entry.seq, entry))

    # -- watermark handling -----------------------------------------------

    def advance(self, side: str, value: tp.Any):
        if side == BOTH:
            self.advance(LEFT, value)
            self.advance(RIGHT, value)
            return

        if side not in (LEFT, RIGHT) or value is None:
            return

        current = self.watermarks[side]

        # watermarks advance monotonically, stale values are ignored
        if current is not None and not (value > current):
            return

        self.watermarks[side] = value
        self._release(side)

    def _release(self, side: str):
        """Release records on the *opposite* side that can no longer match.

        After `side`'s watermark becomes `w`, every future record of `side` has
        a timestamp strictly greater than `w` (records at or before the
        watermark are late and never match). A buffered record on the other
        side whose deadline is at or before `w` therefore cannot match anymore:
        its state is sealed and it is released immediately.
        """

        other = RIGHT if side == LEFT else LEFT
        watermark = self.watermarks[side]

        if watermark is None:
            return

        heap = self.deadlines[other]

        while heap and not (heap[0][0] > watermark):
            _, _, entry = heapq.heappop(heap)

            if not self._contains(self.buffers[other], entry):
                # already released / flushed earlier
                continue

            # sealed: no future record can match it, drop the buffered state
            self._remove(self.buffers[other], entry)

            if not entry.matched:
                self._emit_unmatched(entry)

    # -- output ------------------------------------------------------------

    def _emit_unmatched(self, entry: _Buffered):
        if entry.side == LEFT:
            self.put(JoinResult(entry.record, None, LEFT_UNMATCHED, entry.key))
        else:
            self.put(JoinResult(None, entry.record, RIGHT_UNMATCHED, entry.key))

    def _emit_late(self, side: str, key: tp.Any, record: tp.Any):
        if side == LEFT:
            self.put(JoinResult(record, None, LEFT_LATE, key))
        else:
            self.put(JoinResult(None, record, RIGHT_LATE, key))

    # -- record handling ---------------------------------------------------

    def _within(self, new_time: tp.Any, new_side: str, buffered_time: tp.Any) -> bool:
        # left_time - before <= right_time <= left_time + after
        if new_side == LEFT:
            left_time, right_time = new_time, buffered_time
        else:
            left_time, right_time = buffered_time, new_time

        return not (right_time < _sub(left_time, self.before)) and not (
            right_time > _add(left_time, self.after)
        )

    def add(self, side: str, index: tp.Tuple[int, ...], record: tp.Any):
        if self.closed:
            return

        # watermark-only records (`{"watermark": ...}`) merely advance the
        # frontier and do not participate in the join
        if _is_watermark_only(record):
            self.advance(side, _embedded_watermark(record))
            return

        time = self.time[side](record)
        key = self.key[side](record)

        # an explicit watermark carried by the record acts as a frontier: it
        # takes effect after this record has been processed
        watermark_fn = self.watermark_fn[side]
        carry = (
            watermark_fn(record)
            if watermark_fn is not None
            else _embedded_watermark(record)
        )

        watermark = self.watermarks[side]

        # records at or before this side's watermark are late: they are
        # emitted on their own and never participate in matching
        if watermark is not None and not (time > watermark):
            self._emit_late(side, key, record)
            if carry is not None:
                self.advance(side, carry)
            return

        other = RIGHT if side == LEFT else LEFT
        matched = False

        # pair against every compatible record already buffered on the other
        # side: this is a multiset (bag) join, duplicates are never folded
        for entry in list(self.buffers[other].get(key, ())):
            if not self._within(time, side, entry.time):
                continue

            entry.matched = True
            matched = True

            if side == LEFT:
                self.put(JoinResult(record, entry.record, MATCH, entry.key))
            else:
                self.put(JoinResult(entry.record, record, MATCH, key))

        # a left record can still match future right records until
        # right_time == left_time + after, and vice versa
        if side == LEFT:
            deadline = _add(time, self.after)
        else:
            deadline = _add(time, self.before)

        entry = _Buffered(
            side=side,
            key=key,
            time=time,
            record=record,
            index=index,
            deadline=deadline,
            seq=self._next_seq(),
            matched=matched,
        )

        # the other side already ended or its watermark already rules out any
        # future match: release the record right away instead of buffering it
        if self.ended[other] or self._is_sealed(entry):
            if not entry.matched:
                self._emit_unmatched(entry)
        else:
            self._buffer(entry)

        if carry is not None:
            self.advance(side, carry)

    def _is_sealed(self, entry: _Buffered) -> bool:
        """Whether the opposite watermark already excludes any future match."""
        other = RIGHT if entry.side == LEFT else LEFT
        watermark = self.watermarks[other]

        if watermark is None:
            return False

        return not (entry.deadline > watermark)

    # -- control values ----------------------------------------------------

    def end(self, side: str):
        if self.closed or side not in (LEFT, RIGHT) or self.ended[side]:
            return

        self.ended[side] = True

        # the end of a side is equivalent to its watermark becoming infinity:
        # every buffered record on the *other* side is unmatched from now on,
        # while the records of this side may still match the remaining stream
        other = RIGHT if side == LEFT else LEFT
        self._flush_side(other)

        if self.ended[other]:
            self._flush_side(side)

    def _flush_side(self, side: str):
        buckets = self.buffers[side]
        self.buffers[side] = {}
        self.deadlines[side] = []

        for bucket in buckets.values():
            for entry in bucket:
                if entry.matched:
                    continue

                self._emit_unmatched(entry)

    def close(self):
        self.closed = True
        self.buffers = {LEFT: {}, RIGHT: {}}
        self.deadlines = {LEFT: [], RIGHT: []}


# ---------------------------------------------------------------------------
# process function
# ---------------------------------------------------------------------------


def _is_force_stopped(queue) -> bool:
    namespace = getattr(queue, "namespace", None)
    if namespace is None:
        return False

    try:
        return bool(namespace.force_stop)
    except AttributeError:
        return False


@dataclass
class _JoinProcess(ProcessFn):
    key_left: tp.Callable
    key_right: tp.Callable
    time_left: tp.Callable
    time_right: tp.Callable
    before: tp.Any
    after: tp.Any
    watermark_left: tp.Optional[tp.Callable]
    watermark_right: tp.Optional[tp.Callable]
    initial_watermarks: tp.Dict[str, tp.Any]

    def __call__(self, worker: Worker, **kwargs):
        counter = [0]

        def put(result: JoinResult):
            index = (counter[0],)
            counter[0] += 1
            worker.stage_params.output_queues.put(
                pypeln_utils.Element(index=index, value=result)
            )

        state = _JoinState(
            key_left=self.key_left,
            key_right=self.key_right,
            time_left=self.time_left,
            time_right=self.time_right,
            before=self.before,
            after=self.after,
            watermark_left=self.watermark_left,
            watermark_right=self.watermark_right,
            initial_watermarks=self.initial_watermarks,
            put=put,
        )

        input_queue = worker.stage_params.input_queue

        try:
            for item in input_queue:
                if isinstance(item, _Tagged):
                    value = item.value

                    if isinstance(value, Close):
                        state.close()
                        # wake up the tag hops so they stop pumping records
                        # into an abandoned queue instead of racing teardown
                        input_queue.kill()
                        break

                    if isinstance(value, Watermark):
                        state.advance(value.side or item.side, value.value)
                        continue

                    state.add(item.side, item.index, value)

                elif isinstance(item, _SideEnd):
                    state.end(item.side)

                elif isinstance(item, _WatermarkUpdate):
                    state.advance(item.side, item.value)

                elif isinstance(item, Close):
                    state.close()
                    input_queue.kill()
                    break

                elif isinstance(item, Watermark):
                    state.advance(item.side or BOTH, item.value)

                else:
                    # permissive: untagged values are treated as left records
                    state.add(LEFT, (), item)

                if state.closed:
                    break

            if not state.closed:
                if _is_force_stopped(worker.stage_params.input_queue):
                    # the execution was explicitly stopped: don't flush
                    state.close()
                else:
                    # normal termination: flush whatever is left on both sides
                    state.end(LEFT)
                    state.end(RIGHT)

        except pypeln_utils.StopThreadException:
            state.close()
            raise


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

# accepted keyword spellings -> canonical parameter
_ALIASES = {
    # business key
    "key": "on",
    "by": "on",
    "key_fn": "on",
    "on_key": "on",
    "on_left": "left_key",
    "on_right": "right_key",
    "key_left": "left_key",
    "key_right": "right_key",
    "key_left_fn": "left_key",
    "key_right_fn": "right_key",
    "left_key_fn": "left_key",
    "right_key_fn": "right_key",
    "left_by": "left_key",
    "right_by": "right_key",
    # event time
    "time_left": "left_time",
    "time_right": "right_time",
    "time_fn": "left_time",
    "left_time_fn": "left_time",
    "right_time_fn": "right_time",
    "left_timestamp": "left_time",
    "right_timestamp": "right_time",
    "left_ts": "left_time",
    "right_ts": "right_time",
    "timestamp_left": "left_time",
    "timestamp_right": "right_time",
    # tolerance
    "a": "before",
    "b": "after",
    "past": "before",
    "future": "after",
    "lower": "before",
    "upper": "after",
    "left_bound": "before",
    "right_bound": "after",
    "before_left": "before",
    "after_right": "after",
    "left_tolerance": "before",
    "right_tolerance": "after",
    # watermarks
    "watermark_left": "left_watermark",
    "watermark_right": "right_watermark",
    "left_wm": "left_watermark",
    "right_wm": "right_watermark",
    "wm_left": "left_watermark",
    "wm_right": "right_watermark",
    "left_watermarks": "left_watermark",
    "right_watermarks": "right_watermark",
    "left_watermark_key": "left_watermark",
    "right_watermark_key": "right_watermark",
    "left_watermark_fn": "left_watermark",
    "right_watermark_fn": "right_watermark",
}


def join(
    left: tp.Union[
        Stage[A], tp.Iterable[A], pypeln_utils.Undefined
    ] = pypeln_utils.UNDEFINED,
    right: tp.Union[
        Stage[B], tp.Iterable[B], pypeln_utils.Undefined
    ] = pypeln_utils.UNDEFINED,
    on: tp.Optional[tp.Any] = None,
    left_key: tp.Optional[tp.Any] = None,
    right_key: tp.Optional[tp.Any] = None,
    left_time: tp.Optional[tp.Any] = None,
    right_time: tp.Optional[tp.Any] = None,
    before: tp.Any = 0,
    after: tp.Any = 0,
    left_watermark: tp.Optional[tp.Any] = None,
    right_watermark: tp.Optional[tp.Any] = None,
    maxsize: int = 0,
    timeout: float = 0,
    workers: int = 1,
    on_start: tp.Callable = None,
    on_done: tp.Callable = None,
    **kwargs,
) -> tp.Union[Stage[JoinResult], pypeln_utils.Partial[Stage[JoinResult]]]:
    """
    Full outer join of two streams on a business key with an event-time
    tolerance.

    A left record `l` and a right record `r` are paired when they share the
    same business key and::

        left_time(l) - before  <=  right_time(r)  <=  left_time(l) + after

    Both boundaries are inclusive. Matching is a multiset (bag) join:
    duplicates are never folded, every matching `(left, right)` combination is
    emitted exactly once.

    Each side has an explicit, monotonically advancing watermark. A record
    whose event time is at or before its own side's watermark is emitted on its
    own with `kind == "left_late"` / `"right_late"` and never participates in
    matching. An unmatched row (`kind == "left"` / `"right"`) is only emitted
    once the *opposite* side's watermark has ruled out any future match: for a
    left record that happens when `left_time + after <= right_watermark`, for a
    right record when `right_time + before <= left_watermark`.

    The end of one side is equivalent to that side's watermark becoming
    infinity: all buffered rows on the other side are released immediately
    while the remaining side keeps running. The stage never waits for both
    sides to be complete before producing results, releases sealed state as
    soon as a watermark allows it and honours `maxsize` on all of its internal
    queues.

    Upstream exceptions propagate and abort the execution. A `Close` control
    value observed on either stream also aborts the execution (pending state is
    discarded); a normal termination flushes all remaining rows.

    The returned stage yields `JoinResult` values, which behave like
    `(left, right)` tuples, so the result can be passed directly to the regular
    `map`, `each`, `run` or `to_iterable` functions.

    Arguments:
        left: A Stage or Iterable with the left records.
        right: A Stage or Iterable with the right records.
        on: Business key extractor applied to both sides. It may be a callable,
            a field name, or a list/tuple of field names.
        left_key / right_key: Optional per-side overrides for `on`.
        left_time / right_time: Event time extractors, accepting the same forms
            as `on`. When omitted, mappings are searched for
            `time`/`timestamp`/`ts`/`event_time` and tuples/lists use their
            second element.
        before: Tolerance into the past (`a`), in event-time units.
        after: Tolerance into the future (`b`), in event-time units.
        left_watermark / right_watermark: Explicit watermark source for each
            side. A callable / field name extracts a watermark carried by every
            record (it takes effect after that record is processed). A numeric
            value sets the initial watermark. A Stage or Iterable is consumed
            as a stream of watermark updates. Watermarks can also be injected
            in the data streams by yielding `Watermark` objects.
        maxsize: The maximum number of objects the internal queues can hold,
            propagating back pressure to the upstream stages. Defaults to `0`,
            which means unbounded for the user facing queues; the small queue
            that hops records into the join is always bounded so that a `Close`
            or an upstream error can interrupt the producers promptly.
        timeout: Seconds before stopping the worker if its current task is not
            completed. Defaults to `0` which means it is unbounded.
        workers: Kept for API compatibility; the join keeps a single consistent
            state and therefore always uses one worker.
        on_start / on_done: Regular pypeln lifecycle callbacks.

    Returns:
        A `Stage` yielding `JoinResult` values, or a `Partial` when one of the
        streams is omitted, so both `stage | pl.thread.join(other)` and
        `stage | pl.thread.join(left=other)` work.
    """

    # -- keyword aliases ---------------------------------------------------
    for alias, canonical in _ALIASES.items():
        if alias not in kwargs:
            continue

        value = kwargs.pop(alias)

        if canonical == "on":
            if on is None:
                on = value
        elif canonical == "left_key":
            if left_key is None:
                left_key = value
        elif canonical == "right_key":
            if right_key is None:
                right_key = value
        elif canonical == "left_time":
            if left_time is None:
                left_time = value
        elif canonical == "right_time":
            if right_time is None:
                right_time = value
        elif canonical == "before":
            if before == 0:
                before = value
        elif canonical == "after":
            if after == 0:
                after = value
        elif canonical == "left_watermark":
            if left_watermark is None:
                left_watermark = value
        elif canonical == "right_watermark":
            if right_watermark is None:
                right_watermark = value

    if "watermark" in kwargs:
        value = kwargs.pop("watermark")
        if left_watermark is None:
            left_watermark = value
        if right_watermark is None:
            right_watermark = value

    if "watermarks" in kwargs:
        value = kwargs.pop("watermarks")
        if isinstance(value, Mapping):
            if left_watermark is None:
                left_watermark = value.get(LEFT, value.get("left"))
            if right_watermark is None:
                right_watermark = value.get(RIGHT, value.get("right"))
        else:
            left_watermark, right_watermark = value

    for pair_key in ("tolerance", "interval", "bounds", "window", "range"):
        if pair_key not in kwargs:
            continue

        value = kwargs.pop(pair_key)

        if isinstance(value, Mapping):
            before = value.get("before", value.get("left", value.get("a", before)))
            after = value.get("after", value.get("right", value.get("b", after)))
        else:
            before, after = value

    if "time" in kwargs:
        value = kwargs.pop("time")
        if left_time is None:
            left_time = value
        if right_time is None:
            right_time = value

    # be forgiving with (before, after) taken from a mapping of aliases
    if isinstance(before, Mapping) and "value" in before:
        before = before["value"]
    if isinstance(after, Mapping) and "value" in after:
        after = after["value"]

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"join() got unexpected keyword argument(s): {unknown}")

    before = 0 if before is None else before
    after = 0 if after is None else after

    # -- partials ----------------------------------------------------------
    if isinstance(left, pypeln_utils.Undefined) and isinstance(
        right, pypeln_utils.Undefined
    ):
        raise TypeError("join() requires a `left` and a `right` stream")

    if isinstance(left, pypeln_utils.Undefined):
        return pypeln_utils.Partial(
            lambda left: join(
                left,
                right,
                on=on,
                left_key=left_key,
                right_key=right_key,
                left_time=left_time,
                right_time=right_time,
                before=before,
                after=after,
                left_watermark=left_watermark,
                right_watermark=right_watermark,
                maxsize=maxsize,
                timeout=timeout,
                workers=workers,
                on_start=on_start,
                on_done=on_done,
            )
        )

    if isinstance(right, pypeln_utils.Undefined):
        return pypeln_utils.Partial(
            lambda right: join(
                left,
                right,
                on=on,
                left_key=left_key,
                right_key=right_key,
                left_time=left_time,
                right_time=right_time,
                before=before,
                after=after,
                left_watermark=left_watermark,
                right_watermark=right_watermark,
                maxsize=maxsize,
                timeout=timeout,
                workers=workers,
                on_start=on_start,
                on_done=on_done,
            )
        )

    # -- build the stage ---------------------------------------------------
    left_stage = to_stage(left, maxsize=maxsize)
    right_stage = to_stage(right, maxsize=maxsize)

    # `on` is the shared business key extractor, per-side values override it
    if left_key is None:
        left_key = on
    if right_key is None:
        right_key = on

    dependencies: tp.List[Stage] = [
        _tag_stage(left_stage, LEFT, maxsize),
        _tag_stage(right_stage, RIGHT, maxsize),
    ]

    def resolve_watermark(spec: tp.Any, side: str):
        # returns (extractor, initial value, optional watermark stage)
        if spec is None:
            return None, None, None

        if callable(spec) or isinstance(spec, str):
            return _make_getter(spec), None, None

        if _is_iterable_spec(spec):
            stage = to_stage(spec, maxsize=maxsize)
            return None, None, _watermark_stage(stage, side, maxsize)

        return None, spec, None

    watermark_left_fn, initial_left, wm_stage_left = resolve_watermark(
        left_watermark, LEFT
    )
    watermark_right_fn, initial_right, wm_stage_right = resolve_watermark(
        right_watermark, RIGHT
    )

    if wm_stage_left is not None:
        dependencies.append(wm_stage_left)
    if wm_stage_right is not None:
        dependencies.append(wm_stage_right)

    initial_watermarks: tp.Dict[str, tp.Any] = {}
    if initial_left is not None:
        initial_watermarks[LEFT] = initial_left
    if initial_right is not None:
        initial_watermarks[RIGHT] = initial_right

    return Stage(
        process_fn=_JoinProcess(
            key_left=_make_getter(left_key, _default_key),
            key_right=_make_getter(right_key, _default_key),
            time_left=_make_getter(left_time, _default_time),
            time_right=_make_getter(right_time, _default_time),
            before=before,
            after=after,
            watermark_left=watermark_left_fn,
            watermark_right=watermark_right_fn,
            initial_watermarks=initial_watermarks,
        ),
        workers=1,
        maxsize=maxsize,
        timeout=timeout,
        total_sources=sum(stage.workers for stage in dependencies),
        dependencies=dependencies,
        on_start=on_start,
        on_done=on_done,
        f_args=[],
    )


# common alternative spellings for the same operation
interval_join = join
outer_join = join
full_outer_join = join
interval_outer_join = join
temporal_join = join
join_interval = join
