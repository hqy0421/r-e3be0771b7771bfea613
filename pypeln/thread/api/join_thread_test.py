import itertools
import threading
import time
import typing as tp

import hypothesis as hp
from hypothesis import strategies as st

import pypeln as pl
from pypeln.thread import Close, JoinResult, Watermark, join
from pypeln.thread.api.join import _JoinState

MAX_EXAMPLES = 10

LEFT = "left"
RIGHT = "right"


def make_state(before=0, after=0, **kwargs):
    rows = []

    state = _JoinState(
        key_left=kwargs.get("key", lambda r: r["k"]),
        key_right=kwargs.get("key", lambda r: r["k"]),
        time_left=kwargs.get("time", lambda r: r["t"]),
        time_right=kwargs.get("time", lambda r: r["t"]),
        before=before,
        after=after,
        watermark_left=None,
        watermark_right=None,
        initial_watermarks={},
        put=rows.append,
    )

    return state, rows


def rec(k, t, tag=""):
    return {"k": k, "t": t, "tag": tag}


#########################################################
# JoinResult
#########################################################


def test_join_result_behaves_like_a_pair():
    left, right = rec(1, 1), rec(1, 2)
    row = JoinResult(left, right)

    assert row == (left, right)
    assert row == (left, right, "match")
    assert row.left is left
    assert row.right is right
    assert row.record is left
    assert row.matched and not row.unmatched and not row.late

    l, r = row
    assert (l, r) == (left, right)

    assert row["left"] is left
    assert row["right"] is right
    assert row == {"left": left, "right": right}


def test_join_result_unmatched_and_late_metadata():
    left = rec(1, 1)

    row = JoinResult(left, None, pl.thread.api.join.LEFT_UNMATCHED)
    assert row.unmatched and not row.matched and not row.late
    assert row.side == LEFT and row.record is left

    row = JoinResult(left, None, pl.thread.api.join.LEFT_LATE)
    assert row.late and row.is_late and row.left_late
    assert row.side == LEFT


#########################################################
# semantics of the state machine
#########################################################


def test_exact_time_match():
    state, rows = make_state()
    state.add(LEFT, (), rec("a", 10))
    state.add(RIGHT, (), rec("a", 10))
    state.end(LEFT)
    state.end(RIGHT)

    assert len(rows) == 1
    assert rows[0].matched


def test_bounds_are_inclusive():
    for right_time, expected in [
        (8, False),
        (9, True),
        (10, True),
        (12, True),
        (13, False),
    ]:
        state, rows = make_state(before=1, after=2)
        state.add(LEFT, (), rec("a", 10))
        state.add(RIGHT, (), rec("a", right_time))
        state.end(LEFT)
        state.end(RIGHT)

        matches = [row for row in rows if row.matched]

        assert bool(matches) is expected, right_time


def test_duplicates_are_not_deduplicated():
    state, rows = make_state()
    for i in range(3):
        state.add(LEFT, (), rec("a", 5, tag=f"l{i}"))
    for i in range(2):
        state.add(RIGHT, (), rec("a", 5, tag=f"r{i}"))
    state.end(LEFT)
    state.end(RIGHT)

    matches = [row for row in rows if row.matched]

    assert len(matches) == 6  # 3 x 2
    assert {(row.left["tag"], row.right["tag"]) for row in matches} == {
        (f"l{i}", f"r{j}") for i in range(3) for j in range(2)
    }


def test_matching_requires_the_same_key():
    state, rows = make_state(before=10, after=10)
    state.add(LEFT, (), rec("a", 1))
    state.add(RIGHT, (), rec("b", 1))
    state.end(LEFT)
    state.end(RIGHT)

    assert not any(row.matched for row in rows)
    assert {row.kind for row in rows} == {"left", "right"}


def test_late_records_at_or_before_own_watermark():
    state, rows = make_state()
    state.add(LEFT, (), rec("a", 10))
    state.advance(LEFT, 10)
    state.add(LEFT, (), rec("a", 10))  # exactly at the watermark -> late
    state.add(LEFT, (), rec("a", 11))  # after the watermark -> normal
    state.end(LEFT)
    state.end(RIGHT)

    late = [row for row in rows if row.late]

    assert len(late) == 1
    assert late[0].left["t"] == 10
    assert late[0].kind == "left_late"


def test_late_records_never_participate_in_matching():
    state, rows = make_state()
    state.advance(LEFT, 10)
    state.add(LEFT, (), rec("a", 10))  # late
    state.add(RIGHT, (), rec("a", 10))

    state.end(LEFT)
    state.end(RIGHT)

    assert not any(row.matched for row in rows)
    assert [row.kind for row in rows if row.left is not None] == ["left_late"]


def test_watermarks_advance_monotonically():
    state, _ = make_state()
    state.advance(LEFT, 10)
    state.advance(LEFT, 5)  # ignored

    assert state.watermarks[LEFT] == 10


def test_unmatched_released_only_when_opposite_watermark_excludes_a_match():
    state, rows = make_state(after=5)

    state.add(LEFT, (), rec("a", 10))

    # right watermark below left_time + after: a future right record can match
    state.advance(RIGHT, 14)
    assert rows == []

    # right watermark reaches the deadline (inclusive): sealed
    state.advance(RIGHT, 15)
    assert [row.kind for row in rows] == ["left"]

    state.add(LEFT, (), rec("a", 20))
    state.advance(RIGHT, 100)
    assert [row.kind for row in rows] == ["left", "left"]


def test_right_side_deadline_uses_before():
    state, rows = make_state(before=5, after=100)

    state.add(RIGHT, (), rec("a", 10))

    state.advance(LEFT, 14)
    assert rows == []

    state.advance(LEFT, 15)  # right_time + before
    assert [row.kind for row in rows] == ["right"]


def test_end_of_side_is_an_infinite_watermark():
    state, rows = make_state(before=100, after=100)
    state.add(RIGHT, (), rec("a", 10))
    state.end(RIGHT)

    # the remaining left side can keep matching buffered right records
    state.add(LEFT, (), rec("a", 10))
    assert any(row.matched for row in rows)

    state.add(LEFT, (), rec("b", 10))
    state.end(LEFT)

    assert [row.kind for row in rows].count("left") == 1


def test_end_of_side_flushes_the_other_side_immediately():
    state, rows = make_state(before=100, after=100)
    state.add(LEFT, (), rec("a", 10))
    state.end(RIGHT)

    # no need to wait for the left side to end
    assert [row.kind for row in rows] == ["left"]


def test_close_discards_state_and_emits_nothing():
    state, rows = make_state()
    state.add(LEFT, (), rec("a", 10))
    state.close()
    state.add(RIGHT, (), rec("a", 10))
    state.end(LEFT)
    state.end(RIGHT)

    assert rows == []
    assert state.closed


def test_a_left_record_matching_several_right_records_is_emitted_for_each():
    state, rows = make_state(before=2, after=2)
    state.add(RIGHT, (), rec("a", 1, "r1"))
    state.add(RIGHT, (), rec("a", 2, "r2"))
    state.add(RIGHT, (), rec("a", 3, "r3"))
    state.add(LEFT, (), rec("a", 2, "l"))
    state.end(LEFT)
    state.end(RIGHT)

    assert len([row for row in rows if row.matched]) == 3


#########################################################
# public API through the pipeline
#########################################################


def test_join_basic_pipeline():
    left = [rec("a", 10), rec("b", 20)]
    right = [rec("a", 10), rec("c", 30)]

    rows = list(join(left, right, on="k", left_time="t", right_time="t"))

    assert len(rows) == 3
    assert sum(row.matched for row in rows) == 1
    assert sorted(row.kind for row in rows) == ["left", "match", "right"]


def test_join_accepts_callables_and_defaults():
    left = [(1, 10, "l"), (2, 20, "m")]
    right = [(1, 10, "r")]

    rows = list(join(left, right, before=1, after=1))

    assert len([row for row in rows if row.matched]) == 1


def test_join_result_can_be_mapped():
    left = [{"k": "a", "t": 1, "v": 1}]
    right = [{"k": "a", "t": 1, "v": 2}]

    stage = join(left, right, on="k", left_time="t", right_time="t")
    stage = pl.thread.map(lambda pair: (pair.left["v"], pair.right["v"]), stage)

    assert list(stage) == [(1, 2)]


def test_join_result_can_be_consumed_by_each():
    collected = []

    left = [{"k": "a", "t": 1}]
    right = [{"k": "a", "t": 1}]

    stage = join(left, right, on="k", left_time="t", right_time="t")
    pl.thread.each(lambda pair: collected.append(pair.kind), stage, run=True)

    assert collected == ["match"]


def test_join_watermark_control_values_in_stream():
    left = [
        rec("a", 10),
        Watermark(10),
        rec("a", 10),  # late
        rec("a", 20),
    ]
    right = [rec("a", 10)]

    rows = list(join(left, right, on="k", left_time="t", right_time="t"))

    assert any(row.left_late for row in rows)


def test_join_close_ends_the_execution():
    def left():
        yield rec("a", 1)
        yield Close()
        yield rec("a", 2)

    rows = list(join(left(), [rec("a", 1)], on="k", left_time="t", right_time="t"))

    # rows released before the close may already be out, but the close must cut
    # the execution short: `rec("a", 2)` never gets a chance to be processed
    assert all(row.left is None or row.left["t"] != 2 for row in rows)


def test_join_close_ends_an_infinite_execution():
    def infinite_left():
        yield rec("a", 0)
        yield Close()
        for i in itertools.count(1):
            yield rec("a", i)

    def infinite_right():
        for i in itertools.count():
            yield rec("b", i)

    baseline = len(threading.enumerate())

    start = time.time()
    rows = list(
        join(infinite_left(), infinite_right(), on="k", left_time="t", right_time="t")
    )

    assert rows == []
    assert time.time() - start < 10

    # the workers created by this execution must not survive it
    deadline = time.time() + 10
    while time.time() < deadline and len(threading.enumerate()) > baseline:
        time.sleep(0.05)

    assert len(threading.enumerate()) <= baseline


def test_join_raises_upstream_errors():
    class MyError(Exception):
        pass

    def left():
        yield rec("a", 1)
        raise MyError()

    error = None

    try:
        list(join(left(), [rec("a", 1)], on="k", left_time="t", right_time="t"))
    except MyError as e:
        error = e

    assert isinstance(error, MyError)


def test_join_partial_supports_pipe():
    left = [rec("a", 1)]
    right = [rec("a", 1)]

    rows = left | join(right, on="k", left_time="t", right_time="t") | list

    assert len(rows) == 1 and rows[0].matched


def test_join_does_not_wait_for_both_sides():
    def left():
        yield rec("a", 1)
        yield rec("a", 2)

    def right():
        yield rec("z", 100)
        yield Watermark(1000)
        time.sleep(1.0)
        yield rec("z", 101)

    iterator = iter(
        pl.thread.to_iterable(
            join(left(), right(), on="k", left_time="t", right_time="t"),
            maxsize=2,
        )
    )

    start = time.time()
    first = next(iterator)
    elapsed = time.time() - start

    # a result is available long before the slow right side finishes: the
    # (already ended) left side releases the buffered right record as
    # unmatched, and the end of the right side later releases the left rows
    assert first.kind in ("left", "right")
    assert elapsed < 0.9

    # drain so the pipeline terminates normally
    list(iterator)


def test_join_bounded_queues():
    produced = [0]
    total = 500

    def source(tag):
        for i in range(total):
            produced[0] += 1
            yield {"k": i, "t": i}

    stage = join(
        source("l"),
        source("r"),
        on="k",
        left_time="t",
        right_time="t",
        maxsize=2,
    )
    iterator = iter(pl.thread.to_iterable(stage, maxsize=2))

    for row in itertools.islice(iterator, 5):
        pass

    # back pressure keeps the producers from running ahead of the consumer
    assert produced[0] < 100

    # drain so the pipeline terminates normally
    list(iterator)


@hp.given(
    left=st.lists(st.tuples(st.text(), st.integers(min_value=0)), max_size=6),
    right=st.lists(st.tuples(st.text(), st.integers(min_value=0)), max_size=6),
)
@hp.settings(max_examples=MAX_EXAMPLES)
def test_join_matches_naive_reference(
    left: tp.List[tp.Tuple[str, int]], right: tp.List[tp.Tuple[str, int]]
):
    before, after = 2, 1

    left_records = [{"k": k, "t": t, "i": i} for i, (k, t) in enumerate(left)]
    right_records = [{"k": k, "t": t, "i": i} for i, (k, t) in enumerate(right)]

    expected = {}
    for l in left_records:
        for r in right_records:
            if l["k"] == r["k"] and l["t"] - before <= r["t"] <= l["t"] + after:
                expected[(l["i"], r["i"])] = True

    rows = list(
        join(
            left_records,
            right_records,
            on="k",
            left_time="t",
            right_time="t",
            before=before,
            after=after,
        )
    )

    matches = {(row.left["i"], row.right["i"]) for row in rows if row.matched}

    assert matches == set(expected)

    # every record that never matched appears exactly once as unmatched
    matched_left = {row.left["i"] for row in rows if row.matched}
    matched_right = {row.right["i"] for row in rows if row.matched}

    unmatched_left = [row for row in rows if row.kind == "left"]
    unmatched_right = [row for row in rows if row.kind == "right"]

    assert sorted(row.left["i"] for row in unmatched_left) == sorted(
        l["i"] for l in left_records if l["i"] not in matched_left
    )
    assert sorted(row.right["i"] for row in unmatched_right) == sorted(
        r["i"] for r in right_records if r["i"] not in matched_right
    )
