"""What a live run reports while it goes: ctx.progress and ctx.record."""
from dataclasses import dataclass

import numpy as np
import pytest
import yaml

import runkit.exp
from runkit import Experiment, RunContext, load_metrics


@dataclass
class Cfg:
    n: int = 5


def _status(d):
    return yaml.safe_load((d / "status.yaml").read_text())


def _run(body, tmp_path, *args, name="live"):
    exp = Experiment(name)
    exp.run(body)
    return exp.main([*args, f"--root={tmp_path}"])


# -- progress -------------------------------------------------------------------

def test_progress_total_is_sticky_and_lands_in_status(tmp_path, monkeypatch):
    monkeypatch.setattr(runkit.exp, "PROGRESS_EVERY_S", 0.0)       # write every call

    def body(cfg: Cfg, ctx: RunContext):
        ctx.progress(total=cfg.n)
        seen = [_status(ctx.dir)]
        for i in range(1, cfg.n + 1):
            ctx.progress(i)
        seen.append(_status(ctx.dir))
        return seen

    r = _run(body, tmp_path)
    at_start, at_end = r.retval
    assert (at_start["status"], at_start["progress"], at_start["total"]) == ("running", None, 5)
    assert (at_end["progress"], at_end["total"]) == (5, 5)             # total remembered
    final = _status(r.context.dir)
    assert (final["status"], final["progress"], final["total"]) == ("ok", 5, 5)


def test_progress_writes_are_throttled_but_the_last_value_is_kept(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        for i in range(1000):
            ctx.progress(i)
        return _status(ctx.dir)["progress"]

    r = _run(body, tmp_path)
    assert r.retval == 0                        # only the first call wrote, mid-run ...
    assert _status(r.context.dir)["progress"] == 999      # ... the final write has the last


def test_a_failed_run_shows_how_far_it_got(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        ctx.progress(3, total=10)
        raise ValueError("boom")

    with pytest.raises(ValueError):
        _run(body, tmp_path)
    d = next((tmp_path / "live").glob("2*"))
    st = _status(d)
    assert (st["status"], st["progress"], st["total"]) == ("failed", 3, 10)


def test_progress_without_calls_is_null_and_takes_numpy_counts(tmp_path):
    def silent(cfg: Cfg, ctx: RunContext):
        pass
    r = _run(silent, tmp_path)
    assert (_status(r.context.dir)["progress"], _status(r.context.dir)["total"]) == (None, None)

    def np_count(cfg: Cfg, ctx: RunContext):
        ctx.progress(np.int64(7), total=np.float32(10))
    r = _run(np_count, tmp_path)
    st = _status(r.context.dir)
    assert st["progress"] == 7 and st["total"] == 10.0


def test_progress_rejects_non_numbers_and_needs_total_by_name(tmp_path):
    def bad(cfg: Cfg, ctx: RunContext):
        ctx.progress("3")
    with pytest.raises(TypeError, match="must be a number"):
        _run(bad, tmp_path)

    def positional_total(cfg: Cfg, ctx: RunContext):
        ctx.progress(3, 10)
    with pytest.raises(TypeError):
        _run(positional_total, tmp_path)


# -- record ---------------------------------------------------------------------

def test_record_appends_rows_to_the_run_stream(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        for i in range(3):
            ctx.record(it=i, loss=1.0 / (i + 1))
        ctx.record(it=3, val=np.float32(0.5), stream="as-a-value")  # "stream" as a key: a value

    r = _run(body, tmp_path)
    rows = load_metrics(r.context.dir)
    assert [row["it"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["loss"] == 1.0 and rows[3]["val"] == 0.5 and rows[3]["stream"] == "as-a-value"
    assert all(row["time"] and row["elapsed_s"] >= 0 for row in rows)
    assert (r.context.dir / "metrics" / "run.jsonl").is_file()


def test_eval_records_to_a_named_stream_only(tmp_path):
    exp = Experiment("ev")

    @exp.run
    def run(cfg: Cfg, ctx: RunContext):
        ctx.record(it=0)

    @exp.eval
    def evaluate(cfg: Cfg, ctx: RunContext):
        ctx.record("eval", ret=1.5)
        with pytest.raises(RuntimeError, match="name a stream"):
            ctx.record(ret=1.5)
        with pytest.raises(RuntimeError, match="'run' is the run's own"):
            ctx.record("run", ret=1.5)
        return ctx.dir

    exp.main([f"--root={tmp_path}"])
    d = exp.main(["eval", f"--root={tmp_path}"])
    assert [r["ret"] for r in load_metrics(d, "eval")] == [1.5]
    assert load_metrics(d, "eval")[0]["elapsed_s"] >= 0        # since eval opened the run
    assert [r["it"] for r in load_metrics(d)] == [0]            # the run's stream untouched


def test_record_refuses_runkits_keys_and_bad_stream_names(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        with pytest.raises(ValueError, match="added by runkit"):
            ctx.record(time=1)
        for bad in ("", ".hidden", "a/b"):
            with pytest.raises(ValueError, match="bad metrics stream"):
                ctx.record(bad, x=1)

    _run(body, tmp_path)


def test_a_torn_last_line_is_skipped(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        ctx.record(it=0)

    r = _run(body, tmp_path)
    with open(r.context.dir / "metrics" / "run.jsonl", "a") as f:
        f.write('{"it": 1, "lo')                                 # killed mid-write
    assert [row["it"] for row in load_metrics(r.context.dir)] == [0]
    assert load_metrics(r.context.dir, "nope") == []


def test_only_a_live_run_reports_progress(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        pass

    r = _run(body, tmp_path)
    with pytest.raises(RuntimeError, match="only the live run"):
        r.context.progress(1)
    with pytest.raises(RuntimeError, match="name a stream"):
        r.context.record(x=1)
