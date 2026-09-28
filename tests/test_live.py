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
    assert all(row["_time"] and row["_elapsed_s"] >= 0 for row in rows)
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
    assert load_metrics(d, "eval")[0]["_elapsed_s"] >= 0       # since eval opened the run
    assert [r["it"] for r in load_metrics(d)] == [0]            # the run's stream untouched


def test_record_refuses_runkits_keys_and_bad_stream_names(tmp_path):
    def body(cfg: Cfg, ctx: RunContext):
        with pytest.raises(ValueError, match="keys starting with '_' are runkit's"):
            ctx.record(_time=1)
        with pytest.raises(ValueError, match="runkit's"):
            ctx.record(_anything=1)
        ctx.record(time=1.5, elapsed_s=2)                        # plain names are yours
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


def test_compile_metrics_gives_aligned_columns(tmp_path):
    from runkit import compile_metrics

    def body(cfg: Cfg, ctx: RunContext):
        for i in range(4):
            if i == 2:
                ctx.record(it=i, loss=1.0 / (i + 1), eval_return=12.0, note="eval")
            else:
                ctx.record(it=i, loss=1.0 / (i + 1))
        ctx.record(done=True)                                    # a bool: not numeric

    r = _run(body, tmp_path)
    m = compile_metrics(r.context.dir)                           # {key: array}
    assert list(m) == ["_line", "_time", "_elapsed_s", "it", "loss", "eval_return",
                       "note", "done"]
    assert list(m["_line"]) == [0, 1, 2, 3, 4]                   # added on reading, not stored
    assert all(len(v) == 5 for v in m.values())                  # one entry per line
    assert m["it"].dtype == float and list(m["it"][:4]) == [0, 1, 2, 3]
    assert np.isnan(m["it"][4])                                  # the last line has no "it"
    ev = m["eval_return"]
    assert ev[2] == 12.0 and np.isnan(ev[[0, 1, 3, 4]]).all()     # sparse, but aligned
    assert m["note"].tolist() == [None, None, "eval", None, None]  # non-numeric: object
    assert m["done"].tolist() == [None, None, None, None, True]
    assert m["_time"].dtype == object and m["_elapsed_s"].dtype == float
    later = m["it"] >= 2                                         # a mask from one column ...
    assert m["eval_return"][later][0] == 12.0 and m["note"][later][0] == "eval"  # ... any other
    assert compile_metrics(r.context.dir, "nope") == {}


# -- --follow: the run prints a stream as it goes ---------------------------------

def _rec(cfg: Cfg, ctx: RunContext):
    for i in range(3):
        ctx.record(it=i, loss=1.0 / (i + 1))
        ctx.record("reward", it=i, lin=0.4)


def test_a_run_follows_its_run_stream_by_default(tmp_path, capsys):
    _run(_rec, tmp_path)
    out, err = capsys.readouterr()
    lines = [l.split() for l in err.splitlines()]
    assert ["_elapsed_s", "it", "loss"] in lines                  # the table header
    assert sum(1 for l in lines if l[1:] in (["0", "1"], ["1", "0.5"], ["2", "0.333"])) == 3
    assert "_elapsed_s" not in out                                # stderr, not stdout
    assert not any("lin" in l for l in lines if l and l[0] == "_elapsed_s")


def test_follow_another_stream_or_none(tmp_path, capsys):
    _run(_rec, tmp_path, "--follow", "reward")
    headers = [l.split() for l in capsys.readouterr().err.splitlines() if "_elapsed_s" in l]
    assert headers == [["_elapsed_s", "it", "lin"]]
    _run(_rec, tmp_path, "--follow=none")
    assert "_elapsed_s" not in capsys.readouterr().err
    _run(_rec, tmp_path, "--follow")                              # bare: the run stream
    assert "_elapsed_s" in capsys.readouterr().err


def test_follow_default_from_experiment_toml(tmp_path):
    from runkit.settings import resolve_follow
    exp_file = tmp_path / "e.py"
    exp_file.write_text("")
    assert resolve_follow(exp_file, "e") == "run"
    (tmp_path / "experiment.toml").write_text('[env]\nfollow = "none"\n\n[env.other]\nfollow = "reward"\n')
    assert resolve_follow(exp_file, "e") is None
    assert resolve_follow(exp_file, "other") == "reward"
    assert resolve_follow(exp_file, "e", explicit="run") == "run"   # the flag wins
    assert resolve_follow(exp_file, "e", explicit=True) == "run"    # a bare --follow
