"""Reading metrics without the experiment: plot_metrics, runkit metrics / plot."""
import math
from dataclasses import dataclass

import pytest

from runkit import Experiment, RunContext, cli
from runkit.metrics import plot_metrics, summarize_metrics


@dataclass
class Cfg:
    n: int = 20


@pytest.fixture
def run_dir(tmp_path):
    exp = Experiment("metr")

    @exp.run
    def run(cfg: Cfg, ctx: RunContext):
        for i in range(cfg.n):
            steps = (i + 1) * 1000
            ctx.record(it=i, steps=steps, loss=math.exp(-i / 5))
            if i % 5 == 4:
                ctx.record("eval", steps=steps, ret=float(i))

    return exp.main([f"--root={tmp_path}"]).context.dir


def test_summary(run_dir):
    by_key = {r["key"]: r for r in summarize_metrics(run_dir)}
    assert by_key["it"] == {"key": "it", "rows": 20, "last": 19.0, "min": 0.0, "mean": 9.5,
                            "max": 19.0}
    assert by_key["_time"]["min"] is None and by_key["_time"]["rows"] == 20
    assert {r["key"]: r["rows"] for r in summarize_metrics(run_dir, "eval")}["ret"] == 4


def test_plot_writes_a_png_named_by_its_arguments(run_dir):
    out = plot_metrics(run_dir, ["loss"])
    assert out == run_dir / "metrics" / "loss_vs_line.png" and out.stat().st_size > 0
    both = plot_metrics(run_dir, ["loss", "eval:ret"], x="steps")
    assert both.name == "loss__eval-ret_vs_steps.png"
    assert plot_metrics(run_dir, ["loss"]) == out                 # same plot: overwritten


def test_plot_refuses_what_it_cannot_draw(run_dir):
    with pytest.raises(ValueError, match="need an x key they share"):
        plot_metrics(run_dir, ["loss", "eval:ret"])              # line numbers differ
    with pytest.raises(ValueError, match="no key 'nope'"):
        plot_metrics(run_dir, ["nope"])
    with pytest.raises(ValueError, match="no stream 'x'"):
        plot_metrics(run_dir, ["x:loss"])
    with pytest.raises(ValueError, match="not numeric"):
        plot_metrics(run_dir, ["_time"])


def test_runkit_metrics_and_plot_commands(run_dir, capsys, monkeypatch):
    overview = cli.main(["metrics", str(run_dir)])                # every stream
    out = capsys.readouterr().out
    assert list(overview) == ["run", "eval"]                     # the run's own stream first
    assert "metrics/run.jsonl  20 rows" in out and "metrics/eval.jsonl  4 rows" in out
    assert "20000" in out and "e+04" not in out                  # integers stay integers
    assert {r["key"] for r in overview["run"]} >= {"it", "steps", "loss"}
    assert {r["key"] for r in overview["eval"]} >= {"steps", "ret"}
    tables = cli.main(["metrics", "info", str(run_dir), "loss"])  # a key: its stream
    assert [r["key"] for r in tables["run"]] == ["loss"] and list(tables) == ["run"]
    capsys.readouterr()

    png = cli.main(["metrics", "plot", str(run_dir), "loss", "eval:ret", "--x", "steps"])
    assert capsys.readouterr().out.strip() == str(png) and png.is_file()

    monkeypatch.chdir(run_dir)                                   # PATH defaults to cwd
    assert cli.main(["metrics", "eval:"])["eval"][0]["key"] == "_time"  # another stream
    assert cli.main(["metrics", "plot", "loss", "--x=it"]).name == "loss_vs_it.png"
    monkeypatch.chdir(run_dir / "metrics")                       # ... or its metrics/ folder
    assert list(cli.main(["metrics"])) == ["run", "eval"]


def test_commands_explain_bad_input(run_dir, tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="no metrics stream 'nope'"):
        cli.main(["metrics", str(run_dir), "nope:"])
    with pytest.raises(SystemExit, match="no key 'nope' in stream 'run'"):
        cli.main(["metrics", str(run_dir), "nope"])
    with pytest.raises(SystemExit, match="follow takes one stream"):
        cli.main(["metrics", "follow", str(run_dir), "loss", "eval:ret"])
    with pytest.raises(SystemExit, match="need an x key"):
        cli.main(["metrics", "plot", str(run_dir), "loss", "eval:ret"])
    with pytest.raises(SystemExit, match="unknown option '--y'"):
        cli.main(["metrics", "plot", str(run_dir), "loss", "--y", "a"])
    with pytest.raises(SystemExit, match="now `runkit metrics plot"):
        cli.main(["plot", str(run_dir)])
    with pytest.raises(SystemExit, match="now `runkit metrics follow"):
        cli.main(["metrics", str(run_dir), "-f"])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="is not a run dir"):
        cli.main(["metrics"])


# -- selecting lines ---------------------------------------------------------

from runkit import compile_metrics


@pytest.fixture
def ckpt_run(tmp_path):
    """30 lines of `run`, a checkpoint after lines 10 and 20 ("000001", "000002"),
    one named "best" after line 25; eval every 10 lines (written by the run)."""
    exp = Experiment("sel")

    @exp.run
    def run(cfg: Cfg, ctx: RunContext):
        for i in range(30):
            ctx.record(it=i, steps=(i + 1) * 100)
            if i in (9, 19):
                with ctx.checkpoint():
                    pass
                ctx.record("eval", steps=(i + 1) * 100, ret=float(i))
            if i == 24:
                with ctx.checkpoint("best"):
                    pass

    return exp.main([f"--root={tmp_path}"]).context.dir


def _its(cols):
    return [int(v) for v in cols["it"]]


def test_rows_are_plain_slicing_of_the_columns(ckpt_run):
    m = compile_metrics(ckpt_run)
    assert m["it"][-3:].tolist() == [27, 28, 29]
    assert m["it"][2:5].tolist() == [2, 3, 4]
    assert m["_line"][-3:].tolist() == [27, 28, 29]               # original numbers
    assert m["it"][m["steps"] > 2800].tolist() == [28, 29]


def test_start_end_window_on_line_or_x_with_negatives_from_the_end(ckpt_run):
    assert _its(compile_metrics(ckpt_run, start=-2)) == [27, 28, 29]              # last lines
    assert _its(compile_metrics(ckpt_run, start=5, end=7)) == [5, 6, 7]          # inclusive
    assert _its(compile_metrics(ckpt_run, x="steps", start=-200)) == [27, 28, 29]  # last 200 steps
    assert _its(compile_metrics(ckpt_run, x="steps", start="2800", end="2900")) == [27, 28]


def test_start_end_checkpoint_names_are_exact_row_boundaries(ckpt_run):
    assert _its(compile_metrics(ckpt_run, start="000001", end="000002")) == list(range(10, 20))
    assert _its(compile_metrics(ckpt_run, start="best")) == list(range(25, 30))
    assert _its(compile_metrics(ckpt_run, end="000001")) == list(range(10))
    # the eval stream was written by the run too, so its rows are counted
    assert compile_metrics(ckpt_run, "eval", start="000002")["ret"].tolist() == [19.0]


def test_a_stream_the_checkpoint_did_not_count_asks_for_a_value(ckpt_run):
    with open(ckpt_run / "metrics" / "later.jsonl", "w") as f:
        f.write('{"_time": "t", "_elapsed_s": 0, "steps": 100}\n')
    with pytest.raises(ValueError, match="no row count for stream 'later'"):
        compile_metrics(ckpt_run, "later", start="best")


def test_plot_and_metrics_take_the_selection(ckpt_run, capsys):
    png = cli.main(["metrics", "plot", str(ckpt_run), "it", "--start", "000001", "--end", "000002"])
    assert png.name == "it_vs_line_start000001_end000002.png"
    tail = cli.main(["metrics", "plot", str(ckpt_run), "it", "eval:ret", "--x", "steps",
                     "--start", "-1000"])
    assert tail.name == "it__eval-ret_vs_steps_start-1000.png" and tail.is_file()
    rows = cli.main(["metrics", str(ckpt_run), "--rows", "-5:"])["run"]
    assert {r["key"]: r for r in rows}["it"]["min"] == 25.0
    assert "5 rows" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="neither a number nor a checkpoint"):
        cli.main(["metrics", str(ckpt_run), "--start", "nope"])


def test_plot_without_keys_draws_every_numeric_key(run_dir, capsys):
    out = plot_metrics(run_dir, [])
    assert out.name == "all_vs_line.png" and out.stat().st_size > 0
    assert plot_metrics(run_dir, ["eval:"], x="steps").name == "eval-all_vs_steps.png"
    assert cli.main(["metrics", "plot", str(run_dir)]).name == "all_vs_line.png"   # no key
    assert cli.main(["metrics", "plot", str(run_dir), "--rows", "-5:"]).name == "all_vs_line_rows-5-.png"   # ":" kept out of file names


def test_plot_all_needs_numbers(run_dir):
    with open(run_dir / "metrics" / "words.jsonl", "w") as f:
        f.write('{"_time": "t", "_elapsed_s": 0, "note": "hi"}\n')
    with pytest.raises(ValueError, match="no numeric keys"):
        plot_metrics(run_dir, ["words:"])
    with pytest.raises(ValueError, match="no stream 'nope'"):
        plot_metrics(run_dir, ["nope:"])



def test_metrics_with_keys(run_dir):
    rows = cli.main(["metrics", str(run_dir), "loss", "it"])["run"]
    assert [r["key"] for r in rows] == ["it", "loss"]                # the stream's order
    assert [r["key"] for r in cli.main(["metrics", str(run_dir), "eval:ret"])["eval"]] == ["ret"]


# -- following ------------------------------------------------------------------

import json
import os
import socket

import yaml

from runkit.follow import follow


def _live_run(tmp_path, pid=None):
    """A run dir as a live run leaves it: running, some rows, no end yet."""
    d = tmp_path / "live" / "2026-09-27_10-00-00_abcdef12"
    (d / "metrics").mkdir(parents=True)
    (d / "run_context.yaml").write_text("id: live_abcdef12\nname: live\n")
    _set_status(d, "running", pid=pid or os.getpid())
    with open(d / "metrics" / "run.jsonl", "w") as f:
        for i in range(12):
            f.write(json.dumps({"_elapsed_s": i, "it": i, "loss": 1 / (i + 1)}) + "\n")
    return d


def _set_status(d, status, pid=None, checkpoint=None, **more):
    (d / "status.yaml").write_text(yaml.safe_dump({
        "status": status, "host": socket.gethostname(), "pid": pid or os.getpid(),
        "checkpoint": checkpoint, "duration_s": 5.0, "error": None, **more}))


def test_follow_prints_new_rows_checkpoints_and_the_end(tmp_path, capsys):
    d = _live_run(tmp_path)
    steps = []

    def sleep(_):                                # each poll: the run moves on a bit
        steps.append(1)
        with open(d / "metrics" / "run.jsonl", "a") as f:
            if len(steps) == 1:
                f.write(json.dumps({"_elapsed_s": 12, "it": 12, "loss": 0.07}) + "\n")
                f.write('{"_elapsed_s": 13, "it": 13, "lo')          # still being written
            elif len(steps) == 2:
                f.write('ss": 0.06}\n')
                ck = d / "checkpoints" / "current"
                ck.mkdir(parents=True)
                (ck / "checkpoint.yaml").write_text(yaml.safe_dump(
                    {"index": 1, "name": "current", "elapsed_s": 13, "info": {},
                     "progress": 13, "total": 100, "summary": {}}))
                _set_status(d, "running", checkpoint="checkpoints/current")
            else:
                _set_status(d, "ok", checkpoint="checkpoints/current")

    assert follow(d, sleep=sleep) == "ok"
    out, err = capsys.readouterr()
    its = [int(line.split()[1]) for line in out.splitlines() if line.split()[0][0].isdigit()]
    assert its == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]         # the last 10, then new ones
    assert "loss" in out.splitlines()[0]                              # a header first
    err = "".join(err.split())
    assert "◆checkpointcheckpoints/current" in err and "13/10013%" in err
    assert "✓live_abcdef12okin5.0s" in err


def test_follow_with_keys_and_a_dead_process(tmp_path, capsys):
    d = _live_run(tmp_path, pid=2 ** 22 + 12345)                      # no such process
    assert follow(d, keys=["loss"], sleep=lambda _: None) == "gone"
    out, err = capsys.readouterr()
    assert out.splitlines()[0].split() == ["time", "loss"]            # only the keys asked for
    assert "is still `running`, but its process" in " ".join(err.split())


def test_follow_from_the_command(tmp_path, monkeypatch, capsys):
    d = _live_run(tmp_path)
    _set_status(d, "failed", error="ValueError: boom")
    assert cli.main(["metrics", "follow", str(d), "--rows", "-2:"]) == "failed"   # ends at once
    out, err = capsys.readouterr()
    assert [line.split()[1] for line in out.splitlines()[1:]] == ["10", "11"]
    assert "failed" in err and "ValueError: boom" in " ".join(err.split())


def test_follow_puts_a_checkpoint_after_the_rows_it_followed(tmp_path, capsys, monkeypatch):
    """Rows and a checkpoint that arrive in one poll: the checkpoint's own row
    count (checkpoint.yaml `metrics`) places its line, not the poll."""
    from runkit import ui
    monkeypatch.setattr(ui, "checkpoint_saved", lambda **kw: print("CHECKPOINT"))
    d = _live_run(tmp_path)                                    # rows 0..11
    polls = []

    def sleep(_):
        polls.append(1)
        if len(polls) == 1:                                    # rows 12..15, checkpoint after 14
            with open(d / "metrics" / "run.jsonl", "a") as f:
                for i in range(12, 16):
                    f.write(json.dumps({"_elapsed_s": i, "it": i, "loss": 0.1}) + "\n")
            ck = d / "checkpoints" / "c"
            ck.mkdir(parents=True)
            (ck / "checkpoint.yaml").write_text(yaml.safe_dump(
                {"index": 1, "name": "c", "elapsed_s": 14, "metrics": {"run": 15}}))
            _set_status(d, "running", checkpoint="checkpoints/c")
        else:
            _set_status(d, "ok", checkpoint="checkpoints/c")

    follow(d, sleep=sleep)
    lines = [l.split()[1] if l.split()[0][0].isdigit() else l.strip()
             for l in capsys.readouterr().out.splitlines()[1:]]
    assert lines[-5:] == ["12", "13", "14", "CHECKPOINT", "15"]



def test_metrics_of_a_run_without_any(tmp_path, capsys):
    d = tmp_path / "r"
    d.mkdir()
    (d / "run_context.yaml").write_text("id: r_1\nname: r\n")
    assert cli.main(["metrics", str(d)]) == {}
    assert "no metrics yet" in capsys.readouterr().out



def test_follow_another_stream(tmp_path, capsys):
    d = _live_run(tmp_path)
    with open(d / "metrics" / "eval.jsonl", "w") as f:
        f.write(json.dumps({"_elapsed_s": 3, "ret": 7.5}) + "\n")
    _set_status(d, "ok")
    assert cli.main(["metrics", "follow", str(d), "eval:"]) == "ok"
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == ["time", "ret"] and out[1].split()[1] == "7.5"


def test_keys_group_by_their_prefix(tmp_path, monkeypatch):
    from runkit import metrics as m
    assert m._groups(["it", "loss/train", "loss", "loss/eval", "a/b/c", "a/b/d"]) == {
        "it": ["it"], "loss": ["loss/train", "loss", "loss/eval"], "a/b": ["a/b/c", "a/b/d"]}

    exp = Experiment("grp")

    @exp.run
    def run(cfg: Cfg, ctx: RunContext):
        for i in range(5):
            ctx.record(**{"it": i, "loss/train": 1 / (i + 1), "loss/eval": 2 / (i + 1)})

    d = exp.main([f"--root={tmp_path}"]).context.dir
    drawn = []
    real = m._groups
    monkeypatch.setattr(m, "_groups", lambda keys: drawn.append(real(keys)) or real(keys))
    plot_metrics(d, [])                                         # the overview: grouped panels
    assert drawn == [{"it": ["it"], "loss": ["loss/train", "loss/eval"]}]
    one = plot_metrics(d, ["loss/"])                            # a group on one axis
    assert one.name == "loss-train__loss-eval_vs_line.png"
    with pytest.raises(ValueError, match="no keys nope/"):
        plot_metrics(d, ["nope/"])



def test_keys_switch_the_stream_for_the_keys_after_them():
    assert cli.parse_keys([]) == []
    assert cli.parse_keys(["loss"]) == [("run", "loss")]
    assert cli.parse_keys(["eval:"]) == [("eval", "")]                    # all of it
    assert cli.parse_keys(["eval:", "ret", "len"]) == [("eval", "ret"), ("eval", "len")]
    assert cli.parse_keys(["loss", "eval:", "ret"]) == [("run", "loss"), ("eval", "ret")]
    assert cli.parse_keys(["eval:ret", "loss"]) == [("eval", "ret"), ("run", "loss")]  # no switch
    assert cli.parse_keys(["run:", "eval:", "ret"]) == [("run", ""), ("eval", "ret")]
    assert cli.parse_keys(["eval:", "loss/"]) == [("eval", "loss/")]
    assert cli.parse_keys(["run:a:b"]) == [("run", "a:b")]                # a key with a ':'
    assert cli.parse_keys([], stream="eval") == [("eval", "")]             # a stream file
    assert cli.parse_keys(["ret"], stream="eval") == [("eval", "ret")]


def test_path_can_be_the_metrics_folder_or_a_stream_file(run_dir, capsys):
    tables = cli.main(["metrics", str(run_dir / "metrics")])
    assert list(tables) == ["run", "eval"]
    tables = cli.main(["metrics", str(run_dir / "metrics" / "eval.jsonl"), "ret"])
    assert list(tables) == ["eval"] and [r["key"] for r in tables["eval"]] == ["ret"]
    png = cli.main(["metrics", "plot", str(run_dir / "metrics" / "eval.jsonl")])
    assert png.name == "eval-all_vs_line.png"
    with pytest.raises(SystemExit, match="is not a metrics stream"):
        cli.main(["metrics", str(run_dir / "config.yaml")])


def test_info_shows_several_streams_and_plot_mixes_them(run_dir, capsys):
    tables = cli.main(["metrics", str(run_dir), "loss", "eval:", "ret"])
    assert list(tables) == ["run", "eval"]
    assert [r["key"] for r in tables["eval"]] == ["ret"]
    png = cli.main(["metrics", "plot", str(run_dir), "loss", "eval:", "ret", "--x", "steps"])
    assert png.name == "loss__eval-ret_vs_steps.png"
