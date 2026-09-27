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
    assert by_key["it"] == {"key": "it", "rows": 20, "last": 19.0, "min": 0.0, "max": 19.0}
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
    rows = cli.main(["metrics", str(run_dir)])
    out = capsys.readouterr().out
    assert "metrics/run.jsonl" in out and "20 rows" in out and "also: eval" in out
    assert "20000" in out and "e+04" not in out                  # integers stay integers
    assert {r["key"] for r in rows} >= {"it", "steps", "loss"}

    png = cli.main(["plot", str(run_dir), "loss", "eval:ret", "--x", "steps"])
    assert capsys.readouterr().out.strip() == str(png) and png.is_file()

    monkeypatch.chdir(run_dir)                                   # RUN_DIR defaults to cwd
    assert cli.main(["metrics", "eval"])[0]["key"] == "_time"
    assert cli.main(["plot", "loss", "--x=it"]).name == "loss_vs_it.png"


def test_commands_explain_bad_input(run_dir, tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="no metrics stream 'nope'"):
        cli.main(["metrics", str(run_dir), "nope"])
    with pytest.raises(SystemExit, match="need an x key"):
        cli.main(["plot", str(run_dir), "loss", "eval:ret"])
    with pytest.raises(SystemExit, match="unknown option '--y'"):
        cli.main(["plot", str(run_dir), "loss", "--y", "a"])
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
    png = cli.main(["plot", str(ckpt_run), "it", "--start", "000001", "--end", "000002"])
    assert png.name == "it_vs_line_start000001_end000002.png"
    tail = cli.main(["plot", str(ckpt_run), "it", "eval:ret", "--x", "steps", "--start", "-1000"])
    assert tail.name == "it__eval-ret_vs_steps_start-1000.png" and tail.is_file()
    rows = cli.main(["metrics", str(ckpt_run), "--rows", "-5:"])
    assert {r["key"]: r for r in rows}["it"]["min"] == 25.0
    assert "5 rows" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="neither a number nor a checkpoint"):
        cli.main(["metrics", str(ckpt_run), "--start", "nope"])


def test_plot_without_keys_draws_every_numeric_key(run_dir, capsys):
    out = plot_metrics(run_dir, [])
    assert out.name == "all_vs_line.png" and out.stat().st_size > 0
    assert plot_metrics(run_dir, ["eval:"], x="steps").name == "eval-all_vs_steps.png"
    assert cli.main(["plot", str(run_dir)]).name == "all_vs_line.png"      # the command, no key
    assert cli.main(["plot", str(run_dir), "--rows", "-5:"]).name == "all_vs_line_rows-5-.png"   # ":" kept out of file names


def test_plot_all_needs_numbers(run_dir):
    with open(run_dir / "metrics" / "words.jsonl", "w") as f:
        f.write('{"_time": "t", "_elapsed_s": 0, "note": "hi"}\n')
    with pytest.raises(ValueError, match="no numeric keys"):
        plot_metrics(run_dir, ["words:"])
    with pytest.raises(ValueError, match="no stream 'nope'"):
        plot_metrics(run_dir, ["nope:"])
