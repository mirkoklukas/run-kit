"""Branching a run from a checkpoint (`--branch`), and eval taking a checkpoint.

A branch is a new run that starts from a checkpoint of an earlier one: its config
is the parent's with the command line on top, `meta.yaml` records where it came
from, and a body that declares `branch` gets the `Checkpoint`. An eval body
`evaluate(ckpt)` gets a checkpoint and writes into its `eval/`.
"""
import subprocess
import sys
from dataclasses import dataclass

import pytest
import yaml

from runkit import Experiment, RunContext, load_config, record, save_config
from runkit.runs import dir_hex, run_dirs


@dataclass
class Cfg:
    steps: int = 3
    lr: float = 0.1


def _exp(name="br"):
    """A run that counts on from where its branch left off, checkpointing
    each step; an eval that reads the count back."""
    exp = Experiment(name)

    @exp.run
    def run(cfg: Cfg, ctx: RunContext, branch=None):
        start = branch.info["steps"] if branch else 0
        seen = (branch.state / "count.txt").read_text() if branch else None
        for step in range(start + 1, start + cfg.steps + 1):
            with ctx.checkpoint("current") as ckpt:
                (ckpt.state / "count.txt").write_text(str(step))
                ckpt.info.update(steps=step)
        return {"start": start, "seen": seen, "branched": branch is not None}

    @exp.eval
    def evaluate(ckpt):
        n = int((ckpt.state / "count.txt").read_text())
        (ckpt.eval / "score.txt").write_text(str(10 * n))
        record(ckpt.eval / "episodes.jsonl", n=n)
        return ckpt

    return exp


def _go(exp, tmp_path, *args):
    return exp.main([*args, f"--root={tmp_path}"])


# -- branching ----------------------------------------------------------------------

def test_a_branch_continues_from_the_checkpoint(tmp_path, capsys):
    exp = _exp()
    parent = _go(exp, tmp_path, "lr=0.5")
    assert parent.retval == {"start": 0, "seen": None, "branched": False}
    capsys.readouterr()

    child = _go(exp, tmp_path, "--branch", dir_hex(parent.context.dir), "steps=2")
    assert child.retval == {"start": 3, "seen": "3", "branched": True}
    # the parent's config, the command line on top -- not the class defaults
    assert child.config == Cfg(steps=2, lr=0.5)
    meta = yaml.safe_load((child.context.dir / "meta.yaml").read_text())
    assert meta["branch"] == {"run": parent.context.id, "dir": str(parent.context.dir),
                              "checkpoint": "current", "index": 3, "steps": 3}
    err = capsys.readouterr().err
    assert f"{dir_hex(parent.context.dir)}:current (3 steps)" in err       # the banner
    assert "steps: 2" in err and "lr" not in err.split("config")[1].split("\n")[0]
    assert "1 more as in the parent" in err
    # the parent is not touched
    assert (parent.context.dir / "checkpoints" / "current" / "state" / "count.txt").read_text() == "3"


def test_branch_names_a_checkpoint_several_ways(tmp_path):
    exp = _exp()
    parent = _go(exp, tmp_path)
    ck = parent.context.dir / "checkpoints" / "current"
    for which in (str(parent.context.dir), f"{dir_hex(parent.context.dir)}:current",
                  str(ck), str(parent.context.dir / "checkpoints" / "latest"),
                  f"{parent.context.dir}:latest"):
        assert _go(exp, tmp_path, "--branch", which).retval["start"] == 3, which


def test_a_branch_from_python(tmp_path):
    exp = _exp()
    parent = _go(exp, tmp_path)
    run = exp.roles["run"]
    child = run(Cfg(steps=1), root=tmp_path, branch=dir_hex(parent.context.dir))
    assert child.retval["start"] == 3 and child.config == Cfg(steps=1)   # cfg as given


def test_what_cannot_be_branched_is_refused(tmp_path):
    exp = _exp()
    parent = _go(exp, tmp_path)
    hexid = dir_hex(parent.context.dir)
    with pytest.raises(SystemExit, match="no checkpoint 'best'.*current"):
        _go(exp, tmp_path, "--branch", f"{hexid}:best")
    with pytest.raises(SystemExit, match="needs a value"):
        _go(exp, tmp_path, "--branch")

    plain = Experiment("plain")                        # a body that takes no `branch`

    @plain.run
    def run(cfg: Cfg, ctx: RunContext):
        with ctx.checkpoint("current") as ckpt:
            (ckpt.state / "x").write_text("1")

    first = _go(plain, tmp_path)
    with pytest.raises(SystemExit, match="takes no `branch`"):
        _go(plain, tmp_path, "--branch", dir_hex(first.context.dir))

    empty = Experiment("empty")                        # a checkpoint with nothing saved

    @empty.run
    def run2(cfg: Cfg, ctx: RunContext, branch=None):
        with ctx.checkpoint("current"):
            pass

    e = _go(empty, tmp_path)
    with pytest.raises(SystemExit, match="nothing saved in its state/"):
        _go(empty, tmp_path, "--branch", dir_hex(e.context.dir))

    none = Experiment("none")                          # a run with no checkpoint

    @none.run
    def run3(cfg: Cfg, ctx: RunContext, branch=None):
        pass

    n = _go(none, tmp_path)
    with pytest.raises(SystemExit, match="has no complete checkpoint"):
        _go(none, tmp_path, "--branch", str(n.context.dir))


# -- eval takes a checkpoint -----------------------------------------------------------

def test_eval_gets_the_latest_checkpoint_and_writes_into_it(tmp_path):
    exp = _exp()
    r = _go(exp, tmp_path)
    ckpt = _go(exp, tmp_path, "eval")
    assert ckpt.dir == r.context.dir / "checkpoints" / "current" and ckpt.index == 3
    assert (ckpt.eval / "score.txt").read_text() == "30"
    assert yaml.safe_load((ckpt.eval / "episodes.jsonl").read_text())["n"] == 3
    assert ckpt.run == r.context.dir


def test_eval_picks_a_checkpoint_and_skips_runs_without_one(tmp_path):
    exp = _exp()
    r = _go(exp, tmp_path)
    assert _go(exp, tmp_path, "eval", f"{dir_hex(r.context.dir)}:current").name == "current"

    @dataclass
    class C:
        fail: bool = False

    ex = Experiment("fl")

    @ex.run
    def run(cfg: C, ctx: RunContext):
        with ctx.checkpoint("current") as ckpt:
            (ckpt.state / "x").write_text("1")
        if cfg.fail:
            raise ValueError("boom")

    @ex.eval
    def evaluate(ckpt):
        return ckpt.run

    with pytest.raises(ValueError):
        _go(ex, tmp_path, "fail=true")
    failed = run_dirs(tmp_path, "fl")[-1]             # the newest (same second: by creation)
    assert _go(ex, tmp_path, "eval") == failed         # a failed run: its checkpoint

    none = Experiment("nock")

    @none.run
    def run2(cfg: C, ctx: RunContext):
        pass

    @none.eval
    def evaluate2(ckpt):
        return ckpt

    _go(none, tmp_path)
    with pytest.raises(SystemExit, match="has a complete checkpoint"):
        _go(none, tmp_path, "eval")


def test_saving_a_checkpoint_again_empties_its_eval(tmp_path):
    exp = _exp()
    parent = _go(exp, tmp_path)
    ckpt = _go(exp, tmp_path, "eval")
    assert ckpt.eval.is_dir()
    child = _go(exp, tmp_path, "--branch", dir_hex(parent.context.dir))
    child_ckpt = child.context.dir / "checkpoints" / "current"
    assert not (child_ckpt / "eval").exists()          # a new checkpoint: no eval yet
    assert (parent.context.dir / "checkpoints" / "current" / "eval").is_dir()


def test_runkit_eval_of_a_checkpoint_needs_no_experiment(tmp_path):
    """`runkit eval <checkpoint>`: the run's meta.yaml names the experiment."""
    (tmp_path / "e.py").write_text(
        "from dataclasses import dataclass\n"
        "from runkit import Experiment, RunContext\n"
        "exp = Experiment('e')\n"
        "@dataclass\n"
        "class Cfg:\n"
        "    n: int = 7\n"
        "@exp.run\n"
        "def run(cfg: Cfg, ctx: RunContext):\n"
        "    with ctx.checkpoint('best') as ckpt:\n"
        "        (ckpt.state / 'n.txt').write_text(str(cfg.n))\n"
        "@exp.eval\n"
        "def evaluate(ckpt):\n"
        "    (ckpt.eval / 'got.txt').write_text((ckpt.state / 'n.txt').read_text())\n")
    go = lambda *a: subprocess.run([sys.executable, "-m", "runkit", *a], cwd=tmp_path,
                                   capture_output=True, text=True)
    out = go("run", "e.py", "--root=runs")
    assert out.returncode == 0, out.stderr
    ckpt = next((tmp_path / "runs" / "e").glob("20*")) / "checkpoints" / "best"
    out = go("eval", str(ckpt))
    assert out.returncode == 0, out.stderr
    assert (ckpt / "eval" / "got.txt").read_text() == "7"
    out = go("eval", str(tmp_path / "nowhere"))
    assert out.returncode != 0


# -- the pieces ---------------------------------------------------------------------

def test_a_checkpoint_saved_at_its_top_still_branches(tmp_path):
    """A body that saves into `ckpt.dir` (as before `state/`): no empty state/ is
    left, and `ckpt.state` is the checkpoint folder."""
    exp = Experiment("top")

    @exp.run
    def run(cfg: Cfg, ctx: RunContext, branch=None):
        if branch:
            return (branch.state / "w.txt").read_text()
        with ctx.checkpoint("current") as ckpt:
            (ckpt.dir / "w.txt").write_text("hi")

    parent = _go(exp, tmp_path)
    ck = parent.context.dir / "checkpoints" / "current"
    assert sorted(p.name for p in ck.iterdir()) == ["checkpoint.yaml", "w.txt"]
    assert _go(exp, tmp_path, "--branch", dir_hex(parent.context.dir)).retval == "hi"


def test_save_and_load_config(tmp_path):
    @dataclass
    class Inner:
        lim: tuple = ((-1, 1), (0, 2))

    @dataclass
    class Outer:
        inner: Inner = None
        w: dict = None
        lr: float = 1e-4

    cfg = Outer(inner=Inner(lim=((-3, 3), (0, 1))), w={"a": {"b": 1}}, lr=3e-4)
    save_config(cfg, tmp_path / "config.yaml")
    assert load_config(Outer, tmp_path / "config.yaml") == cfg
    (tmp_path / "bad.yaml").write_text("nope: 1\n")
    with pytest.raises(ValueError, match="unknown field 'nope'"):
        load_config(Outer, tmp_path / "bad.yaml")


def test_record_appends_rows_and_refuses_runkits_keys(tmp_path):
    path = tmp_path / "a" / "b.jsonl"
    record(path, x=1)
    record(path, x=2, note="hi")
    rows = [yaml.safe_load(l) for l in path.read_text().splitlines()]
    assert [r["x"] for r in rows] == [1, 2] and all("_time" in r for r in rows)
    with pytest.raises(ValueError, match="runkit's"):
        record(path, _x=1)


def test_old_eval_signature_still_opens_a_run(tmp_path):
    """`evaluate(cfg, ctx)`, from before: a run, not a checkpoint."""
    exp = Experiment("old")

    @exp.run
    def run(cfg: Cfg, ctx: RunContext):
        pass

    @exp.eval
    def evaluate(cfg: Cfg, ctx: RunContext):
        return cfg, ctx.dir

    r = _go(exp, tmp_path, "lr=0.2")
    assert _go(exp, tmp_path, "eval") == (Cfg(lr=0.2), r.context.dir)
