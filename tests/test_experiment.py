"""The `Experiment` object, its verbs, and the `runkit <experiment> [verb]` CLI.

Each test builds its experiment fresh (roles register once per experiment) and
drives it with an explicit argv against a tmp `--root`.
"""
import pathlib
import subprocess
import sys
from dataclasses import dataclass

import pytest
import yaml

from runkit import Experiment, Run, RunContext, experiment, load_run, main, select_run
from runkit import cli
from runkit.runs import RunNotFound
from runkit.settings import resolve_root


@dataclass
class Cfg:
    seed: int = 1


def _make(name="demo"):
    """An experiment with all three roles; eval/viz report what they were handed."""
    exp = Experiment(name)

    @exp.run
    def run(cfg: Cfg, ctx: RunContext):
        (ctx.out / "seed.txt").write_text(str(cfg.seed))
        if cfg.seed < 0:
            raise ValueError("negative seed")
        return {"seed": cfg.seed}

    @exp.eval
    def evaluate(cfg: Cfg, ctx: RunContext):
        (ctx.out / "eval").mkdir(exist_ok=True)
        (ctx.out / "eval" / "score.txt").write_text(str(cfg.seed * 10))
        return ("eval", cfg.seed, ctx.dir)

    @exp.viz
    def show(cfg: Cfg, ctx: RunContext):
        return ("viz", cfg.seed, ctx.dir)

    return exp


def _run(exp, tmp_path, *args):
    return exp.main([*args, f"--root={tmp_path}"])


def test_run_is_the_default_verb(tmp_path):
    exp = _make()
    r = _run(exp, tmp_path, "seed=3")
    assert isinstance(r, Run) and r.config == Cfg(seed=3)
    assert r.context.dir.parent == tmp_path / "demo"
    assert _run(exp, tmp_path, "run", "seed=4").config == Cfg(seed=4)


def test_viz_opens_the_latest_run(tmp_path):
    exp = _make()
    _run(exp, tmp_path, "seed=1")
    second = _run(exp, tmp_path, "seed=2")
    assert _run(exp, tmp_path, "viz") == ("viz", 2, second.context.dir)


def test_eval_opens_the_latest_ok_run_and_writes_under_out(tmp_path):
    exp = _make()
    good = _run(exp, tmp_path, "seed=5")
    with pytest.raises(ValueError):
        _run(exp, tmp_path, "seed=-1")               # a later run that failed
    assert _run(exp, tmp_path, "eval") == ("eval", 5, good.context.dir)
    assert (good.context.out / "eval" / "score.txt").read_text() == "50"
    # viz does not skip the failed one: looking at it is the point
    assert _run(exp, tmp_path, "viz")[1] == -1


def test_pick_a_run_by_hex_prefix_or_path(tmp_path):
    exp = _make()
    first = _run(exp, tmp_path, "seed=1")
    _run(exp, tmp_path, "seed=2")
    hex8 = first.context.id.rsplit("_", 1)[-1]
    assert _run(exp, tmp_path, "viz", hex8[:4])[1] == 1
    assert _run(exp, tmp_path, "viz", str(first.context.dir))[1] == 1
    assert _run(exp, tmp_path, "viz", str(tmp_path / "demo" / "latest"))[1] == 2


def test_selection_errors_exit_with_a_message(tmp_path):
    exp = _make()
    with pytest.raises(SystemExit, match="no runs of 'demo'"):
        _run(exp, tmp_path, "viz")
    _run(exp, tmp_path)
    with pytest.raises(SystemExit, match="whose id starts with it"):
        _run(exp, tmp_path, "viz", "zzzz")
    with pytest.raises(SystemExit, match="no key=value"):
        _run(exp, tmp_path, "viz", "seed=3")


def test_a_run_of_another_experiment_is_refused(tmp_path):
    other = _run(_make("other"), tmp_path)
    with pytest.raises(RunNotFound, match="a run of 'other', not 'demo'"):
        select_run(tmp_path, "demo", str(other.context.dir))


def test_unregistered_verb(tmp_path):
    exp = Experiment("bare")

    @exp.run
    def run(cfg: Cfg, ctx: RunContext):
        pass

    with pytest.raises(SystemExit, match="no viz function registered"):
        _run(exp, tmp_path, "viz")


def test_a_role_registers_once():
    exp = _make()
    with pytest.raises(ValueError, match="already has a viz"):
        exp.viz(lambda cfg, ctx: None)


def test_eval_and_viz_are_callable_from_python(tmp_path):
    exp = _make()
    r = _run(exp, tmp_path, "seed=7")
    assert exp.roles["viz"](root=tmp_path) == ("viz", 7, r.context.dir)


def test_viz_without_annotation_uses_the_runs_config(tmp_path):
    exp = Experiment("plain")

    @exp.run
    def run(cfg: Cfg, ctx: RunContext):
        pass

    @exp.viz
    def show(cfg, ctx):
        return cfg

    _run(exp, tmp_path, "seed=8")
    assert _run(exp, tmp_path, "viz") == Cfg(seed=8)


def test_main_run_picks_up_the_experiments_roles(tmp_path):
    """`main(run)` at the bottom of a file keeps working, verbs included."""
    exp = _make()
    main(exp.roles["run"], ["seed=2", f"--root={tmp_path}"])
    assert main(exp.roles["run"], ["viz", f"--root={tmp_path}"])[1] == 2


def test_main_with_plain_viz_function(tmp_path):
    @experiment(name="fn")
    def run(cfg: Cfg, ctx: RunContext):
        pass

    def show(cfg: Cfg, ctx: RunContext):
        return cfg.seed

    main(run, ["seed=6", f"--root={tmp_path}"])
    assert main(run, ["viz", f"--root={tmp_path}"], viz=show) == 6


def test_load_run_thaws_config_and_retval(tmp_path):
    r = _run(_make(), tmp_path, "seed=9")
    loaded = load_run(r.context.dir, Cfg)
    assert loaded == r                                # same Run, rebuilt from disk


def test_verb_help(capsys):
    exp = _make()
    exp.main(["viz", "--help"])
    out = capsys.readouterr().out
    assert "usage: <experiment> viz [RUN]" in out and "verbs: run, eval, viz" in out
    exp.main(["--help"])
    assert "seed=1" in capsys.readouterr().out       # run's help: the config fields


# -- runkit <experiment> [verb] ---------------------------------------------

_SRC = '''
from dataclasses import dataclass
from runkit import Experiment, RunContext

exp = Experiment("cli")

@dataclass
class Cfg:
    seed: int = 1

@exp.run
def run(cfg: Cfg, ctx: RunContext):
    return {"seed": cfg.seed}

@exp.viz
def show(cfg: Cfg, ctx: RunContext):
    print(f"VIZ seed={cfg.seed}")

if __name__ == "__main__":
    exp.main()
'''


def _write_pkg(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "exp.py").write_text(_SRC)


_RUNKIT = [sys.executable, "-c", "from runkit.cli import app; app()"]


@pytest.mark.parametrize("launch", [
    ["python", "pkg/exp.py"],
    ["python", "-m", "pkg.exp"],
    ["runkit", "pkg/exp.py"],
    ["runkit", "pkg.exp"],
])
def test_every_way_in_takes_the_same_verbs(tmp_path, launch):
    """`python <target> <verb> ...` and `runkit <verb> <target> ...` agree."""
    _write_pkg(tmp_path)
    target = launch[1:]

    def call(verb, *args):
        cmd = ([sys.executable, *target, verb] if launch[0] == "python"
               else [*_RUNKIT, verb, *target])
        return subprocess.run([*cmd, *args, "--root=runs"], cwd=tmp_path,
                              check=True, capture_output=True, text=True)

    call("run", "seed=4")
    assert "VIZ seed=4" in call("viz").stdout
    meta = yaml.safe_load(next((tmp_path / "runs" / "cli").glob("2*/meta.yaml")).read_text())
    assert meta["module"] == ("pkg.exp" if launch[-1] == "pkg.exp" else None)


def test_cli_rejects_two_experiments_in_one_file(tmp_path):
    f = tmp_path / "two.py"
    f.write_text("from runkit import Experiment\na = Experiment('a')\nb = Experiment('b')\n")
    with pytest.raises(SystemExit, match="one experiment per file"):
        cli.main(["run", str(f)])


# -- experiment.toml and `runkit root` ----------------------------------------


def test_root_resolution_order(tmp_path, monkeypatch):
    """--root > [env.<stem>] > [env] > ./runs; toml paths relative to the toml."""
    lab = tmp_path / "lab"
    (lab / "rl").mkdir(parents=True)
    exp_file = lab / "rl" / "policy.py"
    exp_file.write_text("")
    assert resolve_root(exp_file, "policy") == pathlib.Path("runs")     # no toml

    (lab / "experiment.toml").write_text(
        '[env]\nroot = "../runs/lab"\n\n[env.policy]\nroot = "ctk:policy_runs"\n')
    assert resolve_root(exp_file, "other") == lab / "../runs/lab"      # [env], found upward
    monkeypatch.setenv("RUNKIT_PATH_CTK", str(tmp_path / "ctk"))
    assert resolve_root(exp_file, "policy") == tmp_path / "ctk" / "policy_runs"
    assert resolve_root(exp_file, "policy", explicit="x") == pathlib.Path("x")


def test_a_run_goes_where_experiment_toml_says(tmp_path):
    _write_pkg(tmp_path)
    (tmp_path / "pkg" / "experiment.toml").write_text('[env]\nroot = "../from_toml"\n')
    subprocess.run([sys.executable, "pkg/exp.py", "seed=3"], cwd=tmp_path,
                   check=True, capture_output=True)
    assert len(list((tmp_path / "from_toml" / "cli").glob("2*"))) == 1
    viz = subprocess.run([sys.executable, "pkg/exp.py", "viz"], cwd=tmp_path,
                         check=True, capture_output=True, text=True)
    assert "VIZ seed=3" in viz.stdout                   # eval/viz look there too


def test_runkit_root_prints_the_resolved_root(tmp_path, monkeypatch, capsys):
    _write_pkg(tmp_path)
    (tmp_path / "pkg" / "experiment.toml").write_text(
        '[env]\nroot = "shared"\n\n[env.exp]\nroot = "mine"\n')
    monkeypatch.chdir(tmp_path / "pkg")

    assert cli.main(["root"]) == tmp_path / "pkg" / "shared"        # from cwd: [env]
    out, err = capsys.readouterr()
    assert out == f"{tmp_path / 'pkg' / 'shared'}\n"                # stdout: the path only
    assert "does not exist yet" in err                               # the note: stderr

    # with an experiment: its runs folder, under its own override
    assert cli.main(["root", "exp.py"]) == tmp_path / "pkg" / "mine" / "cli"


def test_runkit_takes_the_verb_first(capsys):
    with pytest.raises(SystemExit, match="the verb comes first: runkit viz exp.py"):
        cli.main(["exp.py", "viz"])
    with pytest.raises(SystemExit, match="unknown runkit option '--detach'"):
        cli.main(["run", "--detach", "exp.py"])
    with pytest.raises(SystemExit, match="usage: runkit viz <experiment>"):
        cli.main(["viz"])


def test_root_and_latest_print_paths(tmp_path, capsys):
    exp = _make()
    assert _run(exp, tmp_path, "root") == tmp_path / "demo"           # not there yet
    assert "does not exist yet" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="no runs of 'demo'"):
        _run(exp, tmp_path, "latest")

    _run(exp, tmp_path, "seed=1")
    second = _run(exp, tmp_path, "seed=2")
    capsys.readouterr()
    assert _run(exp, tmp_path, "latest") == second.context.dir
    assert capsys.readouterr().out == f"{second.context.dir}\n"       # stdout: the path only
    assert _run(exp, tmp_path, "root") == tmp_path / "demo"


def test_latest_via_runkit_follows_experiment_toml(tmp_path):
    """`cd "$(runkit exp.py latest)"`: same root a run used, from any folder."""
    _write_pkg(tmp_path)
    (tmp_path / "pkg" / "experiment.toml").write_text('[env]\nroot = "../from_toml"\n')
    subprocess.run([sys.executable, "pkg/exp.py"], cwd=tmp_path, check=True,
                   capture_output=True)
    out = subprocess.run([*_RUNKIT, "latest", "exp.py"], cwd=tmp_path / "pkg",
                         check=True, capture_output=True, text=True).stdout
    assert pathlib.Path(out.strip()).parent == (tmp_path / "from_toml" / "cli").resolve()
