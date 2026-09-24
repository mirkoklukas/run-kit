"""Behavior tests for the runkit lightweight runner.

Each test drives `autocli.main(run, argv)` with an explicit argv and a tmp
`--root`, then asserts on the produced run dir. Run with: `uv run --extra dev pytest`.
"""
import json
import pathlib
import re
from dataclasses import dataclass

import numpy as np
import pytest
import yaml

from runkit import Run, RunContext, experiment, init_run
from runkit.autocli import main
from runkit.config import build_cfg
from runkit.runs import dir_hex
from runkit.utils import resolve_config_path


@dataclass
class Cfg:
    seed: int = 1
    lr: float = 3e-4
    note: str = "hi"


@experiment(name="mock")
def run(cfg: Cfg, ctx: RunContext):
    (ctx.out / "marker.txt").write_text(str(cfg.seed))
    return {"seed": cfg.seed, "lr": cfg.lr}


@experiment(name="deferred")
def str_ann_run(cfg: "Cfg", ctx: RunContext):
    """As under `from __future__ import annotations`: `cfg`'s annotation is a string."""
    return {"seed": cfg.seed}


@experiment(name="silent")
def none_run(cfg: Cfg, ctx: RunContext):
    return None


@experiment(name="arr")
def arr_run(cfg: Cfg, ctx: RunContext):
    return np.arange(3)


@experiment(name="boom")
def failing_run(cfg: Cfg, ctx: RunContext):
    raise ValueError("bad shape")


@experiment(name="ctrlc")
def interrupted_run(cfg: Cfg, ctx: RunContext):
    raise KeyboardInterrupt


def _run_dirs(root):
    """All run dirs under `root`, across experiments -- `latest` links excluded."""
    return [p for p in root.glob("*/*") if p.is_dir() and not p.is_symlink()]


def _only_run_dir(root):
    dirs = _run_dirs(root)
    assert len(dirs) == 1, f"expected exactly one run dir, got {dirs}"
    return dirs[0]


def test_run_dir_structure_and_naming(tmp_path):
    main(run, ["seed=5", "--tag=t1", f"--root={tmp_path}"])
    d = _only_run_dir(tmp_path)
    # {root}/{name}/{date}_{time}_{hex8}_{tag}  (date=YYYY-MM-DD, time=HH-MM-SS)
    assert d.parent == tmp_path / "mock"
    assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[0-9a-f]{8}_t1$", d.name)
    assert (d / "config.yaml").is_file()
    assert (d / "out" / "marker.txt").read_text() == "5"   # the body writes under out/
    retval = json.loads((d / "retval.json").read_text())
    assert retval == {"seed": 5, "lr": 3e-4}


def test_config_precedence(tmp_path):
    """dataclass defaults < config.yaml < key=value."""
    cfg_yaml = tmp_path / "c.yaml"
    cfg_yaml.write_text("lr: 0.01\nnote: from-yaml\n")
    main(run, [str(cfg_yaml), "seed=9", f"--root={tmp_path}"])
    cfg = yaml.safe_load((_only_run_dir(tmp_path) / "config.yaml").read_text())
    assert cfg == {"seed": 9, "lr": 0.01, "note": "from-yaml"}


def test_yaml_exponent_cast_to_annotated_float(tmp_path):
    """`1e-4` has no dot, so yaml loads it as a str; the float annotation casts it."""
    cfg_yaml = tmp_path / "c.yaml"
    cfg_yaml.write_text("lr: 1e-4\n")
    assert yaml.safe_load(cfg_yaml.read_text())["lr"] == "1e-4"   # the gotcha
    main(run, [str(cfg_yaml), f"--root={tmp_path}"])
    retval = json.loads((_only_run_dir(tmp_path) / "retval.json").read_text())
    assert retval["lr"] == 1e-4 and isinstance(retval["lr"], float)


def test_short_flags_are_flags(tmp_path):
    """`-x` parses as a flag, so a stray one is rejected rather than taken as a
    config path."""
    from runkit.config import split_argv
    assert split_argv(["-x"]) == ([], {"x": True}, [])


def test_unknown_flag_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main(run, [f"--root={tmp_path}", "--bogus=1"])


def test_none_return_writes_no_retval(tmp_path):
    main(none_run, [f"--root={tmp_path}"])
    assert not (_only_run_dir(tmp_path) / "retval.json").exists()


def test_ndarray_return_writes_npy(tmp_path):
    main(arr_run, [f"--root={tmp_path}"])
    d = _only_run_dir(tmp_path)
    assert not (d / "retval.json").exists()
    assert list(np.load(d / "retval.npy")) == [0, 1, 2]


def test_resolve_config_path():
    assert resolve_config_path("foo.yaml", "/exp") == "foo.yaml"          # bare -> cwd
    assert resolve_config_path("cwd:foo.yaml", "/exp") == "foo.yaml"      # explicit cwd
    assert resolve_config_path("exp:foo.yaml", "/exp") == "/exp/foo.yaml"  # experiment dir
    assert resolve_config_path("/abs/foo.yaml", "/exp") == "/abs/foo.yaml"  # absolute
    assert resolve_config_path("C:\\x\\foo.yaml", "/exp") == "C:\\x\\foo.yaml"  # not a scheme
    assert resolve_config_path("weird:foo.yaml", "/exp") == "weird:foo.yaml"  # undefined scheme
    with pytest.raises(ValueError):
        resolve_config_path("exp:foo.yaml", None)


def test_resolve_config_path_env_scheme(monkeypatch):
    monkeypatch.setenv("RUNKIT_PATH_CTK", "/ctk")
    assert resolve_config_path("ctk:configs/x.yaml", "/exp") == "/ctk/configs/x.yaml"
    monkeypatch.setenv("RUNKIT_PATH_HOMEISH", "~/base")
    assert resolve_config_path("homeish:x.yaml", None) == str(
        pathlib.Path(pathlib.Path.home() / "base" / "x.yaml"))


def test_resolve_config_path_undefined_scheme_warns(capsys, monkeypatch):
    monkeypatch.delenv("RUNKIT_PATH_CTK", raising=False)
    assert resolve_config_path("ctk:x.yaml", "/exp") == "ctk:x.yaml"  # env unset -> literal
    assert "no base is defined" in capsys.readouterr().err


def _yaml(d, name):
    return yaml.safe_load((d / name).read_text())


def test_run_context_and_meta_are_written(tmp_path):
    main(run, ["--tag=t1", f"--root={tmp_path}"])
    d = _only_run_dir(tmp_path)
    rc = _yaml(d, "run_context.yaml")
    assert rc == {"id": f"mock_{dir_hex(d)}", "name": "mock"}
    meta = _yaml(d, "meta.yaml")
    assert meta["tag"] == "t1"
    assert meta["script"].endswith("test_runner.py")   # where the @experiment lives


def test_meta_tag_is_none_without_a_tag(tmp_path):
    main(run, [f"--root={tmp_path}"])
    assert _yaml(_only_run_dir(tmp_path), "meta.yaml")["tag"] is None


def test_status_ok_on_success(tmp_path):
    main(run, [f"--root={tmp_path}"])
    d = _only_run_dir(tmp_path)
    st = _yaml(d, "status.yaml")
    assert st["status"] == "ok"
    assert st["error"] is None
    assert st["started"] and st["ended"]
    assert st["duration_s"] >= 0
    assert not (d / "traceback.txt").exists()


def test_status_failed_and_traceback_on_raise(tmp_path):
    with pytest.raises(ValueError, match="bad shape"):     # propagates untouched
        main(failing_run, [f"--root={tmp_path}"])
    d = _only_run_dir(tmp_path)
    st = _yaml(d, "status.yaml")
    assert st["status"] == "failed"
    assert st["error"] == "ValueError: bad shape"
    assert st["duration_s"] >= 0
    tb = (d / "traceback.txt").read_text()
    assert "ValueError: bad shape" in tb and "failing_run" in tb
    assert not (d / "retval.json").exists()    # nothing to dump


def test_status_interrupted_on_ctrl_c(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        main(interrupted_run, [f"--root={tmp_path}"])
    st = _yaml(_only_run_dir(tmp_path), "status.yaml")
    assert st["status"] == "interrupted"
    assert st["error"] is None                             # a Ctrl-C is not a crash


def test_init_run_writes_no_status(tmp_path):
    """`status` is the body's lifecycle; init_run alone has nothing running."""
    ctx = init_run(Cfg(), name="x", tag=None, root=tmp_path)
    assert (ctx.dir / "run_context.yaml").is_file()
    assert not (ctx.dir / "status.yaml").exists()


def test_run_returns_a_run(tmp_path):
    r = main(run, ["seed=5", f"--root={tmp_path}"])
    assert isinstance(r, Run)
    assert r.config == Cfg(seed=5)              # what it ran with, as resolved
    assert r.context.dir == _only_run_dir(tmp_path)
    assert r.context.name == "mock"
    assert r.retval == {"seed": 5, "lr": 3e-4}


def test_run_returns_a_run_with_no_retval(tmp_path):
    r = main(none_run, [f"--root={tmp_path}"])
    assert r.context.dir.is_dir() and r.retval is None


def test_driving_runs_from_python_yields_their_dirs(tmp_path):
    """The caller-facing point of `Run`: a sweep knows where its runs landed."""
    runs = [run(Cfg(seed=s), tag=f"s{s}", root=tmp_path) for s in (1, 2)]
    assert [(r.config.seed, r.retval["seed"]) for r in runs] == [(1, 1), (2, 2)]
    assert {r.context.dir for r in runs} == set(_run_dirs(tmp_path))


def test_run_config_survives_the_cli_path(tmp_path):
    """The caller of `main` never sees the Config `autocli` built -- unless
    `Run` carries it."""
    r = main(run, ["seed=9", "lr=1e-4", f"--root={tmp_path}"])
    assert r.config == Cfg(seed=9, lr=1e-4)
    assert r.config == build_cfg(Cfg, yaml.safe_load(
        (r.context.dir / "config.yaml").read_text()))   # same as what was frozen


def test_string_annotation_resolves_to_the_config_class(tmp_path):
    r = main(str_ann_run, ["seed=4", f"--root={tmp_path}"])
    assert r.config == Cfg(seed=4)


def test_runkit_owns_the_top_level_and_the_body_owns_out(tmp_path):
    r = main(run, [f"--root={tmp_path}"])
    assert r.context.out == r.context.dir / "out"
    top = {p.name for p in r.context.dir.iterdir()}
    assert top == {"config.yaml", "run_context.yaml", "meta.yaml", "status.yaml",
                   "retval.json", "out"}
    assert [p.name for p in r.context.out.iterdir()] == ["marker.txt"]


def test_latest_points_at_the_newest_run(tmp_path):
    first = run(Cfg(seed=1), root=tmp_path)
    second = run(Cfg(seed=2), root=tmp_path)
    latest = tmp_path / "mock" / "latest"
    assert latest.is_symlink()
    assert latest.readlink() == pathlib.Path(second.context.dir.name)   # relative
    assert latest.resolve() == second.context.dir != first.context.dir
    assert [p.name for p in (tmp_path / "mock").iterdir()
            if p.name.startswith(".")] == []                          # no tmp link left



def test_meta_records_the_module(tmp_path):
    main(run, [f"--root={tmp_path}"])
    assert _yaml(_only_run_dir(tmp_path), "meta.yaml")["module"] == __name__


_EXP_SRC = '''
from dataclasses import dataclass
from runkit import RunContext, experiment, main

@dataclass
class Cfg:
    seed: int = 1

@experiment(name="sub")
def run(cfg: Cfg, ctx: RunContext):
    pass

if __name__ == "__main__":
    main(run)
'''


@pytest.mark.parametrize("how, module", [("-m", "pkg.exp"), ("script", None)])
def test_meta_module_under_python_m_and_as_a_script(tmp_path, how, module):
    """`python -m pkg.exp` records the real name, not `__main__`; a plain
    script has no importable name."""
    import subprocess
    import sys
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "exp.py").write_text(_EXP_SRC)
    target = ["-m", "pkg.exp"] if how == "-m" else [str(tmp_path / "pkg" / "exp.py")]
    subprocess.run([sys.executable, *target, f"--root={tmp_path / 'runs'}"],
                   cwd=tmp_path, check=True, capture_output=True)
    assert _yaml(_only_run_dir(tmp_path / "runs"), "meta.yaml")["module"] == module


def test_banner_shows_only_what_differs_from_the_defaults(tmp_path, capsys):
    main(run, ["seed=5", "--tag=t1", f"--root={tmp_path}"])
    err = capsys.readouterr().err
    assert "seed: 5" in err and "lr:" not in err            # lr is at its default
    assert "2 more at their defaults" in err and "t1" in err
    main(run, [f"--root={tmp_path}"])
    assert "all 3 fields at their defaults" in capsys.readouterr().err


def test_config_changes_flattens_nested_fields():
    from dataclasses import field
    from runkit.utils import config_changes

    @dataclass
    class Optim:
        lr: float = 1e-3
        warmup: int = 0

    @dataclass
    class Nested:
        steps: int                                  # no default: always shown
        optim: Optim = field(default_factory=Optim)

    changes, n = config_changes(Nested(steps=10, optim=Optim(lr=1e-4)))
    assert changes == {"steps": 10, "optim.lr": 1e-4} and n == 3


def test_closing_line_says_how_it_went(tmp_path, capsys):
    main(run, [f"--root={tmp_path}"])
    assert re.search(r"✓ mock_[0-9a-f]{8}\s+ok in \d", capsys.readouterr().err)
    with pytest.raises(ValueError):
        main(failing_run, [f"--root={tmp_path}"])
    err = capsys.readouterr().err
    assert "failed after" in err and "ValueError: bad shape" in err
    assert "traceback.txt" in "".join(err.split())         # the path may wrap
    with pytest.raises(KeyboardInterrupt):
        main(interrupted_run, [f"--root={tmp_path}"])
    assert "interrupted after" in capsys.readouterr().err


def test_a_tag_with_underscores_stays_whole(tmp_path):
    """The tag is the remainder after the fixed-width part, underscores and all."""
    r = run(Cfg(), tag="lr_sweep_a", root=tmp_path)
    assert r.context.dir.name[29:] == "lr_sweep_a"
    assert r.context.id == f"mock_{dir_hex(r.context.dir)}"
