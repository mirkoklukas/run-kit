"""Behavior tests for the runkit lightweight runner.

Each test drives `autocli.main(run, argv)` with an explicit argv and a tmp
`--runs-dir`, then asserts on the produced run dir. Run with: `uv run --extra dev pytest`.
"""
import json
import pathlib
import re
from dataclasses import dataclass

import numpy as np
import pytest
import yaml

from runkit import RunContext, experiment
from runkit.autocli import main
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


@experiment(name="silent")
def none_run(cfg: Cfg, ctx: RunContext):
    return None


@experiment(name="arr")
def arr_run(cfg: Cfg, ctx: RunContext):
    return np.arange(3)


def _only_run_dir(runs_dir):
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(dirs) == 1, f"expected exactly one run dir, got {dirs}"
    return dirs[0]


def test_run_dir_structure_and_naming(tmp_path):
    main(run, ["seed=5", "--tag=t1", f"--runs-dir={tmp_path}"])
    d = _only_run_dir(tmp_path)
    # {date}_{time}_{name}_{tag}_{hex8}  (date=YYYY-MM-DD, time=HH-MM)
    assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_mock_t1_", d.name)
    hex8 = d.name.split("_")[-1]
    assert len(hex8) == 8
    assert (d / "config.yaml").is_file()
    assert (d / "marker.txt").read_text() == "5"
    retval = json.loads((d / "results" / "retval.json").read_text())
    assert retval == {"seed": 5, "lr": 3e-4}


def test_config_precedence(tmp_path):
    """dataclass defaults < config.yaml < key=value."""
    cfg_yaml = tmp_path / "c.yaml"
    cfg_yaml.write_text("lr: 0.01\nnote: from-yaml\n")
    main(run, [str(cfg_yaml), "seed=9", f"--runs-dir={tmp_path}"])
    cfg = yaml.safe_load((_only_run_dir(tmp_path) / "config.yaml").read_text())
    assert cfg == {"seed": 9, "lr": 0.01, "note": "from-yaml"}


def test_yaml_exponent_cast_to_annotated_float(tmp_path):
    """`1e-4` has no dot, so yaml loads it as a str; the float annotation casts it."""
    cfg_yaml = tmp_path / "c.yaml"
    cfg_yaml.write_text("lr: 1e-4\n")
    assert yaml.safe_load(cfg_yaml.read_text())["lr"] == "1e-4"   # the gotcha
    main(run, [str(cfg_yaml), f"--runs-dir={tmp_path}"])
    retval = json.loads((_only_run_dir(tmp_path) / "results" / "retval.json").read_text())
    assert retval["lr"] == 1e-4 and isinstance(retval["lr"], float)


def test_out_override_exact_path(tmp_path):
    out = tmp_path / "exactdir"
    main(run, [f"--out={out}"])
    assert (out / "config.yaml").is_file()
    assert (out / "marker.txt").is_file()


def test_out_collision_appends_timestamp_without_force(tmp_path):
    out = tmp_path / "exactdir"
    main(run, ["seed=1", f"--out={out}"])
    main(run, ["seed=2", f"--out={out}"])          # collides -> sibling with _{ts}
    assert (out / "marker.txt").read_text() == "1"  # first run untouched
    siblings = [p for p in tmp_path.iterdir() if p.is_dir() and p != out]
    assert len(siblings) == 1 and (siblings[0] / "marker.txt").read_text() == "2"


def test_force_replaces_existing_out_dir(tmp_path):
    out = tmp_path / "exactdir"
    main(run, ["seed=1", f"--out={out}"])
    (out / "stale.txt").write_text("old")           # leftover from the first run
    main(run, ["seed=2", f"--out={out}", "-f"])      # -f -> reuse exact path, fresh
    assert [p for p in tmp_path.iterdir() if p.is_dir()] == [out]  # no sibling
    assert (out / "marker.txt").read_text() == "2"
    assert not (out / "stale.txt").exists()          # dir was wiped, not merged


def test_short_flag_f_aliases_force(tmp_path):
    from runkit.config import split_argv
    assert split_argv(["-f"]) == ([], {"force": True}, [])


def test_unknown_flag_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main(run, [f"--runs-dir={tmp_path}", "--bogus=1"])


def test_dry_run_creates_nothing(tmp_path, capsys):
    main(run, ["seed=3", "--dry-run", f"--runs-dir={tmp_path}"])
    assert list(tmp_path.iterdir()) == []
    assert "dry-run" in capsys.readouterr().out


def test_none_return_writes_no_retval(tmp_path):
    main(none_run, [f"--runs-dir={tmp_path}"])
    assert not (_only_run_dir(tmp_path) / "results" / "retval.json").exists()


def test_ndarray_return_writes_npy(tmp_path):
    main(arr_run, [f"--runs-dir={tmp_path}"])
    d = _only_run_dir(tmp_path)
    assert not (d / "results" / "retval.json").exists()
    assert list(np.load(d / "results" / "retval.npy")) == [0, 1, 2]


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
