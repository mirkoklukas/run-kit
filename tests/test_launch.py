"""experiment.toml's extras and vars: read before the import, applied by a relaunch."""
import json
from dataclasses import dataclass

import pytest
import yaml

from runkit import Experiment, RunContext
from runkit import launch
from runkit.settings import read_env


TOML = '''
[env]
root = "runs"
extras = ["mjx"]
vars = { XLA_PYTHON_CLIENT_PREALLOCATE = false, A = "base" }

[env.policy]
extras = ["mjx", "sb3"]
vars = { A = "mine", N = 3 }
'''


@pytest.fixture
def lab(tmp_path):
    """tmp/pyproject.toml, tmp/lab/rl/experiment.toml, tmp/lab/rl/{policy,other}.py"""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    rl = tmp_path / "lab" / "rl"
    rl.mkdir(parents=True)
    (tmp_path / "lab" / "__init__.py").write_text("")
    (rl / "__init__.py").write_text("")
    (rl / "experiment.toml").write_text(TOML)
    for name in ("policy", "other"):
        (rl / f"{name}.py").write_text("")
    return tmp_path


def test_lists_replace_and_tables_merge(lab):
    rl = lab / "lab" / "rl"
    other = read_env(rl / "other.py", "other")
    assert other["extras"] == ["mjx"]
    assert other["vars"] == {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "A": "base"}
    mine = read_env(rl / "policy.py", "policy")
    assert mine["extras"] == ["mjx", "sb3"]                      # the full set
    assert mine["vars"] == {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "A": "mine", "N": "3"}
    assert read_env(lab, None)["extras"] == []                   # no toml above: nothing


def test_a_file_named_like_a_key_and_bad_values_are_refused(lab):
    rl = lab / "lab" / "rl"
    with pytest.raises(ValueError, match="an \\[env\\] key"):
        read_env(rl / "vars.py", "vars")
    (rl / "experiment.toml").write_text('[env]\nextras = "mjx"\n')
    with pytest.raises(ValueError, match="list of names"):
        read_env(rl / "policy.py", "policy")


def test_unknown_keys_warn(lab, capsys):
    (lab / "lab" / "rl" / "experiment.toml").write_text('[env]\nextra = ["mjx"]\n')
    read_env(lab / "lab" / "rl" / "policy.py", "policy")
    assert "unknownkey(s)['extra']" in "".join(capsys.readouterr().err.split())  # may wrap


def test_the_file_is_found_without_importing(lab, monkeypatch):
    monkeypatch.chdir(lab)
    assert launch.experiment_file("lab.rl.policy") == (lab / "lab" / "rl" / "policy.py").resolve()
    assert launch.experiment_file("lab.rl") == (lab / "lab" / "rl" / "__init__.py").resolve()
    assert launch.experiment_file("lab/rl/policy.py").name == "policy.py"
    assert launch.experiment_file("no.such.module") is None


def test_extras_relaunch_under_uv_once(lab, monkeypatch):
    monkeypatch.chdir(lab)
    monkeypatch.setattr(launch.shutil, "which", lambda name: "/usr/bin/uv")
    calls, environ = [], {"PATH": "/usr/bin"}
    argv = ["run", "lab.rl.policy", "steps=10", "--tag=x"]
    launch.prepare("lab.rl.policy", argv, environ=environ,
                   execvpe=lambda f, cmd, env: calls.append((f, cmd, env)))
    (f, cmd, env), = calls
    assert cmd == ["/usr/bin/uv", "run", "--project", str(lab.resolve()),
                   "--extra", "mjx", "--extra", "sb3", "python", "-m", "runkit", *argv]
    assert env["A"] == "mine" and env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env[launch.GUARD] == "1" and env["PATH"] == "/usr/bin"
    assert json.loads(env[launch.LAUNCH]) == {
        "extras": ["mjx", "sb3"], "project": str(lab.resolve()),
        "vars": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "A": "mine", "N": "3"}}

    calls.clear()                                                # the relaunched process
    launch.prepare("lab.rl.policy", argv, environ=env, execvpe=lambda *a: calls.append(a))
    assert calls == []


def test_vars_alone_are_set_in_process(lab, monkeypatch):
    (lab / "lab" / "rl" / "experiment.toml").write_text('[env]\nvars = { A = "1" }\n')
    environ, calls = {}, []
    launch.prepare(str(lab / "lab" / "rl" / "policy.py"), [], environ=environ,
                   execvpe=lambda *a: calls.append(a))
    assert calls == [] and environ["A"] == "1"
    assert json.loads(environ[launch.LAUNCH])["vars"] == {"A": "1"}


def test_extras_without_uv_or_a_project_explain_themselves(lab, monkeypatch):
    monkeypatch.chdir(lab)
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="uv is not on PATH"):
        launch.prepare("lab.rl.policy", [], environ={}, execvpe=lambda *a: None)
    monkeypatch.setattr(launch.shutil, "which", lambda name: "/usr/bin/uv")
    (lab / "pyproject.toml").unlink()
    with pytest.raises(SystemExit, match="no pyproject.toml"):
        launch.prepare("lab.rl.policy", [], environ={}, execvpe=lambda *a: None)


@dataclass
class Cfg:
    n: int = 1


def test_meta_records_what_the_process_was_launched_with(tmp_path, monkeypatch):
    exp = Experiment("meta")

    @exp.run
    def run(cfg: Cfg, ctx: RunContext):
        pass

    r = exp.main([f"--root={tmp_path}"])
    assert yaml.safe_load((r.context.dir / "meta.yaml").read_text())["launch"] is None
    launched = {"extras": ["mjx"], "vars": {"A": "1"}, "project": "/p"}
    monkeypatch.setenv(launch.LAUNCH, json.dumps(launched))
    r = exp.main([f"--root={tmp_path}"])
    assert yaml.safe_load((r.context.dir / "meta.yaml").read_text())["launch"] == launched


def test_a_dotted_module_is_found_from_another_folder(lab, monkeypatch):
    """Run from the experiment's own folder (`cd lab/rl`), a dotted target is
    not under the cwd: python's import path finds it, so experiment.toml is not
    skipped silently."""
    monkeypatch.syspath_prepend(str(lab))                  # as an installed package would be
    monkeypatch.chdir(lab / "lab" / "rl")
    assert launch.experiment_file("lab.rl.policy") == (lab / "lab" / "rl" / "policy.py").resolve()
    assert launch.experiment_file("no.such.module") is None
    calls = []
    monkeypatch.setattr(launch.shutil, "which", lambda name: "/usr/bin/uv")
    launch.prepare("lab.rl.policy", ["run", "lab.rl.policy"], environ={},
                   execvpe=lambda f, cmd, env: calls.append(cmd))
    assert calls and "--extra" in calls[0]                  # relaunched with its extras
