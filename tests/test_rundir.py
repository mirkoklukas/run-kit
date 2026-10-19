"""`@rundir`: a run dir per call of a plain function, and `runkit.toml`."""
import inspect
import pathlib

import pytest
import yaml

from runkit import RunContext, rundir
from runkit.settings import resolve_runs_root


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _load(p):
    return yaml.safe_load(pathlib.Path(p).read_text())


def test_fresh_run_dir_under_cwd_runs(cwd):
    seen = {}

    @rundir("homing")
    def main(leg: int, speed: float = 1.0, *, run_dir: pathlib.Path):
        seen["dir"] = run_dir
        (run_dir / "state.txt").write_text(str(leg))
        return {"leg": leg}

    assert main(0) == {"leg": 0}                     # the body's own return value
    d = seen["dir"]
    assert d.parent == cwd / "runs" / "homing"
    assert (d / "state.txt").read_text() == "0"
    assert not (d / "out").exists()                  # run_dir form: no out/
    assert (cwd / "runs" / "homing" / "latest").resolve() == d
    meta = _load(d / "meta.yaml")
    assert meta["kind"] == "rundir"
    assert meta["args"] == {"leg": 0, "speed": 1.0}
    assert meta["cwd"] == str(cwd)
    status = _load(d / "status.yaml")
    assert status["status"] == "ok"
    ctx = _load(d / "run_context.yaml")
    assert ctx["name"] == "homing" and ctx["id"].startswith("homing_")
    assert (d / "retval.json").is_file()
    assert not (d / "config.yaml").exists()          # no Config: nothing to freeze


def test_ctx_form_removes_empty_out(cwd):
    seen = {}

    @rundir("calib")
    def main(*, ctx: RunContext):
        seen["ctx"] = ctx
        assert ctx.live and ctx.out.is_dir()
        ctx.record(err=0.5)
        ctx.progress(1, total=2)

    assert main() is None
    ctx = seen["ctx"]
    assert not ctx.out.exists()                      # nothing written: removed
    assert not ctx.live
    assert (ctx.dir / "metrics" / "run.jsonl").is_file()
    assert _load(ctx.dir / "status.yaml")["progress"] == 1
    assert not (ctx.dir / "retval.json").exists()


def test_ctx_form_keeps_used_out(cwd):
    @rundir("calib")
    def main(*, ctx: RunContext):
        (ctx.out / "x.txt").write_text("x")
        return ctx.dir

    d = main()
    assert (d / "out" / "x.txt").read_text() == "x"


def test_both_run_dir_and_ctx(cwd):
    @rundir("both")
    def main(*, run_dir, ctx):
        assert run_dir == ctx.dir
        return 1

    assert main() == 1


def test_explicit_run_dir(cwd, tmp_path):
    target = tmp_path / "here" / "run"

    @rundir("homing")
    def main(*, run_dir):
        return run_dir

    assert main(run_dir=target) == target.resolve()
    assert _load(target / "status.yaml")["status"] == "ok"
    assert not (cwd / "runs").exists()               # no fresh dir, no latest
    with pytest.raises(FileExistsError):             # not empty any more
        main(run_dir=target)


def test_failure_is_recorded_and_raised(cwd):
    seen = {}

    @rundir("boom")
    def main(*, ctx):
        seen["dir"] = ctx.dir
        raise ValueError("bad leg")

    with pytest.raises(ValueError, match="bad leg"):
        main()
    d = seen["dir"]
    status = _load(d / "status.yaml")
    assert status["status"] == "failed" and status["error"] == "ValueError: bad leg"
    assert "bad leg" in (d / "traceback.txt").read_text()
    assert not (d / "out").exists()


def test_public_signature():
    @rundir("sig")
    def main(leg: int, *, verbose: bool = False, ctx: RunContext, **extra):
        ...

    params = inspect.signature(main).parameters
    assert list(params) == ["leg", "verbose", "run_dir", "extra"]
    assert params["run_dir"].default is None
    assert params["run_dir"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "ctx" not in main.__annotations__
    assert main.__annotations__["run_dir"] == (pathlib.Path | None)


def test_bad_call_makes_no_run_dir(cwd):
    @rundir("homing")
    def main(leg: int, *, run_dir):
        ...

    with pytest.raises(TypeError):
        main()                                       # leg missing
    with pytest.raises(TypeError, match="runkit's"):
        main(0, ctx=object())
    assert not (cwd / "runs").exists()


@pytest.mark.parametrize("body, match", [
    (lambda leg: None, "neither"),
    (lambda run_dir: None, "keyword-only"),
    (lambda *, run_dir=None: None, "no default"),
])
def test_bad_bodies(body, match):
    with pytest.raises(TypeError, match=match):
        rundir("x")(body)


def test_needs_a_name():
    with pytest.raises(TypeError, match="needs a name"):
        @rundir
        def main(*, run_dir): ...


# ── runkit.toml ───────────────────────────────────────────────────────────────

def test_runs_root_default(tmp_path):
    assert resolve_runs_root(tmp_path) == tmp_path.resolve() / "runs"


def test_runs_root_from_toml(tmp_path):
    (tmp_path / "runkit.toml").write_text('runs = "out/runs"\n')
    assert resolve_runs_root(tmp_path, hosts=["laptop"]) == tmp_path.resolve() / "out/runs"


def test_runs_root_per_host(tmp_path):
    (tmp_path / "runkit.toml").write_text(
        'runs = "runs"\n'
        '[host.jetson]\nruns = "/data/runs"\n'
        '[host."jetson.local"]\nruns = "/data/full"\n')
    assert resolve_runs_root(tmp_path, hosts=["jetson"]) == pathlib.Path("/data/runs")
    assert resolve_runs_root(tmp_path, hosts=["jetson.local", "jetson"]) == \
        pathlib.Path("/data/full")                   # the full hostname wins
    assert resolve_runs_root(tmp_path, hosts=["laptop"]) == tmp_path.resolve() / "runs"


def test_runs_root_only_in_cwd(tmp_path):
    (tmp_path / "runkit.toml").write_text('runs = "elsewhere"\n')
    sub = tmp_path / "sub"
    sub.mkdir()
    assert resolve_runs_root(sub) == sub.resolve() / "runs"   # not looked for above


def test_runs_root_used_by_rundir(cwd):
    (cwd / "runkit.toml").write_text('runs = "my-runs"\n')

    @rundir("homing")
    def main(*, run_dir):
        return run_dir

    assert main().parent == cwd / "my-runs" / "homing"
