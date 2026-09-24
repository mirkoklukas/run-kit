"""The lightweight `@experiment` decorator and the run dir it builds.

A decorated function is `run(cfg, ctx)`. The decorator wraps it so that, before
the body runs, a fresh run dir is created and the run frozen into it; after, a
non-`None` return value is dumped into the run dir and the caller gets a `Run`. `config.yaml`,
`run_context.yaml` and `meta.yaml` are what the run was; `status.yaml` (and
`traceback.txt`) are how that run went. runkit owns the top level of the run
dir; the body owns `out/` (`ctx.out`). Staging is driven by keyword args (the
CLI flags): `tag`, `root`.

The decorator carries only the experiment's *identity* (`name`); everything that
stages an attempt is a flag. See design.md.
"""
import dataclasses
import datetime
import functools
import os
import pathlib
import time
import traceback
import uuid

import yaml

from . import ui
from .utils import dump_retval, serialize_cfg


@dataclasses.dataclass
class RunContext:
    dir: pathlib.Path        # the run dir; its top level is runkit's records
    id: str                  # stable unique run id, e.g. "baseline_a3f9c1e7" (for search)
    name: str | None = None  # the experiment's identity, as given to @experiment

    @property
    def out(self) -> pathlib.Path:
        """`{dir}/out` -- everything the experiment writes goes here."""
        return self.dir / "out"


@dataclasses.dataclass
class Run:
    """What a *caller* gets back, as `RunContext` is what the body is handed.

    The in-memory image of a run dir: one field per file it wrote --
    `config.yaml`, `run_context.yaml`, `retval.json`.

    `config` is carried because the caller does not always have it: from the CLI
    it is `autocli` that builds it, and comparing a sweep means pairing each
    config with its result. `retval` is kept in memory rather than left to
    `retval.json`, since that dump is best-effort and a value that will
    not serialize lives only here. There is deliberately no `status` field -- a
    failed run raises, so a caller holding a `Run` always has one that finished.
    """
    config: object
    context: RunContext
    retval: object = None


def _dump_yaml(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _stamp(when=None):
    """A second-resolution ISO timestamp -- what goes in `status.yaml`."""
    return (when or datetime.datetime.now()).isoformat(timespec="seconds")


def _write_status(run_dir, *, status, started, ended=None, duration_s=None, error=None):
    """Write `{run_dir}/status.yaml`. Best-effort, and for a sharper reason than
    `dump_retval`: the final write happens inside a `finally` with the
    experiment's exception in flight, so a failure here must never replace it.
    """
    try:
        _dump_yaml(pathlib.Path(run_dir) / "status.yaml", {
            "status": status, "started": started, "ended": ended,
            "duration_s": duration_s, "error": error,
        })
    except Exception as e:                                   # noqa: BLE001
        ui.warn(f"could not write status.yaml: {e}")


def _write_traceback(run_dir):
    """Dump the in-flight traceback next to the run. Best-effort, as above."""
    try:
        (pathlib.Path(run_dir) / "traceback.txt").write_text(traceback.format_exc())
    except Exception as e:                                   # noqa: BLE001
        ui.warn(f"could not write traceback.txt: {e}")


def _resolve_dir(root, name, tag, hex8):
    """Build the run dir path -- the one place the naming scheme lives.

    {root}/{name}/{date}_{time}[_{tag}]_{hex8}/   (date=YYYY-MM-DD, time=HH-MM-SS)
    """
    now = datetime.datetime.now()
    parts = [f"{now:%Y-%m-%d}", f"{now:%H-%M-%S}", *([tag] if tag else []), hex8]
    return pathlib.Path(root).resolve() / name / "_".join(parts)


def _point_latest(run_dir):
    """Point `{root}/{name}/latest` at `run_dir` -- the latest *started* run.

    A shortcut for people (`cd runs/baseline/latest`), not a record: nothing in
    runkit reads it. So it is best-effort -- a failure (no symlink rights, a real
    dir in the way) warns and moves on. The target is relative, so the link
    survives moving the root, and it is swapped in with `os.replace`, so
    concurrent starts never leave it half-written.
    """
    link = run_dir.parent / "latest"
    tmp = run_dir.parent / f".latest.{run_dir.name}"
    try:
        tmp.symlink_to(run_dir.name, target_is_directory=True)
        os.replace(tmp, link)
    except Exception as e:                                   # noqa: BLE001
        tmp.unlink(missing_ok=True)
        ui.warn(f"could not point {link} at this run: {e}")


def init_run(cfg, *, name, tag, root, script=None):
    """Create the run dir, freeze the run into it, return a RunContext.

    Writes the three files that say what the run was -- `config.yaml` (the
    resolved cfg), `run_context.yaml` (the RunContext) and `meta.yaml` (how the
    attempt was staged) -- and creates the empty `out/` the body writes into.
    `status.yaml` is not written here: nothing is running
    yet, and the lifecycle belongs to whoever calls the body.

    The run id and the dir's `{hex8}` share one uuid, so the dir is
    self-identifying (`id = {name}_{hex8}`) and never collides. It also becomes
    `{root}/{name}/latest`.
    """
    hex8 = uuid.uuid4().hex[:8]
    uid = f"{name}_{hex8}"
    run_dir = _resolve_dir(root, name, tag, hex8)
    run_dir.mkdir(parents=True, exist_ok=False)
    ctx = RunContext(dir=run_dir, id=uid, name=name)
    ctx.out.mkdir()
    _dump_yaml(run_dir / "config.yaml", serialize_cfg(cfg))
    _dump_yaml(run_dir / "run_context.yaml", {"id": uid, "name": name})
    _dump_yaml(run_dir / "meta.yaml",
               {"tag": tag, "script": str(script) if script else None})
    _point_latest(run_dir)
    return ctx


def _announce(name, cfg, ctx):
    """Print a one-time start banner: which run, with what config, where.

    The resolved config is echoed inline; the same values are frozen at
    `{dir}/config.yaml`.
    """
    ui.run_started(
        name=name,
        run_id=ctx.id,
        run_dir=ctx.dir,
        config_path=ctx.dir / "config.yaml",
        cfg=serialize_cfg(cfg),
    )


def experiment(*, name):
    """Mark `run(cfg, ctx)` as an experiment entry point.

    `name` is the experiment's identity (required, no CLI override). The wrapper
    accepts the staging flags as keyword args -- `tag`, `root` -- which `autocli.main` forwards from the CLI; their names *are* the allowed
    flags.
    """
    def decorator(f):
        script = pathlib.Path(f.__code__.co_filename).resolve()

        @functools.wraps(f)
        def wrapper(cfg, *, tag=None, root="runs"):
            ctx = init_run(cfg, name=name, tag=tag, root=root, script=script)
            _announce(name, cfg, ctx)
            started, t0 = datetime.datetime.now(), time.monotonic()
            _write_status(ctx.dir, status="running", started=_stamp(started))
            status, error = "ok", None
            try:
                result = f(cfg, ctx)
            except KeyboardInterrupt:
                status = "interrupted"
                raise
            except BaseException as e:                       # noqa: BLE001
                status, error = "failed", f"{type(e).__name__}: {e}"
                _write_traceback(ctx.dir)
                raise
            finally:
                # every branch re-raises: runkit records the outcome, it does
                # not handle it. A failed run still exits non-zero.
                _write_status(ctx.dir, status=status, started=_stamp(started),
                              ended=_stamp(), duration_s=round(time.monotonic() - t0, 3),
                              error=error)
            if result is not None:
                dump_retval(ctx.dir, result)
            return Run(config=cfg, context=ctx, retval=result)
        wrapper._runkit_name = name          # identity, for introspection
        # where the experiment lives, for resolving `exp:`-prefixed config paths
        wrapper._runkit_dir = script.parent
        return wrapper
    return decorator
