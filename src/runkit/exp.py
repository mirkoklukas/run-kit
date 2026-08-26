"""The lightweight `@experiment` decorator and the run dir it builds.

A decorated function is `run(cfg, ctx)`. The decorator wraps it so that, before
the body runs, a fresh run dir is created and the run frozen into it; after, a
non-`None` return value is dumped into the run dir and the caller gets a `Run`. `config.yaml`,
`run_context.yaml` and `meta.yaml` are what it takes to recreate the run;
`status.yaml` (and `traceback.txt`) are how that run went. Staging is driven by
keyword args (the CLI flags): `tag`, `runs_dir`, `out`, `force`.

The decorator carries only the experiment's *identity* (`name`); everything that
stages an attempt is a flag. See design.md.
"""
import dataclasses
import datetime
import functools
import pathlib
import shutil
import time
import traceback
import uuid

import yaml

from . import ui
from .utils import dump_retval, serialize_cfg


@dataclasses.dataclass
class RunContext:
    out: pathlib.Path        # the run dir; everything the experiment writes goes here
    id: str                  # stable unique run id, e.g. "baseline_a3f9c1e7" (for search)
    name: str | None = None  # the experiment's identity, as given to @experiment


@dataclasses.dataclass
class Run:
    """What a *caller* gets back, as `RunContext` is what the body is handed.

    `retval` is kept in memory rather than left to `results/retval.json`: that
    dump is best-effort, so a value that will not serialize lives only here.
    There is deliberately no `status` field -- a failed run raises, so a caller
    holding a `Run` always has one that finished.
    """
    context: RunContext
    retval: object = None


def _dump_yaml(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _stamp(when=None):
    """A second-resolution ISO timestamp -- what goes in `status.yaml`."""
    return (when or datetime.datetime.now()).isoformat(timespec="seconds")


def _write_status(out, *, status, started, ended=None, duration_s=None, error=None):
    """Write `{out}/status.yaml`. Best-effort, and for a sharper reason than
    `dump_retval`: the final write happens inside a `finally` with the
    experiment's exception in flight, so a failure here must never replace it.
    """
    try:
        _dump_yaml(pathlib.Path(out) / "status.yaml", {
            "status": status, "started": started, "ended": ended,
            "duration_s": duration_s, "error": error,
        })
    except Exception as e:                                   # noqa: BLE001
        ui.warn(f"could not write status.yaml: {e}")


def _write_traceback(out):
    """Dump the in-flight traceback next to the run. Best-effort, as above."""
    try:
        (pathlib.Path(out) / "traceback.txt").write_text(traceback.format_exc())
    except Exception as e:                                   # noqa: BLE001
        ui.warn(f"could not write traceback.txt: {e}")


def _resolve_out(runs_dir, name, tag, hex8, out_override, force):
    """Build the run dir path -- the one place the naming scheme lives.

    default: {runs_dir}/{date}_{time}_{name}[_{tag}]_{hex8}/
             (date=YYYY-MM-DD, time=HH-MM)
    --out:   that exact dir. On collision, append _{timestamp} -- unless `force`,
             which keeps the exact path (the existing dir is replaced in init_run).
    """
    if out_override is not None:
        p = pathlib.Path(out_override).resolve()
        if p.exists() and not force:
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            p = p.parent / f"{p.name}_{ts}"
        return p
    now = datetime.datetime.now()
    parts = [f"{now:%Y-%m-%d}", f"{now:%H-%M}", name, *([tag] if tag else []), hex8]
    return pathlib.Path(runs_dir).resolve() / "_".join(parts)


def init_run(cfg, *, name, tag, runs_dir, out, force=False, script=None):
    """Create the run dir, freeze the run into it, return a RunContext.

    Writes the three files it takes to recreate the run -- `config.yaml` (the
    resolved cfg), `run_context.yaml` (the RunContext) and `meta.yaml` (how the
    attempt was staged). `status.yaml` is not written here: nothing is running
    yet, and the lifecycle belongs to whoever calls the body.

    The run id and the dir's `{hex8}` share one uuid, so the dir is
    self-identifying (`id = {name}_{hex8}`). `force` only bites with an explicit
    `--out` whose dir already exists: that dir is removed and rebuilt fresh.
    """
    if force and out is None:
        ui.warn("--force has no effect without --out (default run dirs never collide)")
    hex8 = uuid.uuid4().hex[:8]
    uid = f"{name}_{hex8}"
    out_path = _resolve_out(runs_dir, name, tag, hex8, out, force)
    if force and out_path.exists():
        shutil.rmtree(out_path)
        ui.warn(f"--force: replaced existing run dir {out_path}")
    out_path.mkdir(parents=True, exist_ok=False)
    (out_path / "results").mkdir()
    _dump_yaml(out_path / "config.yaml", serialize_cfg(cfg))
    _dump_yaml(out_path / "run_context.yaml", {"id": uid, "name": name})
    _dump_yaml(out_path / "meta.yaml",
               {"tag": tag, "script": str(script) if script else None})
    return RunContext(out=out_path, id=uid, name=name)


def _announce(name, cfg, ctx):
    """Print a one-time start banner: which run, with what config, where.

    The resolved config is echoed inline; the same values are frozen at
    `{out}/config.yaml`.
    """
    ui.run_started(
        name=name,
        run_id=ctx.id,
        out_dir=ctx.out,
        config_path=ctx.out / "config.yaml",
        cfg=serialize_cfg(cfg),
    )


def experiment(*, name):
    """Mark `run(cfg, ctx)` as an experiment entry point.

    `name` is the experiment's identity (required, no CLI override). The wrapper
    accepts the staging flags as keyword args -- `tag`, `runs_dir`, `out`, `force`
    -- which `autocli.main` forwards from the CLI; their names *are* the allowed
    flags.
    """
    def decorator(f):
        script = pathlib.Path(f.__code__.co_filename).resolve()

        @functools.wraps(f)
        def wrapper(cfg, *, tag=None, runs_dir="runs", out=None, force=False):
            ctx = init_run(cfg, name=name, tag=tag, runs_dir=runs_dir, out=out,
                           force=force, script=script)
            _announce(name, cfg, ctx)
            started, t0 = datetime.datetime.now(), time.monotonic()
            _write_status(ctx.out, status="running", started=_stamp(started))
            status, error = "ok", None
            try:
                result = f(cfg, ctx)
            except KeyboardInterrupt:
                status = "interrupted"
                raise
            except BaseException as e:                       # noqa: BLE001
                status, error = "failed", f"{type(e).__name__}: {e}"
                _write_traceback(ctx.out)
                raise
            finally:
                # every branch re-raises: runkit records the outcome, it does
                # not handle it. A failed run still exits non-zero.
                _write_status(ctx.out, status=status, started=_stamp(started),
                              ended=_stamp(), duration_s=round(time.monotonic() - t0, 3),
                              error=error)
            if result is not None:
                dump_retval(ctx.out / "results", result)
            return Run(context=ctx, retval=result)
        wrapper._runkit_name = name          # identity, for introspection
        # where the experiment lives, for resolving `exp:`-prefixed config paths
        wrapper._runkit_dir = script.parent
        return wrapper
    return decorator
