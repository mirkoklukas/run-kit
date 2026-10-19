"""`@rundir`: a run dir for a plain function -- a script that is not an experiment.

    from runkit import rundir

    @rundir("homing")
    def main(leg: int, *, run_dir: Path):        # or `*, ctx: RunContext`, or both
        (run_dir / "home-state.yaml").write_text(...)

    if __name__ == "__main__":
        typer.run(main)                          # argv stays the script's own

Every call gets a fresh run dir, `{root}/{name}/{date}_{time}_{hex8}/` with
`latest` beside it, as an experiment's runs do. The root is `./runs` in the
folder the script is started from, unless a `runkit.toml` there says otherwise
(`settings.resolve_runs_root`). Unlike `@exp.run`, the decorator does not touch
argv and knows no `Config`: the function keeps its own arguments, and they are
recorded in `meta.yaml` as `args`.

The body chooses what it is handed by parameter name, keyword-only and without
a default: `run_dir` (the run dir itself) and/or `ctx` (a live `RunContext`:
`ctx.dir` is the run dir, and `ctx.record`, `ctx.progress`, `ctx.checkpoint`
work as in an experiment). The decorated function shows `run_dir: Path | None =
None` instead -- so a CLI built from its signature (typer) gets an optional
`--run-dir` -- and `ctx` not at all. `None` means a fresh run dir; a path
means that folder, which must be new or empty.

With `ctx`, `ctx.out` exists while the body runs and is removed at the end if
nothing was written into it. The decorated function returns what the body
returns (and dumps a non-`None` value to `retval.json`); a raised exception is
recorded and propagates, as for an experiment.
"""
import datetime
import functools
import inspect
import pathlib
import sys
import time
import uuid

from . import ui
from .exp import (RunContext, _Live, _dump_yaml, _launch, _module_name, _report,
                  _resolve_dir, _stamp, _write_status, _write_traceback)
from .settings import resolve_runs_root
from .utils import dump_retval, point_latest

INJECTED = ("run_dir", "ctx")     # what a body may ask for, by parameter name


def rundir(name):
    """Give every call of the decorated function a fresh run dir; see the module
    docstring. `name` names the folder its runs go in: `{root}/{name}/`."""
    if callable(name):
        raise TypeError('@rundir needs a name: @rundir("homing")')
    if not isinstance(name, str) or not name:
        raise TypeError(f"@rundir: the name must be a non-empty string, got {name!r}")
    return lambda f: _wrap(f, name)


def _wrap(f, name):
    sig = inspect.signature(f)
    params = sig.parameters
    wants = [p for p in INJECTED if p in params]
    if not wants:
        raise TypeError(f"@rundir: {f.__name__} takes neither `run_dir` nor `ctx`; "
                        f"declare one, keyword-only: def {f.__name__}(..., *, run_dir: Path)")
    for p in wants:
        if params[p].kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(f"@rundir: `{p}` must be keyword-only: "
                            f"def {f.__name__}(..., *, {p}: ...)")
        if params[p].default is not inspect.Parameter.empty:
            raise TypeError(f"@rundir: `{p}` gets no default -- runkit fills it in")
    public = _public_signature(sig)
    script = pathlib.Path(f.__code__.co_filename).resolve()
    module = _module_name(f)

    @functools.wraps(f)
    def wrapper(*args, run_dir=None, **kwargs):
        if "ctx" in kwargs:
            raise TypeError(f"{f.__name__}: `ctx` is runkit's to hand in; "
                            f"pass run_dir= to choose the folder")
        call = _call_args(public, args, {**kwargs, "run_dir": run_dir})
        ctx = _create(name, run_dir, script=script, module=module, call=call)
        ui.rundir_started(name=name, run_id=ctx.id, run_dir=ctx.dir)
        started, t0 = datetime.datetime.now(), time.monotonic()
        ctx._live = _Live(t0=t0, started=_stamp(started))
        made_out = "ctx" in params
        if made_out:
            ctx.out.mkdir()
        inject = {"run_dir": ctx.dir, "ctx": ctx}
        inject = {k: inject[k] for k in wants}
        _write_status(ctx.dir, status="running", started=_stamp(started))
        status, error = "ok", None
        try:
            result = f(*args, **kwargs, **inject)
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except BaseException as e:                       # noqa: BLE001
            status, error = "failed", f"{type(e).__name__}: {e}"
            _write_traceback(ctx.dir)
            raise
        finally:
            # as in an experiment: the outcome is recorded, the exception re-raised
            duration_s = round(time.monotonic() - t0, 3)
            live, ctx._live = ctx._live, None
            _write_status(ctx.dir, status=status, started=_stamp(started),
                          progress=live.progress, total=live.total,
                          checkpoint=live.checkpoint,
                          ended=_stamp(), duration_s=duration_s, error=error)
            if made_out:
                _remove_if_empty(ctx.out)
            if status != "ok":
                _report(ctx, status, duration_s, error)
        if result is not None:
            dump_retval(ctx.dir, result)
        _report(ctx, status, duration_s, error)
        return result

    # what callers -- and CLIs built from the signature -- see: an optional
    # `run_dir`, and no `ctx`
    wrapper.__signature__ = public
    wrapper.__annotations__ = {k: v for k, v in f.__annotations__.items()
                               if k not in INJECTED}
    wrapper.__annotations__["run_dir"] = pathlib.Path | None
    wrapper._runkit_name = name
    wrapper._runkit_kind = "rundir"
    return wrapper


def _public_signature(sig):
    """`sig` without `run_dir` / `ctx`, plus a keyword-only `run_dir: Path |
    None = None` (before a `**kwargs`, if there is one)."""
    kept = [p for p in sig.parameters.values() if p.name not in INJECTED]
    run_dir = inspect.Parameter("run_dir", inspect.Parameter.KEYWORD_ONLY,
                                default=None, annotation=pathlib.Path | None)
    at = next((i for i, p in enumerate(kept)
               if p.kind is inspect.Parameter.VAR_KEYWORD), len(kept))
    return sig.replace(parameters=[*kept[:at], run_dir, *kept[at:]])


def _call_args(public, args, kwargs):
    """The call's arguments by name, defaults filled in, for `meta.yaml`'s
    `args` (without `run_dir`). A call that does not fit the signature raises
    here, before a run dir is made."""
    bound = public.bind(*args, **kwargs)
    bound.apply_defaults()
    return {k: _plain(v) for k, v in bound.arguments.items() if k != "run_dir"}


def _plain(v):
    """`v` as something yaml can hold: itself when it is plain data, its
    `str` for a path, its `repr` otherwise."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, pathlib.PurePath):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return repr(v)


def _create(name, explicit, *, script, module, call):
    """Make the run dir and write what the call was: `run_context.yaml` and
    `meta.yaml`. A fresh one under the runs root becomes `latest`; an explicit
    folder must be new or empty, and gets no `latest`."""
    hex8 = uuid.uuid4().hex[:8]
    if explicit is None:
        run_dir = _resolve_dir(resolve_runs_root(), name, None, hex8)
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = pathlib.Path(explicit).expanduser().resolve()
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise FileExistsError(f"{run_dir}: already exists and is not empty -- "
                                  f"a run dir starts empty")
        run_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(dir=run_dir, id=f"{name}_{hex8}", name=name)
    _dump_yaml(run_dir / "run_context.yaml", {"id": ctx.id, "name": name})
    _dump_yaml(run_dir / "meta.yaml",
               {"kind": "rundir", "script": str(script), "module": module,
                "argv": list(sys.argv), "cwd": str(pathlib.Path.cwd()),
                "args": call, "launch": _launch()})
    if explicit is None:
        point_latest(run_dir)
    return ctx


def _remove_if_empty(folder):
    """Remove `folder` if nothing was written into it. Best-effort: it runs at
    the end of the run, possibly with the body's exception in flight."""
    try:
        folder.rmdir()
    except OSError:
        pass
