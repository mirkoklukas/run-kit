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
argv and knows no `Config`: the function keeps its own arguments.

The body chooses what it is handed by parameter name, keyword-only and without
a default, and that decides what runkit writes:

- `run_dir` -- a fresh folder, and nothing in it: runkit writes no files.
- `ctx` -- a live `RunContext`, and a recorded run: `config.yaml` (the call's
  arguments by name), `context.yaml`, `meta.yaml`, `status.yaml`, and
  `traceback.txt` / `retval.json` when they apply. `ctx.record`,
  `ctx.progress` and `ctx.checkpoint` work as in an experiment; `ctx.out`
  exists while the body runs and is removed at the end if nothing was written
  into it.

Both may be taken; `run_dir` is then `ctx.dir`. The decorated function shows
`run_dir: Path | None = None` instead -- so a CLI built from its signature
(typer) gets an optional `--run-dir` -- and `ctx` not at all. `None` means a
fresh run dir; a path means that folder, which must be new or empty. The
decorated function returns what the body returns; an exception propagates
(recorded, with `ctx`).

Where a fresh run dir goes: a body may also declare `run_dir_root`,
keyword-only, with or without a default. The default is a path, or a template
filled in with the call's arguments (as in `config.yaml`) and the environment
(`{env[NAME]}`), e.g.

    def main(runtime: Path, *, ctx, run_dir_root: str = "{runtime}/runs"):

The decorated function then shows `run_dir_root: str | None` with that default
(a CLI gets an optional `--run-dir-root`, its help showing the template), and
the body gets the root used. The root is the call's `run_dir_root` if given
(a template too), else the body's default, else the usual one (`./runs` or
`runkit.toml`; also when an argument in the template is None). An explicit
`run_dir` has no root: the body gets None.
"""
import datetime
import functools
import inspect
import os
import pathlib
import string
import sys
import time
import uuid

from . import ui
from .exp import (RunContext, _Live, _dump_yaml, _launch, _module_name, _resolve_dir,
                  _stamp, _write_status, _write_traceback)
from .settings import resolve_runs_root
from .utils import CONTEXT, dump_retval, point_latest

INJECTED = ("run_dir", "ctx")     # what a body may ask for, by parameter name
ROOT = "run_dir_root"             # the runs root: a body may declare it, with or without a default
ENV = "env"                       # in a run_dir_root template: {env[NAME]}, an environment variable


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
    has_root = ROOT in params
    if has_root and params[ROOT].kind is not inspect.Parameter.KEYWORD_ONLY:
        raise TypeError(f"@rundir: `{ROOT}` must be keyword-only: "
                        f"def {f.__name__}(..., *, {ROOT}: ...)")
    # the fallback root: a template (or plain path), or None (runs_root's rule)
    root_default = params[ROOT].default if has_root else None
    if root_default is inspect.Parameter.empty:
        root_default = None
    if root_default is not None:
        if not isinstance(root_default, (str, os.PathLike)):
            raise TypeError(f"@rundir: `{ROOT}`'s default must be a path or a template string, "
                            f"got {root_default!r}")
        root_default = str(root_default)
        _check_template(root_default, params, f.__name__)
    public = _public_signature(sig, root_default if has_root else _NO_ROOT)
    script = pathlib.Path(f.__code__.co_filename).resolve()
    module = _module_name(f)
    full = "ctx" in params           # a run, recorded; else just a fresh folder

    @functools.wraps(f)
    def wrapper(*args, run_dir=None, **kwargs):
        if "ctx" in kwargs:
            raise TypeError(f"{f.__name__}: `ctx` is runkit's to hand in; "
                            f"pass run_dir= to choose the folder")
        root_given = kwargs.pop(ROOT, root_default) if has_root else None
        call = _call_args(public, args, {**kwargs, "run_dir": run_dir,
                                         **({ROOT: root_given} if has_root else {})})
        # the root of a fresh run dir; an explicit run_dir has none
        root = None if run_dir is not None else _root(root_given, call)
        ctx = _create(name, run_dir, root=root, script=script, module=module, call=call, full=full)
        ui.rundir_started(name=name, run_id=ctx.id, run_dir=ctx.dir)
        started, t0 = datetime.datetime.now(), time.monotonic()
        ctx._live = _Live(t0=t0, started=_stamp(started))
        inject = {"run_dir": ctx.dir, "ctx": ctx}
        inject = {k: inject[k] for k in wants}
        if has_root:
            inject[ROOT] = root
        if full:
            ctx.out.mkdir()
            _write_status(ctx.dir, status="running", started=_stamp(started))
        status, error = "ok", None
        try:
            result = f(*args, **kwargs, **inject)
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except BaseException as e:                       # noqa: BLE001
            status, error = "failed", f"{type(e).__name__}: {e}"
            if full:
                _write_traceback(ctx.dir)
            raise
        finally:
            # as in an experiment: the outcome is recorded, the exception re-raised
            duration_s = round(time.monotonic() - t0, 3)
            live, ctx._live = ctx._live, None
            if full:
                _write_status(ctx.dir, status=status, started=_stamp(started),
                              progress=live.progress, total=live.total,
                              checkpoint=live.checkpoint,
                              ended=_stamp(), duration_s=duration_s, error=error)
                _remove_if_empty(ctx.out)
            if status != "ok":
                _report(ctx, status, duration_s, error, full)
        if full and result is not None:
            dump_retval(ctx.dir, result)
        _report(ctx, status, duration_s, error, full)
        return result

    # what callers -- and CLIs built from the signature -- see: an optional
    # `run_dir`, and no `ctx`
    wrapper.__signature__ = public
    wrapper.__annotations__ = {k: v for k, v in f.__annotations__.items()
                               if k not in (*INJECTED, ROOT)}
    wrapper.__annotations__["run_dir"] = pathlib.Path | None
    if has_root:
        wrapper.__annotations__[ROOT] = str | None
    wrapper._runkit_name = name
    wrapper._runkit_kind = "rundir"
    return wrapper


_NO_ROOT = object()               # the body doesn't declare run_dir_root


def _public_signature(sig, root_default=_NO_ROOT):
    """`sig` without `run_dir` / `ctx` / `run_dir_root`, plus a keyword-only
    `run_dir: Path | None = None` and, if the body declares it, `run_dir_root:
    str | None` with the body's template as its default (before a `**kwargs`,
    if there is one): a CLI built from the signature shows the template."""
    kept = [p for p in sig.parameters.values() if p.name not in (*INJECTED, ROOT)]
    added = [inspect.Parameter("run_dir", inspect.Parameter.KEYWORD_ONLY, default=None,
                               annotation=pathlib.Path | None)]
    if root_default is not _NO_ROOT:
        added.append(inspect.Parameter(ROOT, inspect.Parameter.KEYWORD_ONLY,
                                       default=root_default, annotation=str | None))
    at = next((i for i, p in enumerate(kept)
               if p.kind is inspect.Parameter.VAR_KEYWORD), len(kept))
    return sig.replace(parameters=[*kept[:at], *added, *kept[at:]])


def _call_args(public, args, kwargs):
    """The call's arguments by name, defaults filled in, for `config.yaml`
    (without `run_dir`). A call that does not fit the signature raises
    here, before a run dir is made."""
    bound = public.bind(*args, **kwargs)
    bound.apply_defaults()
    return {k: _plain(v) for k, v in bound.arguments.items() if k not in ("run_dir", ROOT)}


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


def _fields(template):
    """The argument names a template refers to (`{runtime}`, `{env[HOME]}` -> runtime, env)."""
    names = []
    for _, field, _, _ in string.Formatter().parse(template):
        if field is not None:
            names.append(field.split("[")[0].split(".")[0])
    return names


def _check_template(template, params, fname):
    """A `run_dir_root` template refers only to the body's arguments and `env`."""
    for name in _fields(template):
        if name == ENV:
            if ENV in params:
                raise TypeError(f"@rundir: {fname} has an argument `{ENV}`, which `{ROOT}`'s "
                                f"{{{ENV}[...]}} would shadow")
        elif name not in params or name in (*INJECTED, ROOT):
            raise TypeError(f"@rundir: `{ROOT}` template {template!r}: {fname} has no argument "
                            f"`{name}`")


def _root(template, call):
    """The runs root of a fresh run dir: `template` (the call's `run_dir_root`,
    or the body's default) filled in with the call's arguments (as in
    `config.yaml`) and `env` (the environment); `resolve_runs_root()` without
    one, or when an argument it refers to is None. A relative path is relative
    to the current folder."""
    if template is None:
        return resolve_runs_root()
    template = str(template)
    for name in _fields(template):
        if name != ENV and name not in call:
            raise ValueError(f"{ROOT} {template!r}: no argument `{name}`")
        if name != ENV and call[name] is None:
            return resolve_runs_root()
    try:
        value = template.format(**call, **{ENV: os.environ})
    except KeyError as e:
        raise ValueError(f"{ROOT} {template!r}: no environment variable {e}") from None
    return pathlib.Path(value).expanduser().resolve()


def _create(name, explicit, *, root, script, module, call, full):
    """Make the run dir. With `full` (the body takes `ctx`), write what the call
    was: `config.yaml` (its arguments), `context.yaml` and `meta.yaml`; without
    it, write nothing -- the folder is the body's. A fresh one under the runs root becomes `latest`; an explicit
    folder must be new or empty, and gets no `latest`."""
    hex8 = uuid.uuid4().hex[:8]
    if explicit is None:
        run_dir = _resolve_dir(root, name, None, hex8)
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = pathlib.Path(explicit).expanduser().resolve()
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise FileExistsError(f"{run_dir}: already exists and is not empty -- "
                                  f"a run dir starts empty")
        run_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(dir=run_dir, id=f"{name}_{hex8}", name=name)
    if full:
        _dump_yaml(run_dir / "config.yaml", call)
        _dump_yaml(run_dir / CONTEXT, {"id": ctx.id, "name": name})
        _dump_yaml(run_dir / "meta.yaml",
                   {"kind": "rundir", "script": str(script), "module": module,
                    "argv": list(sys.argv), "cwd": str(pathlib.Path.cwd()),
                    "launch": _launch()})
    if explicit is None:
        point_latest(run_dir)
    return ctx


def _report(ctx, status, duration_s, error, full):
    """The closing line; a failure points at traceback.txt only if there is one.
    Best-effort: it runs with the body's exception in flight."""
    try:
        ui.run_finished(run_id=ctx.id, status=status, duration_s=duration_s,
                        error=error, run_dir=ctx.dir, traceback=full)
    except Exception:                                        # noqa: BLE001
        pass


def _remove_if_empty(folder):
    """Remove `folder` if nothing was written into it. Best-effort: it runs at
    the end of the run, possibly with the body's exception in flight."""
    try:
        folder.rmdir()
    except OSError:
        pass
