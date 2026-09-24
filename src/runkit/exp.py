"""`Experiment`, its roles, and the run dir a run builds.

An `Experiment` holds an experiment's identity and the functions registered for
its verbs (`run`, `eval`, `viz`); `@experiment(name=...)` is the run-only
shorthand. A run body is `run(cfg, ctx)`. Its wrapper makes it so that, before
the body runs, a fresh run dir is created and the run frozen into it; after, a
non-`None` return value is dumped into the run dir and the caller gets a `Run`. `config.yaml`,
`run_context.yaml` and `meta.yaml` are what the run was; `status.yaml` (and
`traceback.txt`) are how that run went. runkit owns the top level of the run
dir; the body owns `out/` (`ctx.out`). Staging is driven by keyword args (the
CLI flags): `tag`, `root`.

The experiment carries only its *identity* (`name`); everything that stages an
attempt is a flag. See design.md.
"""
import dataclasses
import datetime
import functools
import os
import pathlib
import sys
import time
import traceback
import uuid

import yaml

from . import ui
from .settings import resolve_root
from .utils import config_changes, dump_retval, serialize_cfg


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

    {root}/{name}/{date}_{time}_{hex8}[_{tag}]/   (date=YYYY-MM-DD, time=HH-MM-SS)

    Everything fixed-width first, the one free-form part (the tag) last: the
    hex sits at the same place in every name, and the tag is the remainder.
    """
    now = datetime.datetime.now()
    parts = [f"{now:%Y-%m-%d}", f"{now:%H-%M-%S}", hex8, *([tag] if tag else [])]
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


def init_run(cfg, *, name, tag, root, script=None, module=None):
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
               {"tag": tag, "script": str(script) if script else None,
                "module": module})
    _point_latest(run_dir)
    return ctx


def _announce(name, cfg, ctx, tag):
    """Print a one-time start banner: which run, where, and what it changes.

    Only fields that differ from the defaults are echoed; the whole resolved
    config is frozen at `{dir}/config.yaml`.
    """
    changes, n_fields = config_changes(cfg)
    ui.run_started(name=name, run_id=ctx.id, run_dir=ctx.dir, tag=tag,
                   changes=changes, n_fields=n_fields)


def _report(ctx, status, duration_s, error):
    """The closing line. Best-effort, like `_write_status`: it runs with the
    experiment's exception in flight and must never replace it."""
    try:
        ui.run_finished(run_id=ctx.id, status=status, duration_s=duration_s,
                        error=error, run_dir=ctx.dir)
    except Exception:                                        # noqa: BLE001
        pass


# The module name `runkit <verb> <file.py>` imports a file-path target under. Not an
# importable name, so `_module_name` records None for it, as for `python file.py`.
SCRIPT_MODULE = "_runkit_script"


def _module_name(f):
    """The importable module name of `f`, or None for a plain script.

    Run as `python -m pkg.exp`, `f.__module__` is just `__main__`; the real name
    is on `__main__.__spec__`. Run as `python exp.py` (or `runkit run exp.py`) there
    is no such name -- the file is not importable by name, so `script` is all
    there is.
    """
    if f.__module__ == SCRIPT_MODULE:
        return None
    if f.__module__ != "__main__":
        return f.__module__
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return spec.name if spec else None


class Experiment:
    """One experiment: its identity, and the functions that play its roles.

    Shaped like a `typer.Typer` app: one per file, functions registered on it by
    decorator, and `exp.main()` as the entry point. Each role is a verb on the
    command line, `python experiment.py [run|eval|viz] ...` (`run` when omitted):

        exp = Experiment("baseline")

        @exp.run                 # creates a run dir
        def run(cfg: Config, ctx: RunContext): ...

        @exp.eval                # opens one and processes what the run wrote
        def evaluate(cfg: Config, ctx: RunContext): ...

        @exp.viz                 # opens one: the experimenter's "look here first"
        def show(cfg: Config, ctx: RunContext): ...

        if __name__ == "__main__":
            exp.main()

    Every body has the same `(cfg, ctx)` signature; the decorators change the
    calling convention. `@exp.run` gives `run(cfg, *, tag=..., root=...) -> Run`;
    `@exp.eval` / `@exp.viz` give `show(run=None, *, root=...)`, which picks a run
    dir (the latest by default), thaws its config and calls the body.
    """
    VERBS = ("run", "eval", "viz")

    def __init__(self, name):
        self.name = name
        self.roles = {}          # verb -> the registered (wrapped) function

    def __repr__(self):
        return f"Experiment({self.name!r}, roles={sorted(self.roles)})"

    def run(self, f):
        """Register `f(cfg, ctx)` as the run: it creates a fresh run dir."""
        return self._register("run", _run_wrapper(f, self.name))

    def eval(self, f):
        """Register `f(cfg, ctx)` as the eval: it opens a finished (`ok`) run dir."""
        return self._register("eval", _open_wrapper(f, self, "eval"))

    def viz(self, f):
        """Register `f(cfg, ctx)` as the viz: it opens a run dir to present it."""
        return self._register("viz", _open_wrapper(f, self, "viz"))

    def main(self, argv=None):
        """Dispatch argv (default `sys.argv[1:]`) to the registered verb."""
        from .autocli import dispatch
        return dispatch(self, argv)

    def _register(self, verb, wrapper):
        if verb in self.roles:
            raise ValueError(
                f"experiment {self.name!r} already has a {verb} function "
                f"({self.roles[verb].__name__}); one per experiment")
        wrapper._runkit_experiment = self
        self.roles[verb] = wrapper
        return wrapper


def experiment(*, name):
    """Mark `run(cfg, ctx)` as an experiment entry point.

    Shorthand for an `Experiment` with only a run: `@experiment(name="x")` is
    `@Experiment("x").run`. `name` is the experiment's identity (required, no
    CLI override).
    """
    return Experiment(name).run


def _run_wrapper(f, name):
    """Wrap a run body `f(cfg, ctx)`: create the run dir, record the outcome.

    The wrapper accepts the staging flags as keyword args -- `tag`, `root` --
    which `autocli` forwards from the CLI; their names *are* the allowed flags.
    Without `root`, it comes from the nearest `experiment.toml`, else `./runs`
    (`settings.resolve_root`).
    """
    script = pathlib.Path(f.__code__.co_filename).resolve()
    module = _module_name(f)

    @functools.wraps(f)
    def wrapper(cfg, *, tag=None, root=None):
        root = resolve_root(script, script.stem, explicit=root)
        ctx = init_run(cfg, name=name, tag=tag, root=root, script=script,
                       module=module)
        _announce(name, cfg, ctx, tag)
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
            duration_s = round(time.monotonic() - t0, 3)
            _write_status(ctx.dir, status=status, started=_stamp(started),
                          ended=_stamp(), duration_s=duration_s, error=error)
            if status != "ok":
                _report(ctx, status, duration_s, error)
        if result is not None:
            dump_retval(ctx.dir, result)
        _report(ctx, status, duration_s, error)    # after the dump, whose warning comes first
        return Run(config=cfg, context=ctx, retval=result)
    wrapper._runkit_name = name          # identity, for introspection
    # where the experiment lives: `exp:` config paths, experiment.toml lookup
    wrapper._runkit_script = script
    wrapper._runkit_dir = script.parent
    return wrapper


def _open_wrapper(f, exp, verb):
    """Wrap an eval/viz body `f(cfg, ctx)`: open an existing run dir, call `f`.

    The wrapper is `fn(run=None, *, root=None)`. `run` picks the run dir (see
    `runs.select_run`): None for the latest -- for eval the latest `ok` one,
    since there is nothing to evaluate in a run that failed or is still going --
    a path, or a hex prefix of the run's id. The body gets the run's config,
    thawed from `config.yaml`, and its `RunContext`; `ctx.out` is where it
    writes, like the run did. Returns whatever the body returns.
    """
    script = pathlib.Path(f.__code__.co_filename).resolve()

    @functools.wraps(f)
    def wrapper(run=None, *, root=None):
        from .runs import load_run, select_run
        root = resolve_root(script, script.stem, explicit=root)
        run_dir = select_run(root, exp.name, run, require_ok=(verb == "eval"))
        r = load_run(run_dir, _body_cfg_type(f, exp))
        ui.opened(name=exp.name, verb=verb, run_dir=run_dir)
        return f(r.config, r.context)
    wrapper._runkit_name = exp.name
    wrapper._runkit_script = script
    wrapper._runkit_dir = script.parent
    return wrapper


def _body_cfg_type(f, exp):
    """The config class an eval/viz body receives: its own `cfg` annotation, or
    else the run's -- the config is the one the run was made with."""
    from .config import annotated_cfg
    cls = annotated_cfg(f)
    if cls is None and "run" in exp.roles:
        cls = annotated_cfg(exp.roles["run"])
    if cls is None:
        raise TypeError(
            f"{f.__name__!r}: annotate its `cfg` parameter (or register a run) "
            f"so runkit knows which config class to thaw")
    return cls
