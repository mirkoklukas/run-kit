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
import inspect
import os
import pathlib
import socket
import sys
import time
import traceback
import uuid

import yaml

from . import ui
from .settings import resolve_follow, resolve_root
from .utils import config_changes, dump_retval, point_latest, serialize_cfg


PROGRESS_EVERY_S = 2.0       # ctx.progress writes status.yaml at most this often


def _private():
    """A RunContext field about *this process*, not the run: never written to
    run_context.yaml, and not part of comparing contexts."""
    return dataclasses.field(default=None, repr=False, compare=False)


@dataclasses.dataclass
class _Live:
    """What only the process running the body knows. The run wrapper creates it
    before the body and drops it after, which is what `ctx.live` means."""
    t0: float                          # monotonic start of the body
    started: str                       # the `started` stamp, to rewrite status.yaml
    checkpoints: int = 0               # checkpoints made so far: the counter
    progress: object = None            # ctx.progress: the count ...
    total: object = None               # ... and the total, sticky
    checkpoint: str | None = None      # the latest complete checkpoint, relative to the run dir
    written: float | None = None       # monotonic time progress was last written
    records: dict = dataclasses.field(default_factory=dict)   # lines appended per metrics stream
    windows: dict = dataclasses.field(default_factory=dict)   # per stream: rows since the last checkpoint
    follow: str | None = None          # the stream printed as it is recorded (--follow)
    table: object = None               # ... and the table that prints it



@dataclasses.dataclass
class RunContext:
    dir: pathlib.Path        # the run dir; its top level is runkit's records
    id: str                  # stable unique run id, e.g. "baseline_a3f9c1e7" (for search)
    name: str | None = None  # the experiment's identity, as given to @experiment
    _live: _Live | None = _private()        # set by the run wrapper while the body runs
    _opened: float | None = _private()      # monotonic time a context was opened from disk

    @property
    def out(self) -> pathlib.Path:
        """`{dir}/out` -- everything the experiment writes goes here."""
        return self.dir / "out"

    @property
    def live(self) -> bool:
        """True in the context the run wrapper hands the body -- the process that
        owns the run right now; False in one rebuilt from disk (`load_run`, and
        so in eval and viz). Only a live context may write into the run."""
        return self._live is not None

    def _require_live(self, what):
        if self._live is None:
            raise RuntimeError(
                f"{what}: only the live run can do this -- this context was opened "
                f"from disk (eval, viz, load_run) or its run has finished")
        return self._live

    def progress(self, n=None, /, *, total=None):
        """How far along the run is: `n` of `total`, written into `status.yaml`
        as `progress` / `total`.

        `n` is any count (steps, iterations, candidates tried). `total` is
        sticky: set it once -- `ctx.progress(total=cfg.steps)` at the start --
        and later calls pass only `n`; it stays null for an open-ended run. A
        write happens at most every PROGRESS_EVERY_S seconds (or when `total`
        changes), so calling it every iteration is fine; the run's final
        status.yaml keeps the last values. Live runs only.
        """
        live = self._require_live("ctx.progress")
        if n is not None:
            live.progress = _count(n, "n")
        if total is not None:
            live.total = _count(total, "total")
        now = time.monotonic()
        if (total is not None or live.written is None
                or now - live.written >= PROGRESS_EVERY_S):
            live.written = now
            self._write_running()

    def _write_running(self):
        """Rewrite status.yaml mid-run with what the body has reported so far."""
        live = self._live
        _write_status(self.dir, status="running", started=live.started,
                      progress=live.progress, total=live.total, checkpoint=live.checkpoint)

    def record(self, stream=None, /, **values):
        """Append a row of named values to `{dir}/metrics/<stream>.jsonl`.

        `ctx.record(it=it, ep_return=r)` records to the run's own stream, `run`;
        `ctx.record("eval", ep_return=r)` to a stream of that name. runkit adds
        `time` and `elapsed_s`. Only the live run writes `run` (and may omit the
        stream); a context opened from disk -- in eval, say -- must name its
        stream. See `runkit.metrics`.
        """
        from .metrics import append
        if stream is None:
            if not self.live:
                raise RuntimeError(
                    "ctx.record: name a stream -- only the live run records to the "
                    "default stream 'run' (e.g. ctx.record('eval', ...))")
            stream = "run"
        elif stream == "run" and not self.live:
            raise RuntimeError("ctx.record: the stream 'run' is the run's own; "
                               "only the live run writes it")
        append(self, stream, values)

    def note(self, message, /, **values):
        """Say something about the run, and keep it: `ctx.note("support ramp
        starts", steps=n)`.

        Printed as a line between the rows (`◇ support ramp starts  steps=500k`;
        the `--follow` table names its columns again after it), and recorded to
        `{dir}/metrics/notes.jsonl` -- `note` and the values, with `_time` and
        `_elapsed_s` -- so it stays with the run, and `runkit metrics follow`
        shows it from another terminal. Notes are events, not metrics: they are
        not in the checkpoint summaries. Live runs only.
        """
        from .metrics import NOTES, append
        live = self._require_live("ctx.note")
        if "note" in values:
            raise ValueError("ctx.note: `note` is the message's key; name the value otherwise")
        append(self, NOTES, {"note": str(message), **values})
        try:                                  # best-effort: never costs the run
            ui.note(str(message), values)
            if live.table is not None:
                live.table.interrupt()
        except Exception:                                    # noqa: BLE001
            pass

    def checkpoint(self, name=None):
        """`with ctx.checkpoint(name=None) as ckpt:` -- save into `ckpt.state`.

        `{dir}/checkpoints/<name>/`, named by runkit's counter (`000003`) unless
        `name` is given; complete, recorded and made `latest` only when the block
        exits cleanly. A repeated name replaces the earlier checkpoint. See
        `runkit.checkpoints`. Live runs only.
        """
        from .checkpoints import _Saving
        self._require_live("ctx.checkpoint")
        return _Saving(self, name)

    def checkpoints(self):
        """The run's complete checkpoints, oldest first (by index). Works on any
        context, live or opened from disk."""
        from .checkpoints import load_checkpoints
        return load_checkpoints(self.dir)


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
    """Write yaml atomically: to a temp file beside `path`, then rename over it.

    Other processes read these files while a run writes them (`status.yaml`
    most of all); a rename means a reader sees the old file or the new one,
    never half of one.
    """
    path = pathlib.Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False))
    os.replace(tmp, path)


def _stamp(when=None):
    """A second-resolution ISO timestamp -- what goes in `status.yaml`."""
    return (when or datetime.datetime.now()).isoformat(timespec="seconds")


def _count(v, what):
    """A progress count as a plain int or float (numpy scalars included)."""
    v = v.item() if hasattr(v, "item") and not isinstance(v, (int, float)) else v
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"ctx.progress: {what} must be a number, got {v!r}")
    return v


def _write_status(run_dir, *, status, started, ended=None, duration_s=None, error=None,
                  progress=None, total=None, checkpoint=None):
    """Write `{run_dir}/status.yaml`. Best-effort, and for a sharper reason than
    `dump_retval`: the final write happens inside a `finally` with the
    experiment's exception in flight, so a failure here must never replace it.

    `updated` is when the file was last written. `host` and `pid` name the
    process that owns the run -- a pid means something only on its host -- so a
    run left at `running` can be checked for a process behind it. `progress` /
    `total` are the body's last `ctx.progress`; `checkpoint` is the latest
    complete checkpoint, as a path relative to the run dir.
    """
    try:
        _dump_yaml(pathlib.Path(run_dir) / "status.yaml", {
            "status": status, "started": started, "updated": _stamp(),
            "host": socket.gethostname(), "pid": os.getpid(),
            "progress": progress, "total": total, "checkpoint": checkpoint,
            "ended": ended, "duration_s": duration_s, "error": error,
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


def _launch():
    """What runkit set up for this process from experiment.toml (extras, vars,
    uv project), or None -- e.g. a plain `python experiment.py`."""
    import json
    from .launch import LAUNCH
    raw = os.environ.get(LAUNCH)
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def init_run(cfg, *, name, tag, root, script=None, module=None, branch=None):
    """Create the run dir, freeze the run into it, return a RunContext.

    Writes the three files that say what the run was -- `config.yaml` (the
    resolved cfg), `run_context.yaml` (the RunContext) and `meta.yaml` (how the
    attempt was staged) -- and creates the empty `out/` the body writes into.
    `status.yaml` is not written here: nothing is running
    yet, and the lifecycle belongs to whoever calls the body.

    The run id and the dir's `{hex8}` share one uuid, so the dir is
    self-identifying (`id = {name}_{hex8}`) and never collides. It also becomes
    `{root}/{name}/latest`. `branch`: the checkpoint a branch starts from, recorded
    in `meta.yaml` as its lineage (`_lineage`).
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
                "module": module, "launch": _launch(),
                **({"branch": _lineage(branch)} if branch is not None else {})})
    point_latest(run_dir)                    # {root}/{name}/latest: the latest *started* run
    return ctx


def _lineage(ckpt):
    """Where a branch comes from, for `meta.yaml`: the parent run (id and dir) and
    its checkpoint (name, index, and the step count if its `info` has one)."""
    try:
        parent_id = yaml.safe_load((ckpt.run / "run_context.yaml").read_text())["id"]
    except Exception:                                        # noqa: BLE001
        parent_id = ckpt.run.name
    return {"run": parent_id, "dir": str(ckpt.run), "checkpoint": ckpt.name,
            "index": ckpt.index,
            **({"steps": ckpt.info["steps"]} if "steps" in ckpt.info else {})}


def _announce(name, cfg, ctx, tag, branch=None, derived=None):
    """Print a one-time start banner: which run, where, and what it changes.

    Only fields that differ from the defaults are echoed -- for a branch, from
    its parent's config; the whole resolved config is frozen at
    `{dir}/config.yaml`.
    """
    parent = _parent_cfg(cfg, branch) if branch is not None else None
    changes, n_fields = config_changes(cfg, base=parent)
    ui.run_started(name=name, run_id=ctx.id, run_dir=ctx.dir, tag=tag,
                   changes=changes, n_fields=n_fields, launch=_launch(),
                   branch=_branch_text(branch) if branch is not None else None,
                   against="the parent" if parent is not None else "defaults",
                   derived=derived)


def _parent_cfg(cfg, ckpt):
    """The parent run's config, as `cfg`'s class; None if it no longer builds
    (the class changed since) -- the banner then compares with the defaults."""
    from .config import build_cfg
    from .utils import load_yaml
    try:
        return build_cfg(type(cfg), load_yaml(ckpt.run / "config.yaml"))
    except Exception:                                        # noqa: BLE001
        return None


def _branch_text(ckpt):
    """`a3f9c1e7:best (3.0M steps)` -- the banner's `branch` line."""
    from .runs import dir_hex
    steps = ckpt.info.get("steps")
    return f"{dir_hex(ckpt.run)}:{ckpt.name}" + (
        f" ({ui._num(steps)} steps)" if isinstance(steps, (int, float)) else "")


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
        """Register `f(ckpt)` as the eval: it gets a checkpoint of a run
        (`_checkpoint_wrapper`). The older `f(cfg, ctx)` opens a finished (`ok`)
        run dir instead."""
        return self._register("eval", _eval_wrapper(f, self))

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

    The wrapper accepts the staging flags as keyword args -- `tag`, `root`,
    `follow`, `branch` -- which `autocli` forwards from the CLI; their names *are*
    the allowed flags. `follow` names the stream printed as the run goes (default
    `run`, from experiment.toml's `follow`; `none` for nothing).
    Without `root`, it comes from the nearest `experiment.toml`, else `./runs`
    (`settings.resolve_root`).

    `branch` starts the run from a checkpoint: `RUN[:CHECKPOINT]` (see
    `runs.select_checkpoint`) or a `Checkpoint`. Only a body that declares a
    `branch` parameter -- `run(cfg, ctx, branch=None)` -- can be branched; it
    gets the `Checkpoint`, or None for a fresh run. A body without one is
    called `f(cfg, ctx)`, as always. `cfg` is used as given: from the CLI,
    `autocli` builds it on the parent's config.
    """
    script = pathlib.Path(f.__code__.co_filename).resolve()
    module = _module_name(f)
    takes_branch = "branch" in inspect.signature(f).parameters

    @functools.wraps(f)
    def wrapper(cfg, *, tag=None, root=None, follow=None, branch=None, _derived=None):
        # `_derived`: how `key*=2` values came about, for the banner (from autocli)
        root = resolve_root(script, script.stem, explicit=root)
        follow = resolve_follow(script, script.stem, explicit=follow)
        branch = _branch_checkpoint(wrapper, branch, root)
        ctx = init_run(cfg, name=name, tag=tag, root=root, script=script,
                       module=module, branch=branch)
        _announce(name, cfg, ctx, tag, branch, derived=_derived)
        started, t0 = datetime.datetime.now(), time.monotonic()
        ctx._live = _Live(t0=t0, started=_stamp(started))   # live: the body may write into the run
        if follow is not None:               # print this stream's rows as they are recorded
            from .follow import _Table
            ctx._live.follow, ctx._live.table = follow, _Table(None, stderr=True)
        _write_status(ctx.dir, status="running", started=_stamp(started))
        status, error = "ok", None
        try:
            result = f(cfg, ctx, branch=branch) if takes_branch else f(cfg, ctx)
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
            live, ctx._live = ctx._live, None     # the body is done: no longer live
            _write_status(ctx.dir, status=status, started=_stamp(started),
                          progress=live.progress, total=live.total,
                          checkpoint=live.checkpoint,
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
    wrapper._runkit_takes_branch = takes_branch
    return wrapper


def _branch_checkpoint(wrapper, branch, root):
    """The `Checkpoint` a run branches from, or None; refuses what cannot be
    continued: a body without a `branch` parameter, or a checkpoint whose
    `state/` is empty (the body never saved anything)."""
    from .checkpoints import Checkpoint
    from .runs import select_checkpoint
    if branch is None:
        return None
    if not wrapper._runkit_takes_branch:
        raise ValueError(
            f"{wrapper.__name__} takes no `branch`: it can't continue from a "
            f"checkpoint (declare it: def {wrapper.__name__}(cfg, ctx, branch=None))")
    ckpt = branch if isinstance(branch, Checkpoint) else \
        select_checkpoint(root, wrapper._runkit_name, branch)
    if ckpt.is_empty():
        raise ValueError(f"{ui.short_path(ckpt.dir)}: nothing saved in its state/ -- "
                         f"nothing to continue from")
    return ckpt


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


def _eval_wrapper(f, exp):
    """An eval body `f(ckpt)` gets a checkpoint; one with a `cfg` parameter --
    `f(cfg, ctx)`, from before -- opens a run dir, as viz does."""
    if "cfg" in inspect.signature(f).parameters:
        return _open_wrapper(f, exp, "eval")
    return _checkpoint_wrapper(f, exp)


def _checkpoint_wrapper(f, exp):
    """Wrap an eval body `f(ckpt)`: pick a checkpoint, call `f` with it.

    The wrapper is `fn(which=None, *, root=None)`. `which` picks the checkpoint
    (see `runs.select_checkpoint`): None for the latest checkpoint of the latest
    run that has one, a checkpoint dir, a run dir (its latest), or
    `RUN:CHECKPOINT`. A run that failed or is still going is evaluated at its
    latest complete checkpoint. The body gets the `Checkpoint`: it reads
    `ckpt.state`, writes into `ckpt.eval` (made here), and records with
    `runkit.record(ckpt.eval / "x.jsonl", ...)`. Returns what the body returns.
    """
    script = pathlib.Path(f.__code__.co_filename).resolve()

    @functools.wraps(f)
    def wrapper(which=None, *, root=None):
        from .runs import select_checkpoint
        root = resolve_root(script, script.stem, explicit=root)
        ckpt = select_checkpoint(root, exp.name, which)
        ckpt.eval.mkdir(exist_ok=True)
        ui.opened(name=exp.name, verb="eval", run_dir=ckpt.dir)
        ui.checkpoint_opened(ckpt)
        return f(ckpt)
    wrapper._runkit_name = exp.name
    wrapper._runkit_script = script
    wrapper._runkit_dir = script.parent
    wrapper._runkit_checkpoint = True    # takes a checkpoint, not a run
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
