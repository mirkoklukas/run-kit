"""The @experiment decorator and the machinery behind it.

A decorated function is `run(cfg, ctx)`. Before the body runs, the decorator:
  1. resolves ctx fields (decorator args -> context yaml -> defaults)
  2. captures provenance and refuses to start on uncommitted code
  3. creates an immutable run dir and freezes config/provenance/env into it
  4. injects a RunContext and stamps completed/failed on exit

ctx fields resolvable in four layers (last wins):
  builtin defaults -> @experiment(...) args -> context yaml -> CLI --flags
There is NO auto-discovery of run.context.yaml: by default you get the
builtin defaults (runs_root = ./runs, no tracked repos). Pass --context=PATH
(or context= to @experiment) to opt into a yaml layer. The CLI layer is
applied by callers (e.g. runkit.main) via _ctx_overrides.
"""
import dataclasses
import datetime
import functools
import json
import pathlib
import sys
import uuid

import yaml

from .provenance import software_record, hardware_record, find_lockfile


# Lowest layer of the ctx precedence stack (see the wrapper's `pick`): the value
# used when neither CLI --flags, a context yaml, nor @experiment(...) set a field.
_DEFAULTS = {
    "runs_root": "runs",     # ./runs under cwd; run dirs created here
    "allow_dirty": False,
    "lockfile": None,
    "repos_in_dev": {},
}


@dataclasses.dataclass
class RunContext:
    out: pathlib.Path       # where the run physically lives (auto path, or --out)
    id: str                 # stable unique run id, e.g. "mock_a3f9c1e7"; independent of out


def _serialize_cfg(cfg):
    """Best-effort convert a cfg into a plain dict for the frozen config.yaml.

    Tries the common shapes in order, ending in a repr fallback so recording the
    config never fails the run.
    """
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)            # the usual case: a @dataclass cfg
    for attr in ("model_dump", "dict"):           # pydantic v2 / v1 models
        if hasattr(cfg, attr):
            try:
                return getattr(cfg, attr)()
            except Exception:
                pass
    if hasattr(cfg, "__dict__"):                  # any plain object with attrs
        return dict(vars(cfg))
    return {"repr": repr(cfg)}                     # last resort: never lose it entirely


def _write_json(path, obj):
    """Write `obj` as indented JSON (default=str stringifies Path/datetime)."""
    path.write_text(json.dumps(obj, indent=2, default=str))


def _resolve_out(runs_root, name, tag, out_override):
    """Return the run dir path only. If `out_override` is set, use it (collision-
    safe: appends a timestamp). Otherwise an auto path under
    runs_root/<date>/<time>_<label>. The run's `id` is separate (see init_run)."""
    if out_override is not None:
        p = pathlib.Path(out_override).resolve()
        if p.exists():
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            p = p.parent / f"{p.name}_{ts}"
        return p
    now = datetime.datetime.now()
    label = f"{name}_{tag}" if tag else name
    return pathlib.Path(runs_root).resolve() / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{label}"


def init_run(cfg, *, name, tag=None, out=None, repos_in_dev,
             runs_root, lockfile, allow_dirty):
    """Create the immutable run dir, freeze the run spec into it, return a ctx.

    Order is deliberate: capture provenance and enforce the dirty-git gate
    *before* touching the filesystem, so a refused launch leaves nothing behind.
    Then lay out the dir skeleton and write the four frozen records that make the
    run reproducible (config + software/hardware axes) and searchable (the uid).
    """
    # software axis + dirty gate: refuse to launch on uncommitted code, but only
    # for the repos we were told to track (repos_in_dev); empty => no gate.
    sw, any_dirty = software_record(repos_in_dev, lockfile, " ".join(sys.argv))
    if any_dirty and not allow_dirty:
        dirty = [n for n, r in sw["repos_in_dev"].items() if r["dirty"]]
        sys.exit(f"Refusing to launch: uncommitted changes in {dirty}. "
                 f"Commit, or pass --allow-dirty.")

    out_path = _resolve_out(runs_root, name, tag, out)
    # Stable unique id, independent of where `out` lands (so --out can't pollute
    # it), findable via `grep -rl <id> runs/*/*/status.json`.
    # TODO: fold `tag` into the id too (e.g. f"{name}_{tag}_{uid}") once tags settle.
    uid = f"{name}_{uuid.uuid4().hex[:8]}"

    # dir skeleton. exist_ok=False on checkpoints makes a colliding auto path
    # (same second + label) fail loudly instead of merging into an existing run.
    (out_path / "checkpoints").mkdir(parents=True, exist_ok=False)
    (out_path / "logs").mkdir(parents=True, exist_ok=True)
    (out_path / "results").mkdir(parents=True, exist_ok=True)

    # the four frozen records: the cfg (what to run) + the axes it ran on + status
    (out_path / "config.yaml").write_text(
        yaml.safe_dump(_serialize_cfg(cfg), sort_keys=False))
    _write_json(out_path / "provenance.json", sw)             # software axis
    _write_json(out_path / "env.json", hardware_record())     # hardware axis
    _write_json(out_path / "status.json",                     # live status + uid
                {"status": "running", "id": uid,
                 "name": name, "tag": tag})

    return RunContext(out=out_path, id=uid)


def _mark(out, status, **extra):
    """Patch status.json to a terminal state (completed/failed) + an end time.

    Read-modify-write, so the fields init_run wrote (id, name, tag) survive.
    """
    p = pathlib.Path(out) / "status.json"
    data = json.loads(p.read_text()) if p.is_file() else {}
    data.update(
        status=status,
        ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **extra,
    )
    _write_json(p, data)


# ctx fields the decorator and CLI can both set. Order matters only as
# a closed enumeration: any extra kwarg passed to @experiment that isn't
# here is rejected so typos surface early.
CTX_FIELDS = ("repos_in_dev", "runs_root", "lockfile", "allow_dirty",
              "tag", "out", "context")


def experiment(*, name, **decorator_ctx):
    """Mark a function as an experiment entry point.

    Required:
        name: short label for the run dir (e.g. "hexapod_walk").

    Optional ctx fields (override context yaml entries):
        repos_in_dev, runs_root, lockfile, allow_dirty, tag, out, context

    The wrapped function is called as `f(cfg, ctx=RunContext)`. Pass extra
    ctx overrides via a `_ctx_overrides=` kwarg at call time (runkit.main
    does this with CLI --flags).
    """
    bad = [k for k in decorator_ctx if k not in CTX_FIELDS]
    if bad:
        raise TypeError(
            f"@experiment got unknown kwarg(s) {bad}; allowed: {CTX_FIELDS}")

    def decorator(f):
        @functools.wraps(f)
        def wrapper(cfg, *args, _ctx_overrides=None, **kwargs):
            # Runs on every invocation: resolve ctx -> build run dir -> run body
            # -> stamp status. `cfg` is already built (by runkit.main); the CLI
            # --flags arrive as `_ctx_overrides`, the top precedence layer.
            ctx_overrides = _ctx_overrides or {}
            context_path = ctx_overrides.get("context") \
                or decorator_ctx.get("context")
            # No auto-discovery: builtin defaults unless --context is given.
            context_yaml = {}
            if context_path:
                context_yaml = yaml.safe_load(
                    pathlib.Path(context_path).read_text()) or {}

            # Resolve each ctx field with the documented precedence:
            # CLI --flag > context yaml > @experiment(...) > builtin defaults
            def pick(field, default=None):
                if field in ctx_overrides:
                    return ctx_overrides[field]
                if field in context_yaml:
                    return context_yaml[field]
                if field in decorator_ctx:
                    return decorator_ctx[field]
                return default

            repos = pick("repos_in_dev", _DEFAULTS["repos_in_dev"])
            root  = pick("runs_root",    _DEFAULTS["runs_root"])
            dirty = pick("allow_dirty",  _DEFAULTS["allow_dirty"])
            lock  = pick("lockfile",     _DEFAULTS["lockfile"])
            tag   = pick("tag")
            out   = pick("out")

            # No explicit lockfile? auto-detect the one defining this venv.
            if lock is None:
                found = find_lockfile()
                lock = str(found) if found else None

            ctx = init_run(cfg, name=name, tag=tag, out=out,
                           repos_in_dev=repos, runs_root=root,
                           lockfile=lock, allow_dirty=dirty)
            # run the body inside the prepared run dir; stamp the outcome either
            # way -- completed on return, failed (+ reraise) on any exception.
            try:
                result = f(cfg, *args, ctx=ctx, **kwargs)
                _mark(ctx.out, "completed")
                return result
            except BaseException as e:
                _mark(ctx.out, "failed", error=repr(e))
                raise
        wrapper._runkit_name = name          # for `--help` and introspection
        # where the experiment lives, for resolving `exp:`-prefixed config paths
        wrapper._runkit_dir = pathlib.Path(f.__code__.co_filename).resolve().parent
        wrapper._runkit_ctx = decorator_ctx
        return wrapper
    return decorator
