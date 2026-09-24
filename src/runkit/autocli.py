"""argv-parsing entry point for the lightweight runner.

Shared by `python experiment.py ...` and `runkit <verb> experiment.py ...`
(which hands it `<verb> ...`). The first token may name a verb -- `run` (the default), `eval`, `viz` -- and the rest is
that verb's arguments:

  run   [config.yaml] [key=value ...] [--tag=..] [--root=..]   -> a new run dir
  eval  [RUN] [--root=..]                                      -> opens one
  viz   [RUN] [--root=..]                                      -> opens one

and two built in, for every experiment, that print a path (`cd "$(...)"`):

  root    [--root=..]   -> {root}/{name}, this experiment's runs
  latest  [--root=..]   -> its latest run dir

Two disjoint namespaces (see design.md):
  bare key=value -> cfg overrides   (the "what")
  --flag[=value] -> staging flags   (the "how/where": tag, root)

Resolution (cfg): dataclass defaults -> config.yaml -> CLI key=value. last wins.
"""
import dataclasses
import inspect
import sys

from .config import annotated_cfg, build_cfg, deep_merge, parse_overrides, split_argv
from .utils import load_yaml, resolve_config_path

VERBS = ("run", "eval", "viz")
PATH_VERBS = ("root", "latest")      # built in: print a path, need no registration


def main(run=None, argv=None, *, eval=None, viz=None):
    """Dispatch argv to `run`, or to the `eval` / `viz` given alongside it.

    The functional form of `Experiment.main`: `main(run)` at the bottom of an
    `@experiment` file keeps working, and also picks up whatever else is
    registered on `run`'s experiment. `eval=` / `viz=` take plain `(cfg, ctx)`
    bodies and wrap them for that experiment.
    """
    from .exp import _open_wrapper
    exp = getattr(run, "_runkit_experiment", None)
    if exp is None:
        raise TypeError("main(run): `run` must be decorated with @experiment "
                        "or registered with @exp.run")
    roles = dict(exp.roles)
    for verb, fn in (("eval", eval), ("viz", viz)):
        if fn is not None:
            roles[verb] = (fn if hasattr(fn, "_runkit_experiment")
                           else _open_wrapper(fn, exp, verb))
    return _dispatch(exp.name, roles, argv)


def dispatch(exp, argv=None):
    """`Experiment.main`: dispatch argv to the verbs registered on `exp`."""
    return _dispatch(exp.name, exp.roles, argv)


def _dispatch(name, roles, argv):
    from .runs import RunNotFound
    argv = sys.argv[1:] if argv is None else list(argv)
    verb = "run"
    if argv and argv[0] in VERBS + PATH_VERBS:
        verb, argv = argv[0], argv[1:]
    if verb in PATH_VERBS:
        return _path_verb(name, roles, verb, argv)
    fn = roles.get(verb)
    if fn is None:
        sys.exit(f"{name}: no {verb} function registered "
                 f"(verbs here: {', '.join(v for v in VERBS if v in roles)})")
    if "--help" in argv or "-h" in argv:
        print(_help_text(name, verb, fn, roles))
        return
    try:
        if verb == "run":
            cfg, flags = _run_args(fn, argv)
        else:
            which, flags = _open_args(fn, verb, argv)
    except (ValueError, TypeError) as e:
        sys.exit(str(e))

    if verb == "run":
        return fn(cfg, **flags)             # flags are exactly the staging kwargs
    try:
        return fn(which, **flags)
    except RunNotFound as e:                # the selection, not the body, failed
        sys.exit(str(e))


def _path_verb(name, roles, verb, argv):
    """`root` / `latest`: print this experiment's runs folder, or its latest run
    dir. Only the path goes to stdout, so `cd "$(... latest)"` works."""
    from . import ui
    from .runs import run_dirs
    from .settings import resolve_root
    if "--help" in argv or "-h" in argv:
        what = "its runs folder, {root}/" + name if verb == "root" else "its latest run dir"
        print(f"usage: <experiment> {verb} [--root=DIR]\n\nprints {what}")
        return
    cfg_tokens, flags, positionals = split_argv(argv)
    unknown = sorted(set(flags) - {"root"})
    if cfg_tokens or positionals or unknown:
        sys.exit(f"usage: <experiment> {verb} [--root=DIR]")
    if not roles:
        sys.exit(f"{name}: nothing registered, so no file to resolve its root from")
    script = next(iter(roles.values()))._runkit_script
    root = resolve_root(script, script.stem, explicit=flags.get("root"))
    if verb == "root":
        path = (root / name).resolve()
        if not path.is_dir():
            ui.warn(f"{path} does not exist yet (no runs of {name!r} made there)")
    else:
        dirs = run_dirs(root, name)
        if not dirs:
            sys.exit(f"no runs of {name!r} under {(root / name).resolve()}")
        path = dirs[-1].resolve()
    print(path)
    return path


def _run_args(run, argv):
    """argv -> (cfg, staging flags) for the run verb."""
    cfg_tokens, flags, positionals = split_argv(argv)

    # config layer: a yaml (positional or --config), then key=value on top.
    config_file = _config_file(flags.pop("config", None), positionals)
    if config_file is not None:
        config_file = resolve_config_path(
            config_file, getattr(run, "_runkit_dir", None))
    base = load_yaml(config_file) if config_file else {}
    cfg = build_cfg(_cfg_type(run), deep_merge(base, parse_overrides(cfg_tokens)))

    _check_flags(run, flags)            # reject unknown --flags before we run
    return cfg, flags


def _open_args(fn, verb, argv):
    """argv -> (which run, staging flags) for eval / viz."""
    cfg_tokens, flags, positionals = split_argv(argv)
    if cfg_tokens:
        raise ValueError(f"{verb} takes no key=value overrides (got {cfg_tokens}); "
                         f"the config is the one the run was made with")
    if len(positionals) > 1:
        raise ValueError(f"{verb} takes at most one run (got {positionals})")
    _check_flags(fn, flags)
    return (positionals[0] if positionals else None), flags


def _config_file(flag_value, positionals):
    """Reconcile the two ways to name a base config: a bare positional path and
    `--config=PATH`. At most one source; they must not conflict."""
    if len(positionals) > 1:
        raise ValueError(
            f"unexpected extra arguments {positionals}; only one bare config "
            f"path is allowed (use key=value for overrides)")
    positional = positionals[0] if positionals else None
    if flag_value and positional and flag_value != positional:
        raise ValueError(
            f"config given twice: --config={flag_value} and positional {positional}")
    return flag_value or positional


def _check_flags(fn, flags):
    """Reject any --flag the wrapper doesn't accept (the staging kwargs)."""
    allowed = _staging_flags(fn)
    bad = [k for k in flags if k not in allowed]
    if bad:
        raise ValueError(f"unknown flag(s): {sorted(bad)}; allowed: {sorted(allowed)}")


def _staging_flags(fn):
    """The keyword-only params of the *wrapper* = the allowed staging flags.

    `follow_wrapped=False` so we read the wrapper's own signature (tag, root),
    not the underlying body(cfg, ctx) it wraps.
    """
    sig = inspect.signature(fn, follow_wrapped=False)
    return {p.name for p in sig.parameters.values()
            if p.kind == inspect.Parameter.KEYWORD_ONLY}


def _cfg_type(run):
    """The dataclass annotated on the wrapped run's `cfg` parameter."""
    if "cfg" not in inspect.signature(run).parameters:
        raise TypeError(f"{getattr(run, '__name__', run)!r} must accept a 'cfg' parameter")
    cls = annotated_cfg(run)
    if cls is None:
        raise TypeError(
            "the run's `cfg` parameter needs a dataclass type annotation so "
            "runkit knows what to build from CLI args")
    return cls


def _flag_lines(fn):
    return [f"  --{flag.replace('_', '-')}" for flag in sorted(_staging_flags(fn))]


def _help_text(name, verb, fn, roles):
    verbs = ", ".join([v for v in VERBS if v in roles] + list(PATH_VERBS))
    if verb != "run":
        what = ("the latest finished (ok) run" if verb == "eval"
                else "the latest run")
        return "\n".join([
            f"usage: <experiment> {verb} [RUN] [--flag ...]",
            "",
            f"experiment: {name}   (verbs: {verbs}; default: run)",
            "",
            "RUN: a run dir, or a hex prefix of a run's id",
            f"     (default: {what} under {{root}}/{name})",
            "",
            "staging flags (--flag):",
            *_flag_lines(fn),
        ])
    cfg_cls = _cfg_type(fn)
    lines = [
        "usage: <experiment> [run] [config.yaml] [key=value ...] [--flag ...]",
        "",
        f"experiment: {name}   (cfg: {cfg_cls.__name__}; verbs: {verbs}; default: run)",
        "",
        "config overrides (bare key=value):",
    ]
    for fld in dataclasses.fields(cfg_cls):
        lines.append(f"  {fld.name}={fld.default!r}")
    lines += ["", "staging flags (--flag):", *_flag_lines(fn)]
    lines += ["  --config=PATH   (cwd:/exp:/SCHEME:/abs)"]
    return "\n".join(lines)
