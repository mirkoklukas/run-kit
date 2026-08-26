"""argv-parsing entry point for the lightweight runner.

Shared by `python experiment.py ...` and `runkit run ...`. Builds the cfg from
the config layers and forwards the staging flags to the decorated run.

Two disjoint namespaces (see design.md):
  bare key=value -> cfg overrides   (the "what")
  --flag[=value] -> staging flags   (the "how/where": tag, runs_dir, out)

Resolution (cfg): dataclass defaults -> config.yaml -> CLI key=value. last wins.
"""
import dataclasses
import inspect
import sys

from .config import build_cfg, deep_merge, parse_overrides, split_argv
from .utils import load_yaml, resolve_config_path


def main(run, argv=None):
    """Parse argv, build the cfg, and invoke the decorated `run`."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--help" in argv or "-h" in argv:
        print(_help_text(run))
        return
    try:
        cfg_tokens, flags, positionals = split_argv(argv)

        # config layer: a yaml (positional or --config), then key=value on top.
        config_file = _config_file(flags.pop("config", None), positionals)
        if config_file is not None:
            config_file = resolve_config_path(
                config_file, getattr(run, "_runkit_dir", None))
        base = load_yaml(config_file) if config_file else {}
        cfg = build_cfg(_cfg_type(run), deep_merge(base, parse_overrides(cfg_tokens)))

        _check_flags(run, flags)        # reject unknown --flags before we run
    except (ValueError, TypeError) as e:
        sys.exit(str(e))

    return run(cfg, **flags)            # flags are exactly the staging kwargs


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


def _check_flags(run, flags):
    """Reject any --flag the wrapper doesn't accept (the staging kwargs)."""
    allowed = _staging_flags(run)
    bad = [k for k in flags if k not in allowed]
    if bad:
        raise ValueError(f"unknown flag(s): {sorted(bad)}; allowed: {sorted(allowed)}")


def _staging_flags(run):
    """The keyword-only params of the *wrapper* = the allowed staging flags.

    `follow_wrapped=False` so we read the wrapper's own signature (tag, runs_dir,
    out), not the underlying run(cfg, ctx) it wraps.
    """
    sig = inspect.signature(run, follow_wrapped=False)
    return {p.name for p in sig.parameters.values()
            if p.kind == inspect.Parameter.KEYWORD_ONLY}


def _cfg_type(run):
    """The dataclass annotated on the wrapped run's `cfg` parameter.

    Default `follow_wrapped=True` so this reads the original run(cfg, ctx).
    """
    sig = inspect.signature(run)
    if "cfg" not in sig.parameters:
        raise TypeError(f"{getattr(run, '__name__', run)!r} must accept a 'cfg' parameter")
    ann = sig.parameters["cfg"].annotation
    if ann is inspect.Parameter.empty:
        raise TypeError(
            "the run's `cfg` parameter needs a dataclass type annotation so "
            "runkit knows what to build from CLI args")
    return ann


def _help_text(run):
    name = getattr(run, "_runkit_name", getattr(run, "__name__", "run"))
    cfg_cls = _cfg_type(run)
    lines = [
        "usage: <experiment.py> [config.yaml] [key=value ...] [--flag ...]",
        "",
        f"experiment: {name}   (cfg: {cfg_cls.__name__})",
        "",
        "config overrides (bare key=value):",
    ]
    for fld in dataclasses.fields(cfg_cls):
        lines.append(f"  {fld.name}={fld.default!r}")
    lines += ["", "staging flags (--flag):"]
    for flag in sorted(_staging_flags(run)):
        rendered = f"--{flag.replace('_', '-')}"
        if flag == "force":
            rendered = f"-f, {rendered}   (with --out: replace the dir if it exists)"
        lines.append(f"  {rendered}")
    lines += ["  --config=PATH   (cwd:/exp:/SCHEME:/abs)"]
    return "\n".join(lines)
