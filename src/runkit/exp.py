"""The lightweight `@experiment` decorator and the run dir it builds.

A decorated function is `run(cfg, ctx)`. The decorator wraps it so that, before
the body runs, a fresh run dir is created and the resolved config frozen into it;
after, a non-`None` return value is dumped into the run dir. Staging is driven by
keyword args (the CLI flags): `tag`, `runs_dir`, `out`.

The decorator carries only the experiment's *identity* (`name`); everything that
stages an attempt is a flag. See design.md.
"""
import dataclasses
import datetime
import functools
import pathlib
import sys
import uuid

import yaml

from .utils import dump_retval, serialize_cfg


@dataclasses.dataclass
class RunContext:
    out: pathlib.Path   # the run dir; everything the experiment writes goes here
    id: str             # stable unique run id, e.g. "baseline_a3f9c1e7" (for search)


def _resolve_out(runs_dir, name, tag, hex8, out_override):
    """Build the run dir path -- the one place the naming scheme lives.

    default: {runs_dir}/{name}[_{tag}]_{date}_{time}_{hex8}/
    --out:   that exact dir; on collision, append _{timestamp}.
    """
    if out_override is not None:
        p = pathlib.Path(out_override).resolve()
        if p.exists():
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            p = p.parent / f"{p.name}_{ts}"
        return p
    now = datetime.datetime.now()
    parts = [name, *([tag] if tag else []), f"{now:%Y-%m-%d}", f"{now:%H-%M-%S}", hex8]
    return pathlib.Path(runs_dir).resolve() / "_".join(parts)


def init_run(cfg, *, name, tag, runs_dir, out):
    """Create the run dir, dump the resolved config, return a RunContext.

    The run id and the dir's `{hex8}` share one uuid, so the dir is
    self-identifying (`id = {name}_{hex8}`).
    """
    hex8 = uuid.uuid4().hex[:8]
    uid = f"{name}_{hex8}"
    out_path = _resolve_out(runs_dir, name, tag, hex8, out)
    out_path.mkdir(parents=True, exist_ok=False)
    (out_path / "results").mkdir()
    (out_path / "config.yaml").write_text(
        yaml.safe_dump(serialize_cfg(cfg), sort_keys=False))
    return RunContext(out=out_path, id=uid)


def _announce(name, cfg, ctx):
    """Print a one-time start banner: which run, with what config, where.

    Goes to stderr so it never mixes into data an experiment writes to stdout.
    The resolved config is echoed inline; the same values are frozen at
    `{out}/config.yaml`.
    """
    cfg_inline = yaml.safe_dump(
        serialize_cfg(cfg), default_flow_style=True, sort_keys=False).strip()
    for line in (
        f"[runkit] starting {name!r}  (id={ctx.id})",
        f"[runkit] out:    {ctx.out}",
        f"[runkit] config: {ctx.out / 'config.yaml'}  {cfg_inline}",
    ):
        print(line, file=sys.stderr)


def experiment(*, name):
    """Mark `run(cfg, ctx)` as an experiment entry point.

    `name` is the experiment's identity (required, no CLI override). The wrapper
    accepts the staging flags as keyword args -- `tag`, `runs_dir`, `out` -- which
    `autocli.main` forwards from the CLI; their names *are* the allowed flags.
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(cfg, *, tag=None, runs_dir="runs", out=None):
            ctx = init_run(cfg, name=name, tag=tag, runs_dir=runs_dir, out=out)
            _announce(name, cfg, ctx)
            result = f(cfg, ctx)
            if result is not None:
                dump_retval(ctx.out / "results", result)
            return result
        wrapper._runkit_name = name          # identity, for introspection
        # where the experiment lives, for resolving `exp:`-prefixed config paths
        wrapper._runkit_dir = pathlib.Path(f.__code__.co_filename).resolve().parent
        return wrapper
    return decorator
