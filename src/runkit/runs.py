"""Finding and opening run dirs that already exist -- the read side of `init_run`.

`select_run` turns what someone typed (nothing, a path, a hex prefix) into one
run dir of one experiment; `load_run` rebuilds the `Run` it holds.
"""
import json
import pathlib

import numpy as np
import yaml

from .config import build_cfg
from .exp import Run, RunContext
from .utils import load_yaml


# A run dir's name is `{date}_{time}_{hex8}[_{tag}]`, fixed-width up to the tag:
# `YYYY-MM-DD_HH-MM-SS_` is 20 characters, then the 8 hex.
_HEX = slice(20, 28)


def dir_hex(run_dir):
    """The `{hex8}` in a run dir's name -- the hex of its id."""
    return pathlib.Path(run_dir).name[_HEX]


class RunNotFound(ValueError):
    """No run dir matches the selection. Its message is meant for the user."""


def run_dirs(root, name):
    """The run dirs of experiment `name` under `root`, oldest first.

    Names lead with `{date}_{time}`, so the name orders runs to the second.
    Within a second (a sweep) the rest of the name is a random hex, so ties
    break on when `run_context.yaml` was written -- once, at creation.
    `latest` (a symlink) and dot-entries are skipped.
    """
    d = pathlib.Path(root) / name
    if not d.is_dir():
        return []
    dirs = [p for p in d.iterdir()
            if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")]
    return sorted(dirs, key=_start_key)


def _start_key(run_dir):
    stamp = run_dir.name[:len("YYYY-MM-DD_HH-MM-SS")]
    try:
        created = (run_dir / "run_context.yaml").stat().st_mtime_ns
    except OSError:
        created = 0
    return stamp, created


def _status(run_dir):
    try:
        return yaml.safe_load((run_dir / "status.yaml").read_text()).get("status")
    except Exception:                                        # noqa: BLE001
        return None


def select_run(root, name, which=None, *, require_ok=False):
    """Pick one run dir of experiment `name`.

    which=None   -> the latest run under `{root}/{name}`; with `require_ok`,
                    the latest whose `status.yaml` says `ok`
    a directory  -> that run dir (`runs/baseline/latest` works too)
    anything else-> a hex prefix, matched against the `{hex8}` in every run
                    dir's name, right after the time (the same hex as in its id)

    Raises `RunNotFound` if nothing matches, a prefix matches several, or the
    dir is a run of a different experiment.
    """
    base = pathlib.Path(root) / name
    if which is None:
        dirs = run_dirs(root, name)
        if require_ok:
            dirs = [d for d in dirs if _status(d) == "ok"]
        if not dirs:
            kind = "finished (ok) runs" if require_ok else "runs"
            raise RunNotFound(f"no {kind} of {name!r} under {base}")
        run_dir = dirs[-1]
    elif pathlib.Path(which).is_dir():
        run_dir = pathlib.Path(which).resolve()
    else:
        which = str(which)
        matches = [d for d in run_dirs(root, name)
                   if dir_hex(d).startswith(which)]
        if not matches:
            raise RunNotFound(f"no run dir {which!r}, and no run of {name!r} "
                              f"under {base} whose id starts with it")
        if len(matches) > 1:
            listing = "".join(f"\n  {d.name}" for d in matches)
            raise RunNotFound(f"{which!r} matches {len(matches)} runs of {name!r}; "
                              f"use a longer prefix:{listing}")
        run_dir = matches[0]

    try:
        owner = load_yaml(run_dir / "run_context.yaml").get("name")
    except ValueError:
        raise RunNotFound(f"{run_dir} is not a run dir (no run_context.yaml)") from None
    if owner != name:
        raise RunNotFound(f"{run_dir} is a run of {owner!r}, not {name!r}")
    return run_dir


def load_run(run_dir, cfg_cls):
    """Rebuild the `Run` a run dir holds: the config thawed into `cfg_cls`, the
    `RunContext`, and the return value if one was dumped.

    Whatever the run's status -- a run that failed or is still going loads too;
    `status.yaml` is there to check.
    """
    run_dir = pathlib.Path(run_dir).resolve()
    rc = load_yaml(run_dir / "run_context.yaml")
    ctx = RunContext(dir=run_dir, id=rc["id"], name=rc.get("name"))
    cfg = build_cfg(cfg_cls, load_yaml(run_dir / "config.yaml"))
    retval = None
    if (run_dir / "retval.json").is_file():
        retval = json.loads((run_dir / "retval.json").read_text())
    elif (run_dir / "retval.npy").is_file():
        retval = np.load(run_dir / "retval.npy")
    return Run(config=cfg, context=ctx, retval=retval)
