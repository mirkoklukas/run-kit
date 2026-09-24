"""Small pure helpers for the lightweight runner (exp.py / autocli.py).

No decorator/CLI logic here -- just path resolution, yaml IO, cfg serialization,
and the best-effort return-value dump.
"""
import dataclasses
import json
import os
import pathlib
import re

import numpy as np
import yaml

# Config-path schemes: the optional prefix that picks what a *relative* path is
# resolved against. Two are built in -- `cwd:` (or no prefix) -> cwd, `exp:` ->
# the experiment's dir -- and any other scheme is user-defined via an env var
# `RUNKIT_PATH_<SCHEME>` (uppercased), whose value is the base dir. So exporting
# `RUNKIT_PATH_CTK=~/control-kit` makes `ctk:configs/x.yaml` resolve there.
_BUILTIN_SCHEMES = ("cwd", "exp")
_ENV_PREFIX = "RUNKIT_PATH_"

# A token that *looks* like a scheme use we should have recognized: a short,
# all-alphanumeric prefix before the first ':' (so we can warn on a typo'd
# scheme without mistaking `C:\...` or `https://...` for one).
_SCHEME_LIKE = re.compile(r"^[A-Za-z0-9_]+$")


def _env_base(scheme):
    """Base dir for a user-defined scheme, or None if `RUNKIT_PATH_<SCHEME>` unset."""
    raw = os.environ.get(_ENV_PREFIX + scheme.upper())
    return os.path.expanduser(raw) if raw else None


def resolve_config_path(raw, exp_dir):
    """Expand an optional `SCHEME:` prefix on a config path.

    cwd:NAME / bare NAME -> left relative to cwd (the default).
    exp:NAME             -> joined onto the experiment dir (`exp_dir`).
    ctk:NAME (any other) -> joined onto `RUNKIT_PATH_CTK` if that env var is set.
    Absolute paths, and any token whose prefix isn't a known/defined scheme (so a
    stray ':' in a filename), are returned untouched -- though a prefix that looks
    like a scheme yet has no base gets a warning, to catch typos.
    """
    scheme, sep, rest = raw.partition(":")
    if not sep:
        return raw
    if scheme == "cwd":
        return rest      # strip the prefix, leave it cwd-relative
    if scheme == "exp":
        if exp_dir is None:
            raise ValueError(
                "config path uses the 'exp:' base, but the experiment dir is "
                "unknown (is `run` defined in the experiment module?)")
        return str(pathlib.Path(exp_dir) / rest)
    base = _env_base(scheme)
    if base is not None:
        return str(pathlib.Path(base) / rest)
    if _SCHEME_LIKE.match(scheme):
        from . import ui
        ui.warn(
            f"path '{raw}' looks like it uses a '{scheme}:' base, but no base is "
            f"defined for it (set {_ENV_PREFIX}{scheme.upper()}); treating it as a "
            "literal path")
    return raw


def load_yaml(path):
    """Load a yaml file that must be a mapping (or empty); raise otherwise."""
    p = pathlib.Path(path)
    if not p.is_file():
        raise ValueError(f"config file not found: {p}")
    data = yaml.safe_load(p.read_text())
    if data is not None and not isinstance(data, dict):
        raise ValueError(f"config file {p} must be a mapping, got {type(data).__name__}")
    return data or {}


def serialize_cfg(cfg):
    """Convert a cfg into a plain dict for the frozen config.yaml."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    return {"repr": repr(cfg)}


def dump_retval(run_dir, value):
    """Best-effort dump of a run's return value into `run_dir`.

    `.npy` for a numpy array, otherwise `.json` (with `default=str`, so almost
    anything serializes). Never raises -- a value we can't write is skipped with
    a warning, so a bad return can't fail an otherwise-good run. Returns the path
    written, or None.

    TODO: broaden the type->format dispatch (e.g. `.npz` for a dict of arrays).
    """
    run_dir = pathlib.Path(run_dir)
    try:
        if isinstance(value, np.ndarray):
            out = run_dir / "retval.npy"
            np.save(out, value)
        else:
            out = run_dir / "retval.json"
            out.write_text(json.dumps(value, indent=2, default=str))
        return out
    except Exception as e:                                   # noqa: BLE001 (best-effort)
        from . import ui
        ui.warn(f"could not serialize return value ({type(value).__name__}): {e}")
        return None
