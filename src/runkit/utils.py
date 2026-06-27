"""Small pure helpers for the lightweight runner (exp.py / autocli.py).

No decorator/CLI logic here -- just path resolution, yaml IO, cfg serialization,
and the best-effort return-value dump.
"""
import dataclasses
import json
import pathlib
import sys

import numpy as np
import yaml

# Config-path schemes: the optional prefix that picks what a *relative* path is
# resolved against. `cwd:` (or no prefix) -> cwd; `exp:` -> the experiment's dir.
_PATH_SCHEMES = ("cwd", "exp")


def resolve_config_path(raw, exp_dir):
    """Expand an optional `cwd:` / `exp:` scheme prefix on a config path.

    cwd:NAME / bare NAME -> left relative to cwd (the default).
    exp:NAME             -> joined onto the experiment dir (`exp_dir`).
    Absolute paths, and any token whose prefix isn't a known scheme (so a stray
    ':' in a filename), are returned untouched.
    """
    scheme, sep, rest = raw.partition(":")
    if not sep or scheme not in _PATH_SCHEMES:
        return raw
    if scheme == "exp":
        if exp_dir is None:
            raise ValueError(
                "config path uses the 'exp:' base, but the experiment dir is "
                "unknown (is `run` defined in the experiment module?)")
        return str(pathlib.Path(exp_dir) / rest)
    return rest      # 'cwd:' -> strip the prefix, leave it cwd-relative


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


def dump_retval(results_dir, value):
    """Best-effort dump of a run's return value into `results_dir`.

    `.npy` for a numpy array, otherwise `.json` (with `default=str`, so almost
    anything serializes). Never raises -- a value we can't write is skipped with
    a warning, so a bad return can't fail an otherwise-good run. Returns the path
    written, or None.

    TODO: broaden the type->format dispatch (e.g. `.npz` for a dict of arrays).
    """
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(value, np.ndarray):
            out = results_dir / "retval.npy"
            np.save(out, value)
        else:
            out = results_dir / "retval.json"
            out.write_text(json.dumps(value, indent=2, default=str))
        return out
    except Exception as e:                                   # noqa: BLE001 (best-effort)
        print(f"[runkit] could not serialize return value "
              f"({type(value).__name__}): {e}", file=sys.stderr)
        return None
