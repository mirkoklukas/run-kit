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


def _flat(d, prefix=""):
    """{"optim": {"lr": 1}} -> {"optim.lr": 1}; an empty dict stays a leaf."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _defaults(cls):
    """A dataclass's field defaults as a plain dict; fields without one are left out."""
    out = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            v = f.default
        elif f.default_factory is not dataclasses.MISSING:
            v = f.default_factory()
        else:
            continue
        out[f.name] = (dataclasses.asdict(v)
                       if dataclasses.is_dataclass(v) and not isinstance(v, type) else v)
    return out


def config_changes(cfg):
    """(the fields of `cfg` that differ from its defaults, how many fields in all).

    Flattened to dotted keys (`optim.lr`), as they would be set on the command
    line. A field with no default always counts as changed -- somebody set it.
    """
    if not (dataclasses.is_dataclass(cfg) and not isinstance(cfg, type)):
        flat = _flat(serialize_cfg(cfg))
        return flat, len(flat)
    values = _flat(dataclasses.asdict(cfg))
    defaults = _flat(_defaults(type(cfg)))
    changed = {k: v for k, v in values.items() if k not in defaults or defaults[k] != v}
    return changed, len(values)


def plain(v):
    """What yaml / json can write: numpy scalars and arrays become python values,
    anything else unknown its repr. Never raises -- a value it cannot represent
    must not cost the checkpoint or the metrics row it is part of."""
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return repr(v)


def point_latest(target):
    """Point `latest`, beside `target`, at it: `{folder}/latest -> {target name}`.

    A shortcut for people (`cd runs/baseline/latest`), not something runkit reads.
    So it is best-effort -- a failure (no symlink rights, a real dir in the way)
    warns and moves on. The target is relative, so the link survives moving the
    folder, and it is swapped in with `os.replace`, so concurrent updates never
    leave it half-written.
    """
    target = pathlib.Path(target)
    link = target.parent / "latest"
    tmp = target.parent / f".latest.{target.name}"
    try:
        tmp.symlink_to(target.name, target_is_directory=True)
        os.replace(tmp, link)
    except Exception as e:                                   # noqa: BLE001
        tmp.unlink(missing_ok=True)
        from . import ui
        ui.warn(f"could not point {link} at {target.name}: {e}")


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
