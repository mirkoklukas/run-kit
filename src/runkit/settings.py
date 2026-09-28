"""`experiment.toml`: per-folder settings, read before anything is imported.

The file sits in (or above) an experiment's folder; the nearest one wins, found
by walking up the way tools find `pyproject.toml`:

    # lab/rl_env/experiment.toml
    [env]
    root = "ctk:runs"                        # where run dirs go
    extras = ["mjx"]                         # uv extras to launch with
    vars = { XLA_PYTHON_CLIENT_PREALLOCATE = "false" }   # environment variables

    [env.test_policy]                        # keyed by the experiment file's stem
    extras = ["mjx", "sb3"]                  # lists replace: the full set
    vars = { MUJOCO_GL = "egl" }             # tables merge, this one winning per key

A path is relative to the folder holding `experiment.toml`, or carries one of
runkit's scheme prefixes (`ctk:` -> `$RUNKIT_PATH_CTK`, `exp:` -> that folder).
`extras` and `vars` are applied by `runkit.launch`, before the import.
"""
import pathlib

try:
    import tomllib
except ModuleNotFoundError:          # python < 3.11
    import tomli as tomllib

from .utils import resolve_config_path

FILENAME = "experiment.toml"
DEFAULT_ROOT = "runs"                # relative to cwd, when nothing says otherwise
KEYS = ("root", "extras", "vars", "follow", "project")   # [env] keys; `project`: not read yet


def find_toml(start):
    """The nearest `experiment.toml` at or above `start` (a file or a dir)."""
    start = pathlib.Path(start).resolve()
    if not start.is_dir():
        start = start.parent
    for d in (start, *start.parents):
        if (d / FILENAME).is_file():
            return d / FILENAME
    return None


def read_env(start, stem=None):
    """An experiment's [env] settings: `{"toml", "root", "extras", "vars"}`.

    `[env]` holds the folder's defaults; `[env.<stem>]` is applied on top for
    the experiment file of that stem. Lists (`extras`) replace -- the override
    states the full set; tables (`vars`) merge, the override winning per key.
    `root` is left raw (see `resolve_root`); `vars` values become strings, a
    boolean as `true` / `false`. No file: no settings.
    """
    toml = find_toml(start)
    out = {"toml": toml, "root": None, "extras": [], "vars": {}, "follow": None}
    if toml is None:
        return out
    try:
        data = tomllib.loads(toml.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{toml}: {e}") from None
    env = data.get("env", {})
    if stem in KEYS:
        raise ValueError(f"{toml}: an experiment named {stem!r} cannot have an "
                         f"[env.{stem}] override -- {stem!r} is an [env] key; rename the file")
    base = {k: v for k, v in env.items() if k in KEYS}
    override = env.get(stem, {}) if stem else {}
    # in [env], a table is some experiment's override; anything else must be a key
    loose = {k: v for k, v in env.items() if not isinstance(v, dict) or k == "vars"}
    for where, table in (("[env]", loose), (f"[env.{stem}]", override)):
        unknown = [k for k in table if k not in KEYS]
        if unknown:
            from . import ui
            ui.warn(f"{toml} {where}: unknown key(s) {unknown} (known: {', '.join(KEYS)})")
        if "project" in table:
            from . import ui
            ui.warn(f"{toml} {where}: `project` is not read yet; the uv project is the "
                    f"nearest pyproject.toml above the experiment")
    out["root"] = override.get("root", base.get("root"))
    out["follow"] = override.get("follow", base.get("follow"))
    extras = override["extras"] if "extras" in override else base.get("extras", [])
    if not isinstance(extras, list) or not all(isinstance(e, str) for e in extras):
        raise ValueError(f"{toml}: `extras` must be a list of names, got {extras!r}")
    out["extras"] = list(extras)
    merged = {**base.get("vars", {}), **override.get("vars", {})}
    out["vars"] = {str(k): _env_value(v) for k, v in merged.items()}
    return out


def _env_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (str, int, float)):
        return str(v)
    raise ValueError(f"experiment.toml: a `vars` value must be a string, number or "
                     f"boolean, got {v!r}")


def resolve_follow(start, stem=None, explicit=None):
    """The stream a run prints as it goes: `explicit` (`--follow`) >
    `[env.<stem>]` > `[env]` > `run`. A bare `--follow` is `run`; `none` (or
    false) prints nothing -- for an experiment that prints its own progress."""
    value = explicit if explicit is not None else read_env(start, stem)["follow"]
    if value is None or value is True:
        return "run"
    if value is False or str(value).lower() in ("none", "off", "false", ""):
        return None
    return str(value)


def resolve_root(start, stem=None, explicit=None):
    """Where run dirs go: `explicit` (`--root`) > `[env.<stem>]` > `[env]` > ./runs.

    `start` is where the search for `experiment.toml` begins -- the experiment's
    file, or any folder. `stem` names the experiment file, for its override.
    """
    if explicit is not None:
        return pathlib.Path(explicit)
    env = read_env(start, stem)
    if env["root"] is None:
        return pathlib.Path(DEFAULT_ROOT)
    toml = env["toml"]
    path = pathlib.Path(resolve_config_path(env["root"], toml.parent))
    return path if path.is_absolute() else toml.parent / path
