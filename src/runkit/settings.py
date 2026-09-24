"""`experiment.toml`: per-folder settings, read before anything is imported.

Only `root` is read so far. The file sits in (or above) an experiment's folder;
the nearest one wins, found by walking up the way tools find `pyproject.toml`:

    # lab/rl_env/experiment.toml
    [env]
    root = "ctk:runs"               # the folder's default

    [env.test_policy]               # keyed by the experiment file's stem
    root = "../../runs/policy"      # this experiment only

A path is relative to the folder holding `experiment.toml`, or carries one of
runkit's scheme prefixes (`ctk:` -> `$RUNKIT_PATH_CTK`, `exp:` -> that folder).
"""
import pathlib

try:
    import tomllib
except ModuleNotFoundError:          # python < 3.11
    import tomli as tomllib

from .utils import resolve_config_path

FILENAME = "experiment.toml"
DEFAULT_ROOT = "runs"                # relative to cwd, when nothing says otherwise


def find_toml(start):
    """The nearest `experiment.toml` at or above `start` (a file or a dir)."""
    start = pathlib.Path(start).resolve()
    if not start.is_dir():
        start = start.parent
    for d in (start, *start.parents):
        if (d / FILENAME).is_file():
            return d / FILENAME
    return None


def resolve_root(start, stem=None, explicit=None):
    """Where run dirs go: `explicit` (`--root`) > `[env.<stem>]` > `[env]` > ./runs.

    `start` is where the search for `experiment.toml` begins -- the experiment's
    file, or any folder. `stem` names the experiment file, for its override.
    """
    if explicit is not None:
        return pathlib.Path(explicit)
    toml = find_toml(start)
    raw = _root_setting(toml, stem) if toml else None
    if raw is None:
        return pathlib.Path(DEFAULT_ROOT)
    path = pathlib.Path(resolve_config_path(raw, toml.parent))
    return path if path.is_absolute() else toml.parent / path


def _root_setting(toml, stem):
    try:
        data = tomllib.loads(toml.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{toml}: {e}") from None
    env = data.get("env", {})
    override = env.get(stem) if stem else None
    if isinstance(override, dict) and "root" in override:
        return override["root"]
    return env.get("root")
