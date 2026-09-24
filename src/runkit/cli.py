"""The `runkit` console script.

Usage:
    runkit <verb> [runkit options] <experiment> [args ...]
    runkit root [FOLDER]

verbs: run, eval, viz (as registered on the experiment), root, latest

`<experiment>` is a file (`experiment.py`) or a dotted module
(`lab.rl_env.test_policy`). `args` are what `python experiment.py <verb> ...`
takes after the verb. The slot between the verb and the experiment is for
runkit's own options (none yet).

`root` and `latest` print a path, for `cd "$(runkit latest experiment.py)"`:
the experiment's runs folder ({root}/{name}) and its latest run dir. Without an
experiment, `runkit root [FOLDER]` prints the root itself, resolved from FOLDER
(default: the current one): the nearest `experiment.toml`'s `[env] root`, else
`./runs`.
"""
import importlib
import importlib.util
import pathlib
import sys

from . import ui
from .autocli import PATH_VERBS, VERBS, dispatch
from .exp import SCRIPT_MODULE, Experiment
from .settings import resolve_root

HELP = __doc__[__doc__.index("Usage:"):].strip()


def app():
    """Console-script entry point. Returns None: whatever the verb returned (a
    `Run`, say) must not become the process's exit status."""
    main()


def main(argv=None):
    """`runkit <verb> [options] <experiment> ...` -> the verb's return value."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"runkit — lightweight, reproducible experiment runs.\n\n{HELP}")
        return
    verb, rest = argv[0], argv[1:]
    if verb not in VERBS + PATH_VERBS:
        hint = ""
        if verb.endswith(".py") or "." in verb:      # the old order: runkit exp.py viz
            v = rest[0] if rest and rest[0] in VERBS + PATH_VERBS else "run"
            hint = f" -- the verb comes first: runkit {v} {verb} ..."
        sys.exit(f"unknown verb {verb!r}{hint}\n\n{HELP}")

    # runkit's own options sit between the verb and the experiment
    while rest and rest[0].startswith("-"):
        opt = rest.pop(0)
        if opt in ("-h", "--help"):
            print(HELP)
            return
        sys.exit(f"unknown runkit option {opt!r} for {verb} "
                 f"(experiment arguments go after the experiment)")

    if verb == "root" and (not rest or pathlib.Path(rest[0]).is_dir()):
        return root_cmd(rest)
    if not rest:
        sys.exit(f"usage: runkit {verb} <experiment> ...")
    target, args = rest[0], rest[1:]
    return dispatch(_find_experiment(_import(target), target), [verb, *args])


def root_cmd(argv):
    """`runkit root [FOLDER]`: print the root resolved from FOLDER.

    Printing is all it can do: a process cannot change its parent shell's
    directory. Only the path goes to stdout, so `cd "$(runkit root)"` is clean.
    """
    if len(argv) > 1:
        sys.exit("usage: runkit root [FOLDER | <experiment>]")
    start = pathlib.Path(argv[0]) if argv else pathlib.Path.cwd()
    try:
        root = resolve_root(start).resolve()
    except ValueError as e:
        sys.exit(str(e))
    if not root.is_dir():
        ui.warn(f"{root} does not exist yet (no runs made there)")
    print(root)
    return root


def _import(target):
    """Import a file-path or dotted-module target the way `python` would run it."""
    path = pathlib.Path(target)
    if target.endswith(".py") or path.is_file():
        if not path.is_file():
            sys.exit(f"experiment file not found: {target}")
        # like `python file.py`: the file's own folder is importable
        sys.path.insert(0, str(path.resolve().parent))
        spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[SCRIPT_MODULE] = mod
        spec.loader.exec_module(mod)
        return mod
    # like `python -m pkg.mod`: importable from the current directory
    sys.path.insert(0, str(pathlib.Path.cwd()))
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as e:
        sys.exit(f"cannot import {target!r} (not a file, and not a module: {e})")


def _find_experiment(mod, target):
    """The one `Experiment` in `mod` -- an `Experiment(...)` object, or the one
    behind an `@experiment`-decorated function. One experiment per file."""
    found = {}
    for value in vars(mod).values():
        exp = value if isinstance(value, Experiment) else getattr(
            value, "_runkit_experiment", None)
        if isinstance(exp, Experiment):
            found[id(exp)] = exp
    if not found:
        sys.exit(f"{target}: no experiment found (an `Experiment(...)` or an "
                 f"@experiment function at module level)")
    if len(found) > 1:
        names = sorted(e.name for e in found.values())
        sys.exit(f"{target}: {len(found)} experiments ({', '.join(names)}); "
                 f"one experiment per file")
    return next(iter(found.values()))
