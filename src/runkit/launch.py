"""Starting the process an experiment needs, before it is imported.

`experiment.toml` can name uv extras and environment variables. Extras have to
be in place *before* the experiment is imported -- without them, the import is
what fails -- so they cannot come from the experiment itself. `runkit <verb>
<experiment> ...` reads them first:

- with `extras`, it re-executes itself as
      uv run --project <dir> --extra A --extra B python -m runkit <the same argv>
  with `vars` in the environment. `<dir>` is the nearest pyproject.toml at or
  above the experiment. A guard variable makes this happen once.
- with only `vars`, no relaunch is needed: they are set in this process's
  environment before the import.

What the process was launched with goes to `RUNKIT_LAUNCH` (json), which
`init_run` records in `meta.yaml` as `launch`. `python experiment.py` cannot
change its own environment, so it is left as it is.
"""
import importlib.util
import json
import os
import pathlib
import shutil
import sys

from . import ui
from .settings import read_env

GUARD = "RUNKIT_RELAUNCHED"        # set in the relaunched process: do not relaunch again
LAUNCH = "RUNKIT_LAUNCH"           # json: extras / vars / project, for meta.yaml


def experiment_file(target, cwd=None):
    """The file an experiment target names, found without importing it.

    A path (`lab/rl_env/test_policy.py`) as given; a dotted module
    (`lab.rl_env.test_policy`) looked up under the current folder, as `runkit`
    imports it -- `.py`, or a package's `__init__.py` -- and otherwise where
    python would import it from (an installed package, run from any folder).
    That last lookup imports the module's parent packages, never the module.
    None if not found (the import will then say why).
    """
    path = pathlib.Path(target)
    if target.endswith(".py") or path.is_file():
        return path.resolve() if path.is_file() else None
    base = pathlib.Path(cwd or pathlib.Path.cwd()).joinpath(*target.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate.resolve()
    try:
        spec = importlib.util.find_spec(target)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or spec.origin in ("built-in", "frozen"):
        return None
    return pathlib.Path(spec.origin).resolve()


def _stem(file):
    return file.parent.name if file.name == "__init__.py" else file.stem


def uv_project(file):
    """The nearest folder at or above `file` with a pyproject.toml, or None."""
    for d in file.parents:
        if (d / "pyproject.toml").is_file():
            return d
    return None


def prepare(target, argv, environ=None, execvpe=None):
    """Apply experiment.toml's `vars`, and relaunch under uv for its `extras`.

    Called by `runkit` before it imports `target`; `argv` is runkit's own argv,
    repeated in the relaunch. Returns if nothing needs to change, or after
    setting `vars` in `environ`; with extras, does not return (the process is
    replaced). `environ` / `execvpe` are for tests.
    """
    environ = os.environ if environ is None else environ
    execvpe = os.execvpe if execvpe is None else execvpe
    if environ.get(GUARD):
        return                              # the relaunched process: already set up
    file = experiment_file(target)
    if file is None:
        return
    try:
        env = read_env(file, _stem(file))
    except ValueError as e:
        sys.exit(str(e))
    extras, vars_ = env["extras"], env["vars"]
    if not extras:
        if vars_:
            environ.update(vars_)
            environ[LAUNCH] = json.dumps({"extras": [], "vars": vars_, "project": None})
        return

    uv = shutil.which("uv")
    if uv is None:
        sys.exit(f"{env['toml']} asks for uv extras {extras}, but uv is not on PATH. "
                 f"Install uv, or start the experiment inside an environment that has "
                 f"them: python -m <module> ...")
    project = uv_project(file)
    if project is None:
        sys.exit(f"{env['toml']} asks for uv extras {extras}, but there is no "
                 f"pyproject.toml at or above {file} to take them from")
    launch = {"extras": extras, "vars": vars_, "project": str(project)}
    cmd = [uv, "run", "--project", str(project)]
    for e in extras:
        cmd += ["--extra", e]
    cmd += ["python", "-m", "runkit", *argv]
    ui.line(f"[dim]↻ runkit: relaunching under uv with extras {', '.join(extras)}"
            + (f" and {len(vars_)} env var(s)" if vars_ else "") + "[/dim]")
    execvpe(uv, cmd, {**environ, **vars_, GUARD: "1", LAUNCH: json.dumps(launch)})
