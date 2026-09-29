"""runkit — lightweight, reproducible experiment runs.

An `Experiment` holds an experiment's identity and the functions that play its
roles: `@exp.run` creates a fresh run dir (`ctx.dir`), freezes the resolved
config into it and dumps a non-`None` return value; `@exp.eval` and `@exp.viz`
open an existing one. `@experiment(name=...)` is the run-only shorthand. Call
via `python experiment.py [verb] ...`, `python -m pkg [verb] ...`, or
`runkit <verb> experiment.py ...` — all share one dispatcher.

Two disjoint namespaces: config (the "what") via `key=value`; staging (the
"how/where") via `--flags` (`--tag`, `--root`). See design.md.
"""
from .exp import Experiment, experiment, Run, RunContext, init_run
from .runs import load_run, select_checkpoint, select_run
from .checkpoints import Checkpoint, load_checkpoints
from .metrics import compile_metrics, load_metrics, plot_metrics, record
from .autocli import main
from .config import random_seed
from .utils import load_config, save_config

__all__ = ["Experiment", "experiment", "Run", "RunContext", "init_run",
           "load_run", "select_run", "select_checkpoint", "Checkpoint",
           "load_checkpoints", "load_metrics", "compile_metrics", "plot_metrics",
           "record", "random_seed", "save_config", "load_config", "main"]
