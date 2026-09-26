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
from .runs import load_run, select_run
from .checkpoints import Checkpoint, load_checkpoints
from .metrics import load_metrics
from .autocli import main
from .config import random_seed

__all__ = ["Experiment", "experiment", "Run", "RunContext", "init_run",
           "load_run", "select_run", "Checkpoint", "load_checkpoints",
           "load_metrics", "random_seed", "main"]
