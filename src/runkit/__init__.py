"""runkit — lightweight, reproducible experiment runs.

Decorate `run(cfg, ctx)` with @experiment(name=...); the decorator creates a
fresh run dir (`ctx.dir`), freezes the resolved config into it, and dumps a
non-`None` return value. Run via `python experiment.py ...`, `python -m pkg`,
or `runkit run experiment.py ...` — all share `runkit.autocli.main`.

Two disjoint namespaces: config (the "what") via `key=value`; staging (the
"how/where") via `--flags` (`--tag`, `--root`). See design.md.
"""
from .exp import experiment, Run, RunContext, init_run
from .autocli import main

__all__ = ["experiment", "Run", "RunContext", "init_run", "main"]
