"""runkit — lightweight, reproducible experiment runs.

Decorate `run(cfg, ctx)` with @experiment(name=...); the decorator creates a
fresh run dir (`ctx.out`), freezes the resolved config into it, and dumps a
non-`None` return value. Run via `python experiment.py ...`, `python -m pkg`,
or `runkit run experiment.py ...` — all share `runkit.autocli.main`.

Two disjoint namespaces: config (the "what") via `key=value`; staging (the
"how/where") via `--flags` (`--tag`, `--runs-dir`, `--out`). See design.md.

The pre-redesign implementation (provenance, dirty-gate, context yaml) is
archived under `runkit.old`, not wired into this API.
"""
from .exp import experiment, RunContext, init_run
from .autocli import main

__all__ = ["experiment", "RunContext", "init_run", "main"]
