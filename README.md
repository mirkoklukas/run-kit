# runkit

Lightweight, reproducible experiment runs. Decorate a `run(cfg, ctx)` with
`@experiment(name=...)`; the decorator creates a fresh, uniquely-named run dir,
freezes the resolved config into it, and dumps a non-`None` return value.

```python
# experiment.py
from dataclasses import dataclass
from runkit import experiment, RunContext, main


@dataclass
class Config:
    seed: int = 1
    lr: float = 3e-4


@experiment(name="baseline")
def run(cfg: Config, ctx: RunContext):
    (ctx.out / "result.txt").write_text(f"seed={cfg.seed}")   # ctx.out = the run dir
    return {"score": cfg.seed * 0.1}                          # -> results/retval.json


if __name__ == "__main__":
    main(run)
```

Run it (both forms share the same core):

```bash
python experiment.py [config.yaml] [key=value ...] [--flag ...]
runkit run experiment.py [config.yaml] [key=value ...] [--flag ...]
```

- bare `key=value` → overrides into **config** (the "what")
- `--flag=value` → sets **staging** (the "how/where"): `--tag`, `--runs-dir`, `--out`
- config paths resolve as `cwd:NAME` (default) / `exp:NAME` (next to the experiment) / `/abs`

Run dirs land at `{runs_dir}/{name}[_{tag}]_{date}_{time}_{hex8}/`, with the run
`id = {name}_{hex8}` (greppable in `results`/`config.yaml`). See `src/runkit/re-design.md`.

## Layout

- `src/runkit/` — `exp` (decorator), `autocli` (the CLI core), `utils`, `cli`, `config`.
- `src/runkit/old/` — the pre-redesign implementation (provenance, dirty-gate,
  context yaml), archived for reference and not wired into the API.

## Status

Early. Provenance and the dirty-git gate from the old implementation are not yet
ported to the lightweight runner. See `src/runkit/re-design.md` for the plan.
