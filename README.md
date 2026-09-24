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
    (ctx.out / "result.txt").write_text(f"seed={cfg.seed}")   # ctx.out = {run dir}/out
    return {"score": cfg.seed * 0.1}                          # -> {run dir}/retval.json


if __name__ == "__main__":
    main(run)
```

Run it (both forms share the same core):

```bash
python experiment.py [config.yaml] [key=value ...] [--flag ...]
runkit run experiment.py [config.yaml] [key=value ...] [--flag ...]
```

- bare `key=value` → overrides into **config** (the "what")
- `--flag=value` → sets **staging** (the "how/where"): `--tag`, `--root`
- config paths resolve as `cwd:NAME` (default) / `exp:NAME` (next to the experiment) / `/abs`,
  plus user-defined `SCHEME:NAME` bases via `$RUNKIT_PATH_<SCHEME>` (e.g. `ctk:x.yaml` → `$RUNKIT_PATH_CTK/x.yaml`)

Run dirs land at `{root}/{name}/{date}_{time}[_{tag}]_{hex8}/` (plus a
`{root}/{name}/latest` link to the newest one), with the run
`id = {name}_{hex8}` (greppable in `run_context.yaml`). runkit writes its records
at the top of the run dir; the experiment writes under `out/` (`ctx.out`). See `design.md`.

## Layout

- `src/runkit/` — `exp` (decorator), `autocli` (the CLI core), `utils`, `cli`,
  `config`, `ui`.

## Status

Early. Provenance and a dirty-git gate are not yet part of the lightweight
runner. See `proposals.md` for the plan.
