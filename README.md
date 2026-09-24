# runkit

Lightweight, reproducible experiment runs. An `Experiment` registers a
`run(cfg, ctx)` — which gets a fresh, uniquely-named run dir with the resolved
config frozen into it — and, optionally, an `eval` and a `viz` that open an
existing run.

```python
# experiment.py
from dataclasses import dataclass
from runkit import Experiment, RunContext

exp = Experiment("baseline")


@dataclass
class Config:
    seed: int = 1
    lr: float = 3e-4


@exp.run
def run(cfg: Config, ctx: RunContext):
    (ctx.out / "result.txt").write_text(f"seed={cfg.seed}")   # ctx.out = {run dir}/out
    return {"score": cfg.seed * 0.1}                          # -> {run dir}/retval.json


@exp.viz
def show(cfg: Config, ctx: RunContext):                       # "look here first"
    print((ctx.out / "result.txt").read_text())


if __name__ == "__main__":
    exp.main()
```

Run it — with `python` the verb follows the file; with `runkit` it comes first:

```bash
python experiment.py [run] [config.yaml] [key=value ...] [--flag ...]
python experiment.py viz [RUN]            # RUN: a run dir or hex prefix; default: latest
python experiment.py eval [RUN]
runkit <verb> experiment.py ...           # runkit takes the verb first; or: some.module
cd "$(runkit root)"                       # go where runs go
cd "$(runkit latest experiment.py)"       # ... or to this experiment's latest run (or: root)
```

- bare `key=value` → overrides into **config** (the "what")
- `--flag=value` → sets **staging** (the "how/where"): `--tag`, `--root`
- without `--root`, runs go where the nearest `experiment.toml` says (`[env] root = ...`), else `./runs`
- config paths resolve as `cwd:NAME` (default) / `exp:NAME` (next to the experiment) / `/abs`,
  plus user-defined `SCHEME:NAME` bases via `$RUNKIT_PATH_<SCHEME>` (e.g. `ctk:x.yaml` → `$RUNKIT_PATH_CTK/x.yaml`)

Run dirs land at `{root}/{name}/{date}_{time}_{hex8}[_{tag}]/` (plus a
`{root}/{name}/latest` link to the newest one), with the run
`id = {name}_{hex8}` (greppable in `run_context.yaml`). runkit writes its records
at the top of the run dir; the experiment writes under `out/` (`ctx.out`). See `design.md`.

## Layout

- `src/runkit/` — `exp` (`Experiment`, the run wrapper), `runs` (finding and
  opening run dirs), `autocli` (the verb dispatcher), `cli` (the `runkit`
  command), `settings` (`experiment.toml`), `config`, `utils`, `ui`.

## Status

Early. Provenance and a dirty-git gate are not yet part of the lightweight
runner. See `proposals.md` for the plan.
