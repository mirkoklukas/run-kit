# runkit — proposals

Designs that are not built yet. `design.md` describes what the code does; this
file is where things live until they do. When a proposal lands, its settled
parts move to `design.md` and the rest is deleted here.

---

## Branching and checkpoint eval: open

`--branch`, the `branch` argument, a checkpoint's `state/` and `eval/`,
`evaluate(ckpt)`, `runkit.record` and `save_config` / `load_config` are built
(see design.md, "Branching: a new run from a checkpoint", "Verbs" and
"Checkpoints"). Still open:

- **An unread branch**: a branch that reaches its first checkpoint without the
  body having touched `branch.state` could warn -- "forgot to load", not
  "loaded half".
- **A round-trip test**, the only real check of the handshake, run once per
  experiment: run to a checkpoint, branch from it, and compare the branch's
  first rows with the parent's after that checkpoint -- equal within noise, or
  exactly with the random state saved. A `runkit` command, perhaps.
- **Plotting a lineage** as one curve: the parent up to the branch point, then
  the branch.
- **`runkit runs tree`** (or in a `runkit runs ls`): the branches of a run.
- **Branching across experiments** (a checkpoint of one as the start of another,
  e.g. pretraining) -- the same `--branch`, if the state fits; refused for now
  (a checkpoint of another experiment is not found).
- **Schedules in runkit** -- control-kit's `ScheduledConfig` as a runkit
  feature: the `_schedule` convention, the effective config at a step, the
  effective values recorded as a stream, continued across a branch -- and then
  runkit saving the effective config in each checkpoint itself.
- **State codecs**, if the same save/load code keeps repeating across
  experiments: `run(cfg, ctx, state)`, where each file in a checkpoint's
  `state/` folder is an attribute (`model.sb3.zip` -> `state.model`, codec by
  suffix; small values together in `values.yaml`). Values load as they are;
  objects the body builds from `cfg` are attached (`state.attach(model=model)`)
  and restored in place on a branch, so a fresh run and a branch take one code
  path. runkit ships codecs for yaml, numpy and torch; a library registers its
  own on import (control-kit: sb3, VecNormalize), an experiment with
  `exp.codec`; pickle only when asked for. It sits on top of the `branch`
  argument without changing it.
- **Live adjustment** of a running run (`runkit adjust RUN key=value`, applied
  by the experiment at safe points and recorded with its step) -- deliberately
  left out until branching proves too slow for a real case.
- **Eval's own parameters** (episodes, a different env) -- see "Open between
  the proposals".
- **An eval curve over training**: an eval's result against each checkpoint's
  step count. It needs several checkpoints (numbered ones, not one `current`),
  each evaluated -- `runkit eval` over every checkpoint of a run, skipping those
  with an `eval/` -- and `runkit metrics` reading one eval file across a run's
  checkpoints. Until then, `ctx.record("eval", ...)` in the run body gives the
  curve as it trains.
- **The launch of `runkit eval <checkpoint>`** comes from `experiment.toml` as
  it is now, like the code; the run's recorded `launch` (meta.yaml) is not used.
  Use it if the two ever need to differ.

---

## Checkpoints: open

Checkpoints are built (see design.md, "Checkpoints"). Still open:

- **Retention**: keep the last N, keep the best by an `info` key. Add when a
  folder of checkpoints gets too big.

---

## Progress and metrics: open

`ctx.progress`, `ctx.record` and `ctx.live` are built (see design.md, "Progress
and metrics"). Still open:

- `record` from `eval` appends to the same stream every time eval runs; each
  line has its `_time`, but a re-run of eval is not otherwise marked.
- Showing progress: `runkit ls` (not built), or the closing line of a failed
  run ("failed at 3.2M / 10M").

---

## Naming: `record` writes to `metrics/` (soft)

The method is `ctx.record`, what it writes is "metrics": the `metrics/` folder,
`runkit metrics`, `load_metrics` / `compile_metrics` / `plot_metrics`, the
`metrics:` row counts in `checkpoint.yaml`. Two words for one thing, and
"metrics" undersells it -- streams hold schedule values, strings
(`note="eval"`), and would hold an eval's episodes. Two ways to make them one
word; not decided.

**A. Everything is "records".** `records/<stream>.jsonl`, `runkit records
info|follow|plot`, `load_records` / `compile_records` / `plot_records`,
`records:` in `checkpoint.yaml` (the counter is already `_live.records`), and
the raw `runkit.record(path, ...)` fits. Frees "metrics" for nothing in
particular. Old runs read from `metrics/` (and old `checkpoint.yaml`'s
`metrics:`) when there is no `records/`; the old command and functions stay as
aliases for a while (control-kit's `test_policy` uses `load_metrics`).

**B. The method says "metrics".** `ctx.record_metrics(...)` (or
`log_metrics`, as MLflow; W&B's is `log`), everything else stays. Smaller:
one method renamed, the old one an alias. But the name is long for the most
called method, it still undersells a stream of strings or schedule values,
and the raw `runkit.record(path, ...)` would need a name of its own.

Leaning A: the verb and the noun match, and it is the more accurate word.
Do it as its own change, not mixed into another.

---

## Cross-run comparison

`eval` and `viz` (built; see design.md) look at one run. Comparing runs is a
different signature — it takes many runs and there is no single `cfg` to hand
it:

```python
@exp.compare
def compare(runs: list[Run]): ...
```

`runkit compare experiment.py [RUN ...]`, defaulting to every run of the
experiment. Add it when a real need shows up.

---

## `experiment.toml` and launching

### The motivating case

A PPO experiment in control-kit (`lab/rl_env/test_policy.py`) is used like this
today:

```bash
uv run --extra mjx --extra sb3 python -m lab.rl_env.test_policy steps=10e6 --tag=gait
uv run --extra mjx --extra sb3 python -m lab.rl_env.test_policy eval runs/<run dir>
uv run --extra mjx --extra sb3 python -m lab.rl_env.test_policy bench
```

Three things hurt:

- **Extras on every call.** The experiment needs the `mjx` and `sb3` extras, and
  the caller has to remember them. Forget one and the import fails.
- **Hand-rolled subcommands.** Training goes through `main(run)`; `eval` and
  `bench` are dispatched by the module's own `__main__`, with their own argv
  handling. *(Now covered for `eval` by `Experiment` and its verbs — see
  design.md, "Verbs: run, eval, viz".)*
- **Evaluation is outside runkit.** *(Covered the same way: `@exp.eval`.)*

(Detached runs are the fourth pain: starting in the background, following the
log, stopping a run cleanly. They get their own section when designed.)

### What is left of the verbs

The verbs themselves are built (`python experiment.py [run|eval|viz] ...`, and
`runkit <verb> experiment.py ...`). Still open:

- **Later verbs:** `runkit ls <experiment>` (its runs, with status), and `tail`
  / `stop` once detached runs exist.
- **runkit's own options**, in the slot between verb and experiment:
  `runkit run --detach experiment.py ...` is the obvious first.
- **Role-less commands** (`bench` above) stay in the module's own `__main__` for
  now. Revisit if several appear, e.g. with a registered `@exp.command`.

### `experiment.toml`: what is left

`root`, `extras` and `vars`, and the relaunch under uv, are built (see
design.md, "Where runs go" and "Starting the process"). Still open:

- **`project`** — a uv project other than the nearest `pyproject.toml` above
  the experiment (see below). The key is reserved; setting it warns.
- **A check under `python -m`**, which cannot relaunch: warn when the
  experiment.toml asks for extras the running environment was not started with.
- **No `${VAR}` interpolation** in paths for now; add it when a case needs more
  than a base path, and then fail loudly on an unset variable
  (`os.path.expandvars` leaves it in silently).

### An experiment with its own uv environment

If the experiment's folder has its own `pyproject.toml`, runkit already launches
with `uv run --project <folder>` — the nearest one wins. `project = ...` would
point elsewhere. `--project`
selects that environment without changing the current directory, so the caller
never has to `cd`. Such a project lists the shared library as a path dependency
(e.g. `controlkit = { path = "../..", editable = true }`).

Separate projects are for genuinely conflicting dependencies. For extras that only
conflict with each other (torch's CUDA wheels vs. `jax[cuda12]`), uv can declare
them mutually exclusive in one project instead —
`[tool.uv] conflicts = [[{ extra = "sb3" }, { extra = "vm" }]]` — untested here.

### Deliberately left out

- **`[commands]`** (rewiring a verb to a function without touching code). The
  decorators already say which function plays which role; a table would only
  duplicate that. Add it if rewiring is ever needed.
- **Per-command extras.** Extras belong to a module (its imports decide them), so
  overrides are per module.
- **Per-machine settings.** Machine differences go in `pyproject.toml` markers
  (see design.md, "Starting the process"), which cover the Mac-laptop vs.
  Linux-GPU-box split. If a difference ever depends on hardware rather than
  platform (a Linux box with vs. without a GPU), the design would be a named
  profile each machine declares once (`export RUNKIT_PROFILE=gpu`) and the toml
  defines (`[env.profile.gpu]`), applied between `[env]` and `[env.<stem>]` with
  the same merge rules and recorded in `meta.yaml`'s `launch`. Not by hostname
  (brittle), and not by an untracked local override file (runs on two machines
  could silently differ).

---

## Open between the proposals

Questions left over from building `eval` and `viz`.

- **Eval's own parameters.** Episode count, a different env: nowhere to go
  yet. With `evaluate(ckpt)`
  there is no training config in the signature for `key=value` to be confused
  with, so one option is an annotated eval config, `evaluate(ckpt, ecfg:
  EvalCfg)`, built from the `key=value` of `runkit eval CKPT episodes=10`.
  Today `eval` rejects `key=value` outright.
- **`load_run` vs. "no status on `Run`".** That rule holds because a failed run
  raises, but `load_run` opens a run that failed or is still `running` (it has
  to: `viz` looks at failed runs). Should a loaded `Run` carry `status`?
