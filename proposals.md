# runkit — proposals

Designs that are not built yet. `design.md` describes what the code does; this
file is where things live until they do. When a proposal lands, its settled
parts move to `design.md` and the rest is deleted here.

---

## Branching: a new run from a checkpoint

Continue an earlier run -- with a change, or none -- as a new run, starting from
one of its checkpoints:

```bash
runkit run lab.rl_env.test_policy --branch a3f9:best env.w_support=10
runkit run lab.rl_env.test_policy --branch runs/test_policy/latest      # just continue
```
```python
run(cfg, branch="a3f9:best")
```

This covers changing a run midway (stop, branch from its latest checkpoint with
the new value -- at most one checkpoint interval lost), continuing one that
crashed or was stopped (branch with no changes), and trying variants from a
common point (several branches of one checkpoint). A run is never modified
after the fact: every run stays described by its own `config.yaml` and, for a
branch, its parent.

### What `--branch` names

`RUN[:CHECKPOINT]`: the run as a run dir or a hex prefix of its id (as for
`eval` / `viz`), and optionally a checkpoint name -- by default the parent's
latest complete checkpoint (the highest index). A branch of a run that has no
complete checkpoint is refused.

### What runkit does

- **A new run dir**, as for any run. The parent is not touched.
- **The config: the parent's `config.yaml`, not the class defaults**, with the
  command line's `key=value` (and a config yaml) on top -- a branch states only
  what differs. The banner shows those differences against the parent.
- **The lineage in `meta.yaml`**:
  `branch: {run: test_policy_a3f9c1e7, dir: ..., checkpoint: best, steps: 3000000}`
  -- so any run's history can be traced back, branch by branch.
- **The checkpoint handed to the body**, as its `branch` argument: the
  `Checkpoint` the run branches from, or None for a fresh run. The same object
  `ctx.checkpoint` gives when saving -- what the body wrote into `ckpt.state`
  and `ckpt.info`, it reads back from `branch.state` and `branch.info` (also
  `name`, `index`, `summary`). runkit passes it only to a run function that
  declares the parameter, found by name as `cfg` is; the others keep
  `(cfg, ctx)`.
- **A banner line**: `branch  a3f9c1e7:best (3.0M steps)`.

### What the experiment does

Only the experiment knows its state, so it saves what it needs to continue in
its checkpoints, and loads it when handed a `branch`:

```python
@exp.run
def run(cfg, ctx, branch=None):
    if branch:
        model = PPO.load(branch.state / "model.zip", env=venv)
        venv = VecNormalize.load(branch.state / "vecnormalize.pkl", venv)
        start = branch.info["steps"]
    ...
    model.learn(total_timesteps=cfg.steps - start, reset_num_timesteps=False)
```

What "continue" needs, for control-kit's `test_policy` (mostly saved already):

| state | where | needed |
| --- | --- | --- |
| policy and value weights | `model.zip` | yes |
| optimizer state (Adam moments) | `model.zip` (SB3 saves it) | yes |
| observation / reward normalization | `vecnormalize.pkl` | yes |
| step count (schedules, curriculum) | `ckpt.info["steps"]` | yes |
| random generator state | not saved | only for bit-exact repeats |
| an episode in progress | not saved | no: episodes restart |

### The contract is a handshake

runkit hands over a checkpoint; the experiment has to have saved enough in it
to continue. Only the experiment knows what "enough" is, so runkit cannot check
it -- it is a convention, documented as a practice: *a checkpoint holds what
continuing needs*. (This is Ray Train's `get_checkpoint()` and SB3's
`load` + `reset_num_timesteps=False`; frameworks that own the training loop,
like Lightning, can do more because they know the state.) runkit catches the
obvious slips:

- **A run that takes no `branch`**: `--branch` on an experiment whose run
  function has no `branch` parameter is refused before the run starts
  ("test_policy.run takes no `branch`: it can't continue from a checkpoint").
  Declaring the parameter is the experiment's side of the handshake, and it
  shows in the signature, not only in the body.
- **An empty checkpoint**: a branch of one whose `state/` is empty is refused
  -- the body never saved anything.
- **An unread branch**: `branch` notes when its folder is used; if a branch
  reaches its first checkpoint without the body having touched it, runkit
  warns. This catches "forgot to load", not "loaded half".
- **A round-trip test**, the only real check, run once per experiment rather
  than every time: run to a checkpoint, branch from it, and compare the
  branch's first rows with the parent's after that checkpoint -- equal within
  noise, or exactly with the random state saved. A `runkit` command later,
  perhaps.

### Config over training, and the step count

A checkpoint does not store the config: the run's `config.yaml` holds it, a
schedule's *spec* included (control-kit's `_schedule`), and the values a schedule
gives are a function of the step count, which the checkpoint has. So config plus
checkpoint determine the effective config at that point, and a branch that
continues the step count continues its schedules where they were. Changing a
schedule's parameters in the branch moves the value to what the new schedule
gives at that step. Anything else describing the experiment's state at a
checkpoint goes in `ckpt.info`.

Schedules are control-kit's today (`scheduled_config.py`: a `_schedule` section
that mirrors the config, evaluated at a step), and are likely to move into
runkit. Then runkit knows the effective config over training, and can do what
control-kit does by hand now: record the effective values as a stream of its
own, hand the body the config for the current step, and -- for a branch --
continue the schedules from the checkpoint's step count without the experiment
doing anything. The argument above holds either way: config plus step count
determine the effective values, so a checkpoint need not store them.

### Metrics

A branch records its own streams from its first row. Its step count continues
the parent's (3.0M, 3.01M, ...), so the two plot on one axis with `--x steps`.

### Open

- **Plotting a lineage** as one curve: the parent up to the branch point, then
  the branch.
- **`runkit runs tree`** (or in a `runkit runs ls`): the branches of a run.
- **Branching across experiments** (a checkpoint of one as the start of another,
  e.g. pretraining) -- the same `--branch`, if the state fits; refused for now.
- **Schedules in runkit** -- control-kit's `ScheduledConfig` as a runkit
  feature: the `_schedule` convention, the effective config at a step, the
  effective values recorded as a stream, continued across a branch.
- **State codecs**, if the same save/load code keeps repeating across
  experiments: `run(cfg, ctx, state)`, where each file in a checkpoint's
  `state/` folder is an attribute (`model.sb3.zip` -> `state.model`, codec by
  suffix; small values together in `values.yaml`). Values load as they are;
  objects the body builds from `cfg` are attached (`state.attach(model=model)`)
  and restored in place on a branch, so a fresh run and a branch take one code
  path. runkit ships codecs for yaml, numpy and torch; a library registers its
  own on import (control-kit: sb3, VecNormalize), an experiment with
  `exp.codec`; pickle only when asked for. It sits on top of the `branch` argument
  without changing it.
- **Live adjustment** of a running run (`runkit adjust RUN key=value`, applied
  by the experiment at safe points and recorded with its step) -- deliberately
  left out until branching proves too slow for a real case.

---

## Eval takes a checkpoint

`eval` evaluates a checkpoint, not a run. A checkpoint knows its run -- its
folder is `<run dir>/checkpoints/<name>`, and `checkpoint.yaml` names the run --
and the run knows its experiment (`meta.yaml`: `module`, `script`, `launch`), so
a checkpoint is all `eval` needs to be pointed at:

```bash
runkit eval runs/test_policy/latest/checkpoints/best    # that checkpoint
runkit eval runs/test_policy/latest                      # the run's latest complete one
runkit eval a3f9:best                                    # RUN[:CHECKPOINT], as for --branch
runkit eval lab.rl_env.test_policy [RUN ...]             # today's form: the experiment's runs
```

From a path, runkit finds the eval function through the run's `meta.yaml` and
relaunches with the recorded extras, as `run` does from `experiment.toml`. The
experiment form stays: for every run of an experiment, and for evaluating with
a module other than the one recorded (a file that moved). Either way, eval runs
the module's code as it is now, not as it was when the run trained -- true
today, easier to forget when only a path is named.

### A checkpoint's layout

Like a run dir: runkit's own file at the top, the rest in named folders.

```
checkpoints/best/
  checkpoint.yaml    runkit's: index, name, time, info, row counts, summary
  state/             the run body's: what continuing needs
    model.zip
    vecnormalize.pkl
    config.yaml      if the body needs one (save_config, below)
  eval/              the eval's
    eval.yaml
    episodes.jsonl
    rollout.npz
```

- `ckpt.dir` is the checkpoint folder, `ckpt.state` is `ckpt.dir / "state"` --
  as `ctx.dir` is the run dir and `ctx.out` is `ctx.dir / "out"`. runkit makes
  `state/` when the block starts (in the staging folder, so `ckpt.state`
  follows the folder when it is swapped into place). The body saves into it
  (`model.save(ckpt.state / "model.zip")`); a branch reads `branch.state`, and
  nothing else.
- **`info` stays in `checkpoint.yaml`**: the body fills the dict
  (`ckpt.info.update(steps=...)`), runkit writes it with its own record when
  the block exits. It holds small scalars that describe the checkpoint --
  runkit shows them (the checkpoint line, the branch banner's step count) and
  may sort by them (retention by an `info` key), never requires one; the body
  may read them back on a branch (`branch.info["steps"]`). What continuing
  needs beyond such scalars goes in `state/`, which runkit never opens.
- `eval/` is made on first use and written after the checkpoint is complete;
  it is not state (a branch ignores it).
- **A checkpoint saved again under its name (`current`) replaces the whole
  folder, `eval/` with it.** That is right: the eval described a state that is
  gone.
- Old checkpoints have their files at the top: with no `state/`, runkit takes
  the checkpoint folder itself as the state, so they still branch and eval.

The config is not part of it. The run's `config.yaml` is the run's; a
config that changes over training (a schedule) is the body's to save in
`state/` if it needs it -- two functions make that a line each:

```python
from runkit import save_config, load_config
save_config(schedule(steps), ckpt.state / "config.yaml")   # atomic yaml
cfg = load_config(PolicyCfg, ckpt.state / "config.yaml")   # the typed config back
```

`load_config` builds the class the way a run's config is built (nested
configs, dict fields, tuples, `1e-4` as a float); a key the class no longer has
is an error, a field it gained takes its default. Configs stay plain
dataclasses -- no base class for `cfg.save()`.

### What the eval function gets

The checkpoint, and only that:

```python
@exp.eval
def evaluate(ckpt):
    cfg = load_config(PolicyCfg, ckpt.state / "config.yaml")
    model = PPO.load(ckpt.state / "model.zip")
    ...
    np.savez(ckpt.eval / "rollout.npz", ...)
    record(ckpt.eval / "episodes.jsonl", episode=i, ret=r, vx=vx)
```

`ckpt` is the same `Checkpoint` a branch gets (`dir`, `state`, `info`, `name`,
`index`, `summary`), with `eval` -- its `eval/` folder -- and `run`, the run
dir, for anything else (the run's `config.yaml` is there). No `ctx`: its
output folder and its records are the checkpoint's now. The eval no longer
picks a checkpoint itself (`ctx.checkpoints()[-1]` in control-kit's
`test_policy`). A run with no complete checkpoint is refused; one that failed
or is still going is evaluated at its latest complete one -- today `eval`
skips such runs.

### `record`, raw

`runkit.record(path, **values)` appends one json line to a file: `_time`
added, numpy values made plain, whole lines only. No streams, no run -- the
file is where it is told. `ctx.record(stream, ...)` becomes a layer on it
(`metrics/<stream>.jsonl`, `_elapsed_s`, the live window, `--follow`). An
eval's records go to its checkpoint like everything else it writes, and
`runkit metrics` reads them as they are:
`runkit metrics info checkpoints/best/eval/episodes.jsonl`.

### Open

- **Eval's own parameters** (episodes, a different env) -- see "Open between
  the proposals".
- **An eval curve over training**: an eval's result against each
  checkpoint's step count. It needs several checkpoints (numbered ones, not
  one `current`), each evaluated -- `runkit eval` over every checkpoint of a
  run, skipping those with an `eval/` -- and `runkit metrics` reading one eval
  file across a run's checkpoints. Until then, `ctx.record("eval", ...)` in the
  run body gives the curve as it trains.

---

## Checkpoints: open

Checkpoints are built (see design.md, "Checkpoints"). Still open:

- **Retention**: keep the last N, keep the best by an `info` key. Add when a
  folder of checkpoints gets too big.
- **`eval` of a run that is not `ok`**: covered by "Eval takes a checkpoint".

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
  yet (which checkpoint: see "Eval takes a checkpoint"). With `evaluate(ckpt)`
  there is no training config in the signature for `key=value` to be confused
  with, so one option is an annotated eval config, `evaluate(ckpt, ecfg:
  EvalCfg)`, built from the `key=value` of `runkit eval CKPT episodes=10`.
  Today `eval` rejects `key=value` outright.
- **`load_run` vs. "no status on `Run`".** That rule holds because a failed run
  raises, but `load_run` opens a run that failed or is still `running` (it has
  to: `viz` looks at failed runs). Should a loaded `Run` carry `status`?
