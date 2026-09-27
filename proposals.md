# runkit — proposals

Designs that are not built yet. `design.md` describes what the code does; this
file is where things live until they do. When a proposal lands, its settled
parts move to `design.md` and the rest is deleted here.

---

## Checkpoints: open

Checkpoints are built (see design.md, "Checkpoints"). Still open:

- **Retention**: keep the last N, keep the best by an `info` key. Add when a
  folder of checkpoints gets too big.
- **`eval` of a run that is not `ok`**: today `eval` skips runs that failed or
  are still going. With complete checkpoints, it could evaluate their latest
  one instead.

---

## Progress and metrics: open

`ctx.progress`, `ctx.record` and `ctx.live` are built (see design.md, "Progress
and metrics"). Still open:

- `record` from `eval` appends to the same stream every time eval runs; each
  line has its `_time`, but a re-run of eval is not otherwise marked.
- Showing progress: `runkit ls` (not built), or the closing line of a failed
  run ("failed at 3.2M / 10M").

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

---

## Open between the proposals

Questions left over from building `eval` and `viz`.

- **Eval's own parameters.** `evaluate(cfg, ctx)` gets the *training* config;
  episode count or which checkpoint has nowhere to go. And in
  `runkit eval mod <run dir> episodes=10`, `key=value` could mean either config.
  One option: a second annotated dataclass, `evaluate(cfg: PolicyCfg, ctx,
  ecfg: EvalCfg)`, with `key=value` routed to `EvalCfg` only. Today `eval`
  rejects `key=value` outright.
- **`load_run` vs. "no status on `Run`".** That rule holds because a failed run
  raises, but `load_run` opens a run that failed or is still `running` (it has
  to: `viz` looks at failed runs). Should a loaded `Run` carry `status`?
