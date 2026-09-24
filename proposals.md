# runkit — proposals

Designs that are not built yet. `design.md` describes what the code does; this
file is where things live until they do. When a proposal lands, its settled
parts move to `design.md` and the rest is deleted here.

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

### `experiment.toml`: how to start the process

Extras have to be known **before** the module is imported — without them the
import is what fails. So they cannot live only on a decorator: reading
`run._runkit_extras` needs the import that the extras are required for. Parsing
the source for a literal `extras=[...]` would work, but is brittle.

Instead, each experiment folder may carry an `experiment.toml`, read with
`tomllib` before anything is imported:

```toml
# lab/rl_env/experiment.toml
[env]
extras = ["mjx"]
root = "ctk:runs"

[env.test_policy]
extras = ["mjx", "sb3"]
```

The split follows the order of events:

| when | where | owns |
| ---- | ----- | ---- |
| before import | `experiment.toml` | how to start the process: extras, env vars, project, root |
| after import | decorators | what a function is: name, config type, role, run-dir behavior |

Rules:

- **Discovery.** runkit walks up from the target module to the nearest
  `experiment.toml`, the way tools find `pyproject.toml`. No file means today's
  behavior.
- **Per-module overrides.** `[env]` holds the folder's defaults; `[env.<module>]`
  (keyed by the module's file stem) is applied on top for that module. Lists
  *replace* (the override states the full set, readable in one place); tables
  such as `vars` *merge*, the override winning per key. A module named like one of
  the `[env]` keys (`extras`, `vars`, ...) is rejected with a clear error.
- **Keys:**
  - `extras` — uv extras to launch with.
  - `vars` — environment variables for the process (e.g. the JAX flags that
    today live as copy-paste prefixes in READMEs).
  - `root` — where run dirs go. *(Built — see design.md, "Where runs go".)*
  - `project` — the uv project to launch in (see below).
- **Paths** in the toml are relative to the folder containing `experiment.toml`
  (`"../../runs"`), or use runkit's existing scheme prefixes, which already
  resolve through environment variables: `"ctk:runs"` resolves against
  `$RUNKIT_PATH_CTK`, `exp:` against the experiment's folder. No `${VAR}`
  interpolation for now; add it when a case needs more than a base path, and then
  fail loudly on an unset variable (`os.path.expandvars` leaves it in silently).

### Launching

`runkit <verb> <module>` resolves the toml, then re-executes itself as

```bash
uv run [--project <dir>] --extra mjx --extra sb3 runkit <verb> <module> ...
```

and only then imports the module (guarded by an env var so it relaunches once).
The same trick `ctk play` uses to relaunch under `mjpython`. The resolved extras
go into `meta.yaml` as provenance.

`python -m module` keeps working unchanged, but cannot fix its own environment. A
decorator attribute (e.g. `extras` recorded on the wrapped function) can still
serve as a **check** after import: warn when the running environment lacks what
the experiment declares.

### An experiment with its own uv environment

If the experiment's folder has its own `pyproject.toml`, runkit launches with
`uv run --project <folder>` (or wherever `project = ...` points). `--project`
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
