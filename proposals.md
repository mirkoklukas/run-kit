# runkit — proposals

Designs that are not built yet. `design.md` describes what the code does; this
file is where things live until they do. When a proposal lands, its settled
parts move to `design.md` and the rest is deleted here.

---

## Visualization / analysis

A run writes; something has to look at what it wrote. The convention:

```
experiment.py    Config + run(cfg, ctx)       -- produces a run dir
visualize.py     visualize(cfg, ctx)          -- consumes one
```

```python
@visualization
def visualize(cfg: Config, ctx: RunContext):
    fig = plot(json.loads((ctx.dir / "retval.json").read_text()))
    fig.savefig(ctx.out / "figures" / "loss.png")
```

`@experiment` creates a run dir and freezes the config into it; `@visualization`
is the inverse — it opens an existing one and thaws the config back out. Same two
arguments in the body, and as with `@experiment` the decorator changes the
calling convention:

| decorator | body you write | decorated callable |
| --------- | -------------- | ------------------ |
| `@experiment` | `run(cfg, ctx)` | `run(cfg, *, tag=..., root=...)` |
| `@visualization` | `visualize(cfg, ctx)` | `visualize(run_dir)` |

```bash
python visualize.py runs/baseline/2026-06-26_15-40-12_abl-a_a3f9c1e7
```

The plumbing is public, mirroring `init_run`:

```python
init_run(cfg, name=..., ...)  -> RunContext   # create a run dir
load_run(run_dir, Config)     -> Run          # open one
```

`load_run` returning a `Run` means one type covers both directions: a run you
just executed and a run rebuilt from disk are the same thing, which is what
makes a list of them workable. In a notebook, skip the decorator and call
`load_run` directly.

Figures go to `out/figures/` -- visualization is experiment code, so it writes
where the experiment does, and everything about a run stays in one place. This
does mean a finished run dir is no longer immutable; the containment rule is
that visualization writes *only* under `out/figures/`, and re-running
overwrites rather than accumulates.

Cross-run comparison is a different signature — it takes many run dirs and there
is no single `cfg` to hand it. Naming it so the single-run form above is not
mistaken for the general answer:

```python
@comparison
def compare(runs: list[Run]): ...
```

Add it when a real need shows up.

---

## Verbs, roles and `experiment.toml`

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
  handling.
- **Evaluation is outside runkit.** It reads a run dir and writes into it, which
  is exactly the `@visualization` shape, but runkit does not know about it.

(Detached runs are the fourth pain: starting in the background, following the
log, stopping a run cleanly. They get their own section when designed.)

### Verbs on runkit, roles on functions

runkit gets a small, fixed set of verbs; the module says which function plays
which role:

```bash
runkit run  lab.rl_env.test_policy steps=10e6 --tag=gait
runkit eval lab.rl_env.test_policy runs/<run dir>
runkit eval lab.rl_env.test_policy               # latest run of this experiment
```

```python
@experiment(name="test_policy")
def run(cfg: PolicyCfg, ctx: RunContext): ...     # what `runkit run` calls

@evaluation
def evaluate(cfg: PolicyCfg, ctx: RunContext): ...   # what `runkit eval` calls
```

- **The target is a module** (dotted, or a file path as today). runkit imports
  it and finds the function carrying the role's decorator. No naming convention.
- **Each verb has fixed semantics.** `run` creates a run dir; `eval` opens an
  existing one, thaws its `config.yaml`, and writes only inside it. So every
  experiment behaves the same way, and runkit can enforce it.
- **`eval` without a run dir** picks the latest run of that experiment: runkit
  knows the experiment's name, and its runs are `{root}/{name}/*`. Computed from
  the dirs, not read from the `latest` link (which is a human shortcut).
- **`@evaluation`** is the `@visualization` proposal generalized: same calling
  convention (`evaluate(run_dir)`), same containment rule (writes only under the
  run's `out/`, e.g. `out/eval/eval.yaml`, `out/eval/rollout.npz`).
- **Later verbs:** `ls` (runs of an experiment, with status), and `tail` / `stop`
  once detached runs exist.
- **Role-less commands** (`bench` above) stay in the module's own `__main__` for
  now. Revisit if several appear, e.g. with `runkit call module:function` or a
  `[commands]` table (below).

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
root = "ctk:runs/rl_env"

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
  - `root` — where run dirs go. Precedence:
    `--root` > `[env.<module>]` > `[env]` > runkit's `./runs`.
  - `project` — the uv project to launch in (see below).
- **Paths** in the toml are relative to the folder containing `experiment.toml`
  (`"../../runs/rl_env"`), or use runkit's existing scheme prefixes, which already
  resolve through environment variables: `"ctk:runs/rl_env"` resolves against
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

### Small fix found on the way

- **Start banner.** It prints the whole config — ~60 lines for the PPO config.
  A shorter form (e.g. only fields that differ from the defaults, full config in
  `config.yaml`) would help large configs.

---

## Open between the proposals

Questions the sections above raise against each other. Settle them before
building eval.

- **One decorator, not two.** `@evaluation` generalizes `@visualization`, and
  now that the body owns `out/` and runkit the top level, both write under
  `out/` anyway. Likely answer: drop `@visualization` and give eval its own
  subdir (`out/eval/`), overwritten on re-run.
- **Eval's own parameters.** `evaluate(cfg, ctx)` gets the *training* config;
  episode count or which checkpoint has nowhere to go. And in
  `runkit eval mod <run dir> episodes=10`, `key=value` could mean either config.
  One option: a second annotated dataclass, `evaluate(cfg: PolicyCfg, ctx,
  ecfg: EvalCfg)`, with `key=value` routed to `EvalCfg` only.
- **`load_run` vs. "no status on `Run`".** That rule holds because a failed run
  raises, but `load_run` can open a run that failed or is still `running`.
  Either loaded runs carry `status`, or `load_run` refuses anything not `ok`.
- **"Latest run" needs a root.** `runkit eval <module>` without a run dir has
  to know where to look; today the default is `runs` relative to cwd. So
  `root` in `experiment.toml` (or something like it) comes first. Also:
  latest *started* (what the `latest` link means), or latest `ok`?
