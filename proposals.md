# runkit — proposals

Designs that are not built yet. `design.md` describes what the code does; this
file is where things live until they do. When a proposal lands, its settled
parts move to `design.md` and the rest is deleted here.

---

## Checkpoints

A long run saves its state along the way, so a crash or a stop does not lose
everything, and so a run can be evaluated or resumed before it ends. Today every
experiment does this its own way (control-kit's PPO overwrites `out/model.zip`
in place). runkit gives it a place, a name, and a record of which checkpoints
are complete.

```python
with ctx.checkpoint() as ckpt:                 # checkpoints/000003/
    model.save(ckpt.dir / "model.zip")

with ctx.checkpoint("best") as ckpt:           # checkpoints/best/
    model.save(ckpt.dir / "model.zip")
    ckpt.info["ep_return"] = ret               # saved with it
```

```python
@dataclass
class Checkpoint:
    name: str        # the folder name; runkit's counter ("000003") when none is given
    dir: Path        # {run dir}/checkpoints/<name>/
    index: int       # runkit's counter: the order of checkpoints, whatever their names
    info: dict       # yours; saved to checkpoint.yaml when the block exits
```

`ctx.checkpoint(name=None)` is the only way to make one — `index` and `dir` are
runkit's to assign. The name is free-form: the counter by default, or whatever
suits the experiment (`f"{step}"`, `"best"`, `"last"`). There is no step
argument: not every experiment has a step, and a step is one possible name.

### On disk

Checkpoints sit at the top level of the run dir, next to runkit's other
records, not under `out/`:

```
runs/test_policy/2026-09-25_10-02-11_a3f9c1e7/
├── config.yaml, run_context.yaml, meta.yaml, status.yaml
├── out/                       the body's, as always
└── checkpoints/
    ├── 000001/
    ├── 3000000/
    ├── best/
    │   ├── model.zip          the body's: written into ckpt.dir
    │   └── checkpoint.yaml    runkit's: written when the block exits
    └── latest -> best         runkit's: the highest index, once complete
```

The split follows the ownership rule (runkit owns the top level, the body owns
`out/`): runkit owns the folder structure — names, `checkpoint.yaml`, `latest` —
and the body owns the files inside each `ckpt.dir`, the way it is handed
`ctx.out`. It also keeps `ctx.checkpoint` optional: an experiment that would
rather manage checkpoints itself writes them under `out/` as before, and runkit
never touches them.

```yaml
# checkpoints/best/checkpoint.yaml
index: 7                          # runkit's counter: the order; latest = highest
name: best
time: '2026-09-25T13:47:40'       # when it finished writing
elapsed_s: 13529.4                # since the run started
run: test_policy_a3f9c1e7         # which run it came from, if the folder travels
info: {ep_return: 20.7}           # ckpt.info, if anything was put there
```

Rules:

- **`checkpoint.yaml` is written last, and only on a clean exit from the
  block.** Its existence is what makes a checkpoint complete: a folder without
  it was interrupted mid-save (killed, or the block raised), and every tool
  ignores it. If the block raises, nothing is recorded, `latest` does not move,
  and the exception propagates.
- **Order is `index`, never the name.** Names sort badly as text (`500000`
  after `3000000`) or not at all (`best`), so "latest" is the highest `index`:
  the `latest` link and anything that picks a checkpoint use it.
- **A repeated name replaces the old checkpoint** and takes a new `index`. That
  is what `best` and `last` want.
- **Only the live run can checkpoint** (`ctx.live`, see "Live runs" below). A
  `RunContext` rebuilt by `load_run` (in `eval` / `viz`) raises on
  `ctx.checkpoint`: it must not write into a run it only opened.
- **Reading back** returns the same type: a run's checkpoints as `Checkpoint`
  objects (read from their `checkpoint.yaml`), newest last — for `eval` to pick
  one, or a run to resume from.

### Open

- **A record in `status.yaml`** of the latest checkpoint, so `runkit ls` can
  show it without scanning — or is the folder enough?
- **Retention**: keep the last N, keep the best by an `info` key. Add when a
  folder of checkpoints gets too big.
- **`eval` of a run that is not `ok`**: today `eval` skips runs that failed or
  are still going. With complete checkpoints, it could evaluate their latest
  one instead.
- **Progress reporting** (how far along a live run is) — separate from
  checkpoints; see "Live runs" below.

---

## Live runs: progress and metrics

What a run reports about itself while it goes, beyond `status.yaml`'s lifecycle
(which runkit writes, and which already records `updated`, `host` and `pid` —
built, see design.md).

### `ctx.live`

`True` in the context the run wrapper hands the body — the process that owns
the run right now. `False` in a context rebuilt from disk (`load_run`, and so in
`eval` and `viz`). It is the one rule behind what a context may write:

| call | live (the run) | not live (eval, viz, `load_run`) |
| ---- | -------------- | -------------------------------- |
| `ctx.progress(...)` | yes | error |
| `ctx.checkpoint(...)` | yes | error |
| `ctx.record(**values)` | stream `run` | error: name a stream |
| `ctx.record("eval", **values)` | yes | yes |
| `ctx.record("run", **values)` | yes | error: `run` is the run's |

`live` describes this process, not the run: a read-only property backed by a
private field that only the run wrapper sets. It is not written to
`run_context.yaml` and does not take part in comparing contexts, so a context
rebuilt by `load_run` still equals the one the run had.

### Progress: `ctx.progress(n=None, *, total=None)`

How far along the run is, called by the body whenever it likes:

```python
@exp.run
def run(cfg, ctx):
    ctx.progress(total=cfg.steps)     # at the start: 0 of 10M
    for step in ...:
        ctx.progress(step)            # the total is remembered
```

- **Written flat into `status.yaml`**, the one file that changes:
  `progress: 3200000` and `total: 10000000` next to `status` and `updated`.
  `null` when never set. No stored fraction — a reader derives it.
- **`n` is any count** — env steps, iterations, candidates tried — not a
  "step": not every experiment has one. A bare percentage is
  `ctx.progress(0.32, total=1)`.
- **`total` is sticky**: set once (at the start, or when it becomes known), kept
  until set again; null for an open-ended run ("3.2M, still going").
  Keyword-only, so `ctx.progress(5, 10)` cannot be misread.
- **Throttled**: a write at most every few seconds, so calling it every
  iteration is fine; the final `status.yaml` write keeps the last values, so a
  failed run shows how far it got.
- **Live only.**

### Metrics: `ctx.record(stream=None, /, **values)`

A time series of named values, the thing every tracking library has at its core
(`tf.summary.scalar`, `wandb.log`, `mlflow.log_metric`), kept as plain files in
the run dir:

```python
ctx.record(it=it, steps=n, ep_return=ret, vx=vx)     # during the run -> metrics/run.jsonl
ctx.record("eval", ep_return=ret, fell=False)         # in eval         -> metrics/eval.jsonl
```

```
{run dir}/
├── checkpoints/
├── metrics/
│   ├── run.jsonl
│   └── eval.jsonl
└── out/
```

```
# metrics/run.jsonl -- one line per call; runkit adds time and elapsed_s
{"time": "2026-09-25T13:47:40", "elapsed_s": 13529.4, "it": 781, "steps": 3200000, "ep_return": 20.7}
```

- **Named `record`**, not `log`: "log" is the run's captured stdout (planned)
  and python's `logging`, both text.
- **A folder of streams**, next to `checkpoints/` and `out/`, runkit-owned. The
  deciding case is eval: its numbers must not mix into the training series.
  Each stream keeps its own keys.
- **The stream is the first argument, positional-only** (`/`), so a metric
  named `stream` is just a value. Without it, the stream is `run` — which only
  the live run may write; everything else names its stream.
- **JSON Lines**: calls may carry different keys (CSV needs fixed columns), an
  append is crash-safe (a killed run loses at most a line), and it reads back
  with `pandas.read_json(path, lines=True)`.
- **No step argument**: the experiment's x axis (`steps`, `it`, `epoch`) is
  just another key. runkit adds only `time` and `elapsed_s`, which always exist.
- **Separate from progress**: progress is one current position, overwritten;
  metrics are the whole history, appended. `record` does not update progress.

Why standardize it at all: this is the one output a generic tool can read
without knowing the experiment — `@exp.compare` across a sweep, or "final return
per run" in a `runkit ls` — where today each experiment writes its own format
(control-kit's `progress.csv`). A viewer, or export to TensorBoard / W&B, would
be small adapters over these files, later if at all.

### Open

- `record` from `eval` appends to the same stream every time eval runs; each
  line has its `time`, but a re-run of eval is not otherwise marked.
- A reader: `load_run(...).metrics("run")`, or a free function.

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
