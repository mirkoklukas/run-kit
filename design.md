# runkit — design

Lightweight, reproducible experiment runs. You write a `Config` and a
`run(cfg, ctx)`; runkit turns the script into a CLI, gives every run a fresh
self-describing directory, and records what the run was and how it went. Next to
the run, an experiment can say how to evaluate a run and how to look at one.

It is deliberately small. Features get added when a real need shows up, not
before — there is no config composition, no server, no database, and no tracking
UI. The run directory is the whole interface.

Designs that are proposed but not built live in `proposals.md`.

---

## Usage

### An experiment

An experiment is a python file with an `Experiment`, a `Config` dataclass and
a `run(cfg, ctx)` — plus, optionally, an eval and a viz:

```python
# experiment.py
from dataclasses import dataclass
from runkit import Experiment, RunContext

exp = Experiment("baseline")


@dataclass
class Config:
    lr: float = 3e-4
    seed: int = 1
    steps: int = 1000


@exp.run
def run(cfg: Config, ctx: RunContext):
    ckpt = ctx.out / "checkpoints"       # ctx.out:  {run dir}/out -- write everything here
    ckpt.mkdir()                         # ctx.id:   unique run id, "baseline_a3f9c1e7"
    ...                                  # ctx.name: "baseline"
    return {"loss": 0.31}                # optional -> retval.json


@exp.eval                                # optional: process what a run wrote
def evaluate(cfg: Config, ctx: RunContext):
    ...                                  # writes under ctx.out, e.g. ctx.out / "eval"


@exp.viz                                 # optional: the first thing to look at
def show(cfg: Config, ctx: RunContext):
    ...


if __name__ == "__main__":
    exp.main()
```

Two things are required and checked: `Config` must be a dataclass, and the
run's `cfg` parameter must carry it as a type annotation — that annotation is
how runkit knows what to build from the command line.

A run-only experiment has a shorthand, and `main(run)` works for it as it
always has:

```python
@experiment(name="baseline")             # == Experiment("baseline").run
def run(cfg: Config, ctx: RunContext): ...

if __name__ == "__main__":
    main(run)
```

### Running it

```bash
python experiment.py                                  # dataclass defaults
python experiment.py lr=1e-4 seed=7                   # + overrides
python experiment.py config.yaml                      # + a config yaml
python experiment.py config.yaml lr=1e-4 --tag=abl-a  # + a variant label
python experiment.py --help                           # fields and flags for this experiment

python experiment.py viz                              # look at the latest run
python experiment.py eval a3f9                        # evaluate the run whose id starts a3f9
python experiment.py viz --help                       # what viz takes

runkit viz experiment.py                              # the same, via runkit
runkit viz lab.baseline.experiment                    # ... or by module name
```

### Arguments vs flags

The command line has exactly two namespaces, and they never touch each other:

| form | namespace | means |
| ---- | --------- | ----- |
| `key=value` | **config** | *what* the experiment does — fields of your `Config` |
| `--flag=value` | **staging** | *how and where* this attempt runs — never seen by `Config` |

So `seed=7` can only ever set `Config.seed`, and `--tag=abl-a` can only ever
stage the attempt. A typo'd field (`sed=7`) is rejected against the dataclass; a
typo'd flag (`--tagg`) is rejected against the list of accepted flags. Neither
can silently land in the other namespace.

Config values are type-coerced (`7` → int, `true` → bool, `null` → None), and
dotted keys nest into sub-dataclasses:

```bash
python experiment.py optim.lr=1e-4 optim.warmup=100
```

Each namespace has the same three layers, last wins, and they never cross — a
`config.yaml` cannot set `--root`, an `experiment.toml` cannot set `lr`:

| namespace | defaults | file layer | command line |
| --------- | -------- | ---------- | ------------ |
| config — the "what" | dataclass defaults | `config.yaml` | `key=value` |
| staging — the "how/where" | runkit's (`./runs`) | `experiment.toml` | `--flag` |

The differences follow from what each namespace is about:

- **How the file is picked.** A `config.yaml` is named per call. The
  `experiment.toml` is found by location — the nearest one upward — because the
  environment belongs to where the code lives, not to one attempt.
- **What is recorded.** The resolved config is frozen into the run dir: it is
  what the run *was*. Staging mostly is not: the root is where the run dir sits,
  the tag goes in `meta.yaml`.
- **When it is read.** The toml is read before the experiment is imported, so it
  can set up the process (extras, env vars — see `proposals.md`); the config is
  needed only once the dataclass exists.

Which is also the rule for what belongs in `experiment.toml`: what is true of
*every* attempt from that folder (root, extras, env vars). Per-attempt staging
such as `--tag` stays on the command line. And the file is TOML rather than
YAML on purpose: `experiment.yaml` next to `config.yaml` files would read as
"the experiment's config", and TOML is what tool settings use in python projects
(`pyproject.toml`, `uv.toml`).

### Config resolution

Three layers, last wins:

```
dataclass defaults  →  config.yaml  →  key=value
```

The yaml is optional, and so is any key in it — a partial config is normal and
most runs pass none at all. Defaults are not a merge step: only keys somebody
actually set are passed to the constructor, so every unmentioned field falls
through to its dataclass default. A field with *no* default that nobody sets
fails loudly, with python's own `missing 1 required positional argument`.

The same holds inside a nested config. Overriding `model.lr` changes that one
field of the config `model` would otherwise hold — the field's own default —
and leaves the rest of it alone:

```python
@dataclass
class PolicyCfg:
    model: ModelCfg = field(default_factory=lambda: ModelCfg(pad_cells=1))
```

`model.lr=1e-4` gives `ModelCfg(pad_cells=1, lr=1e-4)`, not `ModelCfg`'s class
defaults with `lr` set: the nested override is applied to the field's default
(`default_factory()` or `default`), and a subclass default stays that subclass.
Only a nested field with no default is built fresh from its class.

The config yaml is named either as a bare positional or as `--config=PATH`
(giving both is an error unless they agree). Its path may carry a scheme:

```
/abs/path         →  used as-is
NAME or cwd:NAME  →  relative to cwd (no prefix = cwd)
exp:NAME          →  relative to the experiment file's own directory
SCHEME:NAME       →  relative to $RUNKIT_PATH_<SCHEME>
```

`cwd:` and `exp:` are built in. Anything else is yours to define: `ctk:x.yaml`
resolves against `$RUNKIT_PATH_CTK` (scheme uppercased, `~` expanded). If that
variable is unset the token is left as a literal path and you get a warning, so
a typo'd scheme surfaces instead of silently becoming a filename. Pairs well
with `direnv` for per-project bases.

One yaml quirk worth knowing: YAML 1.1 only reads a float when the mantissa has
a dot, so `lr: 1e-4` in a file loads as the *string* `"1e-4"`. The `float`
annotation on the field is what recovers it. The same repair covers ints written
as `1e3`.

### Flags

```
--tag=TAG        variant label; becomes part of the run dir name
--config=PATH    the config yaml (same as the bare positional)
--root=DIR       where run dirs are created (default: experiment.toml, else ./runs)
```

The accepted flags are not a hardcoded list — they are the keyword arguments the
decorated function takes, so anything else is rejected before the run starts.
`name` is not among them: it is the experiment's identity, set once in the
decorator, with no command-line override.

There is deliberately no flag for an exact run dir. runkit always places the run
(see "Run dir naming"), so every run dir follows the scheme, sits with its
experiment and is reachable through `latest`. Where a known path is needed,
`{root}/{name}/latest` is one; where a scheduler hands out a folder, pass it as
`--root`.

### Where runs go: `experiment.toml`

Without `--root`, the root comes from the nearest `experiment.toml` at or above
the experiment's file, else `./runs` (relative to where you launch from):

```toml
# lab/rl_env/experiment.toml
[env]
root = "ctk:runs"               # every experiment in this folder (and below)

[env.test_policy]               # keyed by the experiment file's stem
root = "../../runs/policy"      # this experiment only
```

Precedence: `--root` > `[env.<stem>]` > `[env]` > `./runs`. A path is relative
to the folder holding `experiment.toml`, or uses a scheme prefix (`ctk:` →
`$RUNKIT_PATH_CTK`, `exp:` → that same folder). `eval` and `viz` resolve it the
same way, so they look where the run wrote. `root` is the only key read so far;
the rest of the file is in `proposals.md`.

```bash
runkit root                              # the root, resolved from the current folder
runkit root lab/rl_env                   # ... from that folder
cd "$(runkit root)"                      # go there
```

`runkit root` prints the path — only the path, on stdout, so it drops into
`$(...)`. It cannot `cd` for you: no program can change its parent shell's
directory. A shell function does it in one word:

```bash
rkroot() { cd "$(runkit root "$@")"; }   # in ~/.zshrc
```

From a folder it applies `[env]`; given an experiment file it also applies that
file's `[env.<stem>]`, so it names exactly the root a run of it would use.

### Verbs: run, eval, viz

The first argument may name a verb; without one, it is `run`. The three roles
form a pipeline over a run dir:

| verb | what it does | argv after the verb | the body writes |
| ---- | ------------ | ------------------- | --------------- |
| `run` | creates a run dir | `[config.yaml] [key=value ...] [--tag] [--root]` | `out/` |
| `eval` | processes what a run wrote | `[RUN] [--root]` | e.g. `out/eval/` |
| `viz` | presents a run: "look here first" | `[RUN] [--root]` | e.g. `out/figures/`, the terminal |

`viz` is the experimenter's answer to "how do I check how this went?". Someone
who does not know how `out/` is laid out — or you, months later — runs it
instead of reading the code. It is cheap and repeatable; anything expensive
belongs in `eval`, which `viz` can show the results of, or call.

`RUN` picks one run dir of this experiment:

```
(nothing)          the latest run under {root}/{name}; for eval, the latest `ok` one
runs/baseline/...  a run dir (`runs/baseline/latest` works)
a3f9               a hex prefix of the run's id -- the {hex8} in its dir name
```

`eval` skips runs that failed or are still going, since there is nothing to
evaluate; `viz` does not, since a failed run is exactly when you want a look.
A run dir of a different experiment is refused.

`eval` and `viz` get the same `(cfg, ctx)` as the run did: the config thawed
from `config.yaml` — no `key=value`, the config is the one the run was made
with — and the run's `RunContext`. Their `cfg` annotation says which class to
thaw into; without one, the run's is used.

**One experiment per file.** A file holds one `Experiment`, and its roles are
registered on it, once each. That is what makes the file a complete answer to
"what can I do with this experiment".

**`runkit` takes the verb first.** `runkit <verb> experiment.py ...` takes
after the experiment what `python experiment.py <verb> ...` takes after the
verb. With `python` the file comes first because it is python's own argument;
with `runkit` the verb does, which leaves the slot between verb and experiment
for runkit's own options (none yet — say `runkit run --detach experiment.py`).
The verb is required there. The experiment may also be a dotted module
(`runkit viz lab.baseline.experiment`, like `python -m`); runkit imports it and
finds its one `Experiment` — or the one behind an `@experiment` function.

Two more verbs are built in, for every experiment, and print a path — only the
path, on stdout — for `cd "$(...)"`:

```bash
cd "$(runkit root experiment.py)"       # {root}/{name}: this experiment's runs
cd "$(runkit latest experiment.py)"     # its latest run dir
python experiment.py latest             # the same, the python way
```

Both resolve the root as a run does (`--root`, else `experiment.toml`, else
`./runs`). `root` with an experiment is that experiment's folder *in* the root;
without one (or with a folder) it is the root itself. `latest` is computed from the run
dirs (the latest started, as the `latest` link means), not read from the link.

One catch: a bare first argument that is a verb is taken as the verb, so a
config file named exactly `eval`, `viz`, `root` or `latest` (no extension)
needs `--config=`.

### What it prints

runkit's own output goes to stderr, so stdout is the experiment's: `2>/dev/null`
leaves only what the body prints. A run is bracketed by two things:

```
╭─ ▶ runkit · baseline ─────────────────────────────────╮
│  id  baseline_a3f9c1e7                                │
│ tag  abl-a                                            │
│ dir  runs/baseline/2026-06-26_15-40-12_a3f9c1e7_abl-a │
│                                                       │
│ lr: 0.0001                                            │
│ 2 more at their defaults · all in config.yaml         │
╰───────────────────────────────────────────────────────╯
  ...whatever the body prints...
  ✓ baseline_a3f9c1e7  ok in 3m 12s  → runs/baseline/2026-06-26_15-40-12_a3f9c1e7_abl-a
```

The banner shows only the config fields that differ from the dataclass defaults
(as dotted keys, `optim.lr`), since a large config would fill the screen and the
whole of it is in `config.yaml`. A field without a default always shows.

The closing line is how the run went, how long it took and where it is — so
the end of a long run answers that without scrolling back. For a failed run it
is `✗ ... failed after 2.1s (ValueError: bad shape) → .../traceback.txt`,
printed just before python's own traceback; an interrupted one says so. Like the
status writes, it is best-effort and never replaces the experiment's exception.

`eval` and `viz` print one header line, `▶ runkit · baseline · viz <run dir>`,
and then whatever their body prints.

### What lands on disk

```
runs/baseline/2026-06-26_15-40-12_a3f9c1e7_abl-a/
├── config.yaml        the resolved config -- what it ran with
├── run_context.yaml   id + name -- what it ran as
├── meta.yaml          tag, script, module -- how the attempt was staged
├── status.yaml        running | ok | failed | interrupted, and how long
├── traceback.txt      only if the run raised
├── retval.json        the return value, if there was one (.npy for an array)
├── checkpoints/       only if the run used ctx.checkpoint (see "Checkpoints")
├── metrics/           only if something used ctx.record (see "Progress and metrics")
└── out/               ctx.out -- everything the body writes
```

**runkit owns the top level; the experiment owns `out/`.** Of runkit's files,
the first three are what the run was — in principle, enough to recreate it. The
next two are how that run went. `retval.json` is the body's return value, but
runkit writes it, so it sits at the top with the rest of runkit's records.

"In principle" because the code is only referenced by `script:`, a path whose
contents can change after the run. Pinning it down (commit sha, dirty flag,
environment) is provenance, and it goes in `meta.yaml` when it's added.

```yaml
# config.yaml -- the resolved config, all three layers collapsed
lr: 0.0001
seed: 7
steps: 1000
```

```yaml
# run_context.yaml -- exactly the RunContext the body was handed
id: baseline_a3f9c1e7
name: baseline
```

```yaml
# meta.yaml -- the circumstances (an open bag; grows over time)
tag: abl-a
script: /abs/path/to/experiment.py
module: lab.baseline.experiment   # null for a plain `python experiment.py`
```

`script` is the file; `module` is its importable name, which is what re-imports
a package experiment that uses relative imports. `python -m
lab.baseline.experiment` records the real name, not `__main__`. A plain script
has no such name.

```yaml
# status.yaml -- the lifecycle; the one file that changes
status: failed
started: '2026-06-26T15:40:12'
updated: '2026-06-26T15:40:19'    # when this file was last written
host: node-17                     # the process that owns the run:
pid: 48213                        #   a pid means something only on its host
progress: 3200000                 # the body's last ctx.progress (null if never called)
total: 10000000
checkpoint: checkpoints/best      # the latest complete checkpoint, relative to the run dir
ended: '2026-06-26T15:40:19'
duration_s: 6.83
error: 'ValueError: bad shape'
```

The run dir itself is stored in none of them: it is the directory the files
sit in.

`status.yaml` is written `running` before the body starts and rewritten with the
verdict when it ends, which makes three states honest rather than two — a run
still sitting at `running` with no process behind it was killed hard (`kill -9`,
OOM, the node went away), and nothing that runs at the end of a run can report
that about itself. `host` and `pid` are what make that checkable: on the same
host, a `running` run whose pid is gone was killed (they are also what stopping a
detached run will need). Every runkit yaml is written atomically — to a temp
file, then renamed — so a process reading `status.yaml` mid-run sees the old
file or the new one, never half of one. The `error` line stays a one-liner so a
root full of runs is scannable:

```bash
grep -l "status: failed" runs/*/*/status.yaml
```

The full traceback is what you actually debug from and does not belong in a
yaml, so it goes to `traceback.txt`, present only when there is one.

If the body raises, the exception propagates untouched — a failed experiment
still exits non-zero with its own traceback. runkit records the outcome; it does
not handle it. A `KeyboardInterrupt` is recorded as `interrupted` rather than
`failed`, because you stopping it is not the same fact as it breaking.

### Checkpoints

A long run saves its state along the way, so a crash or a stop does not lose
everything, and a run can be evaluated or resumed before it ends:

```python
@exp.run
def run(cfg: Config, ctx: RunContext):
    for it in ...:
        with ctx.checkpoint() as ckpt:             # checkpoints/000003/
            model.save(ckpt.dir / "model.zip")
        if ret > best:
            with ctx.checkpoint("best") as ckpt:   # checkpoints/best/
                model.save(ckpt.dir / "model.zip")
                ckpt.info["ep_return"] = ret       # saved with it
```

```python
@dataclass
class Checkpoint:
    name: str        # the folder name; runkit's counter ("000003") when none is given
    dir: Path        # {run dir}/checkpoints/<name>/
    index: int       # runkit's counter: the order of checkpoints, whatever their names
    info: dict       # yours; saved to checkpoint.yaml when the block exits
```

`ctx.checkpoint(name=None)` is the only way to make one. The name is free-form —
the counter by default, or what suits the experiment (`f"{step}"`, `"best"`) —
and there is no step argument: not every experiment has a step, and a step is
one possible name.

```
runs/test_policy/2026-09-25_10-02-11_a3f9c1e7/
├── ...
└── checkpoints/
    ├── 000001/
    ├── 000002/
    ├── best/
    │   ├── model.zip          the body's: written into ckpt.dir
    │   └── checkpoint.yaml    runkit's: written when the block exits
    └── latest -> best         runkit's: the highest index
```

```yaml
# checkpoints/best/checkpoint.yaml
index: 3                          # runkit's counter: the order; latest = highest
name: best
time: '2026-09-25T13:47:40'       # when it finished writing
elapsed_s: 13529.4                # since the run started
run: test_policy_a3f9c1e7         # which run it came from, if the folder travels
info: {ep_return: 20.7}           # ckpt.info (numpy values made plain)
```

Checkpoints sit at the top level, not under `out/`: runkit owns the structure —
names, `checkpoint.yaml`, `latest` — and the body owns the files inside each
`ckpt.dir`, the way it owns `ctx.out`. So `ctx.checkpoint` stays optional: an
experiment that would rather manage its own checkpoints writes them under `out/`,
and runkit never touches them.

- **Complete means recorded.** The block writes into a hidden staging folder
  (`checkpoints/.best.staging-3/` — that is `ckpt.dir` during the block); when
  it exits cleanly, `checkpoint.yaml` is written and the folder is swapped into
  place. If the block raises, nothing is recorded, nothing is replaced, and the
  exception propagates. A killed save leaves at most a dot-folder, and a folder
  without `checkpoint.yaml` is not a checkpoint.
- **A repeated name replaces** the earlier checkpoint and takes a new `index` —
  what `best` wants. The old one is removed only once the new one is in place.
- **Order is `index`, never the name.** Names sort badly as text (`500000` after
  `3000000`) or not at all (`best`), so "latest" is the highest index.
- **Only the live run can checkpoint.** `ctx.live` is true in the context the
  run wrapper hands the body, while the body runs; a context rebuilt from disk
  (`load_run`, `eval`, `viz`) or a finished run's raises on `ctx.checkpoint`.
  `live` is about this process, not the run: it is not written to
  `run_context.yaml` and not part of comparing contexts.
- **`status.yaml` names the latest one** as a path relative to the run dir —
  `checkpoint: checkpoints/best` — written as soon as a checkpoint completes
  (not throttled like progress) and kept in the final write, so a failed run
  says what it can be resumed from: `run_dir / status["checkpoint"]`. Only the
  path: index, time and info stay in that checkpoint's own `checkpoint.yaml`.
- **Reading back** works on any context: `ctx.checkpoints()`, or
  `load_checkpoints(run_dir)`, gives the complete ones as `Checkpoint`s, oldest
  first — for an `eval` to pick one, or a run to resume from.

### Progress and metrics

Two more things a live run can report while it goes — how far along it is, and
the numbers it produces:

```python
@exp.run
def run(cfg: Config, ctx: RunContext):
    ctx.progress(total=cfg.steps)                   # status.yaml: progress 0 of 10M
    for it in ...:
        ctx.record(it=it, steps=n, ep_return=ret)   # metrics/run.jsonl
        ctx.progress(n)                             # the total is remembered
```

**`ctx.progress(n=None, /, *, total=None)`** writes `progress` and `total` into
`status.yaml`, flat, next to `status` and `updated`. `n` is any count — env
steps, iterations, candidates tried — since not every experiment has a "step";
a bare percentage is `ctx.progress(0.32, total=1)`. `total` is sticky: set it
once, and later calls pass only `n`; it stays null for an open-ended run. It is
keyword-only, so `ctx.progress(5, 10)` cannot be misread. Writes happen at most
every two seconds (or when `total` changes), so calling it every iteration is
fine, and the run's final `status.yaml` keeps the last values — a failed run
shows how far it got.

**`ctx.record(stream=None, /, **values)`** appends one line to
`metrics/<stream>.jsonl`, with `time` and `elapsed_s` added by runkit:

```
# metrics/run.jsonl
{"time": "2026-09-26T13:47:40.112", "elapsed_s": 13529.4, "it": 781, "steps": 3200000, "ep_return": 20.7}
```

- **Named `record`**, not `log`: "log" is the run's captured stdout (planned)
  and python's `logging`, both text.
- **A folder of streams**, runkit-owned, next to `checkpoints/` and `out/`. The
  run records to `run`; anything else names its stream — `eval` records with
  `ctx.record("eval", ...)` to `metrics/eval.jsonl`, so its numbers never mix
  into the training series. `elapsed_s` there counts from when eval opened the
  run.
- **The stream is positional-only** (`/`), so a metric that happens to be called
  `stream` is just a value. `time` and `elapsed_s` are runkit's and refused as
  keys.
- **JSON Lines**: calls may carry different keys (CSV needs fixed columns), an
  append is crash-safe (a killed run loses at most its last, torn line, which
  the reader skips), and it reads back with `load_metrics(run_dir, "run")` or
  `pandas.read_json(path, lines=True)`. numpy values are written as plain ones.
- **No step argument**: the experiment's x axis (`it`, `steps`, `epoch`) is just
  another key.
- **Separate from progress**: progress is one current position, overwritten;
  metrics are the whole history, appended. `record` does not move progress.

What a context may write follows from `ctx.live` — true only in the body of a
running run:

| call | live (the run) | not live (eval, viz, `load_run`, a finished run) |
| ---- | -------------- | ------------------------------------------------ |
| `ctx.progress(...)` | yes | error |
| `ctx.checkpoint(...)` | yes | error |
| `ctx.record(**values)` | stream `run` | error: name a stream |
| `ctx.record("eval", **values)` | yes | yes |
| `ctx.record("run", **values)` | yes | error: `run` is the run's own |

Why standardize metrics at all: they are the one output a generic tool can read
without knowing the experiment — comparing a sweep, or "final return per run"
in a future `runkit ls` — where otherwise each experiment writes its own format.

### What you get back

Calling the experiment from python — a sweep, a notebook — returns a `Run`:

```python
from experiment import Config, run

runs = [run(Config(lr=lr), tag=f"lr{lr}") for lr in (1e-3, 3e-4, 1e-4)]

[(r.config.lr, r.retval["loss"]) for r in runs]   # pair each config with its result
[r.context.dir for r in runs]                     # where they landed
```

The same staging flags are available as keyword arguments:
`run(cfg, tag=..., root=...)`. The other roles are callable the same way —
`show(run=None, *, root=...)` picks and opens a run and returns what the body
returns — and a run dir opens without any role:

```python
from runkit import load_run

r = load_run("runs/baseline/latest", Config)   # the same Run, rebuilt from disk
```

---

## The two objects

```python
@dataclass
class RunContext:            # what the *body* is handed
    dir: Path                # the run dir; its top level is runkit's records
    id: str                  # unique run id, "{name}_{hex8}"
    name: str | None = None  # the experiment's identity, from @experiment

    @property
    def out(self) -> Path:   # {dir}/out; everything the experiment writes goes here
        return self.dir / "out"


@dataclass
class Run:                   # what the *caller* gets back
    config: object           # what it ran with
    context: RunContext      # what it ran as
    retval: object = None    # whatever the body returned
```

They are duals: `RunContext` goes in, `Run` comes out. `Run` has one field per
file in the run dir — `config` ↔ `config.yaml`, `context` ↔ `run_context.yaml`,
`retval` ↔ `retval.json` — so it is the in-memory image of a run dir.

Three deliberate choices:

- **`id` and `name` are fields, not derived.** Both are recoverable by parsing
  (`id` from the dir name, `name` from `id`), but a helper handed only a `ctx`
  should not have to do string surgery to know which experiment it is in.
- **`config` is carried on `Run` even though a python caller passed it in**,
  because the caller does not always have it: from the command line it is runkit
  that builds the `Config`. And pairing each config with its result is what
  comparing a sweep *is*.
- **`retval` is held in memory**, not left to `retval.json`. That dump is
  best-effort, so a value that will not serialize lives only here.

There is deliberately no `status` on `Run`: a failed run raises, so a caller
holding a `Run` always has one that finished. Status is written to disk for
*other* processes to read, not for the caller who was there. The one exception
is `load_run`, which opens a run dir whatever its status; there, check
`status.yaml` (open question: whether a loaded `Run` should carry it).

`tag` is likewise absent from `RunContext`. It stages an attempt; it is not the
run's identity. It lives in `meta.yaml`.

---

## How it works

The call at the bottom of the script, `exp.main()` (or `main(run)`), is the
handoff into runkit. It reads an optional verb off the front of argv and passes
the rest to that verb. For `run`:

1. **argv becomes a `Config`** — the two namespaces are split, the yaml (if any)
   is loaded, `key=value` is merged on top, and the dataclass is built.
2. **the wrapper creates the run** — a fresh run dir and the `RunContext`,
   frozen into `config.yaml` / `run_context.yaml` / `meta.yaml`, `status.yaml`
   set to `running` — then calls your body. From here the body owns `ctx.out`;
   runkit writes nothing more until it returns.
3. **the wrapper closes the run** — `status.yaml` gets the verdict, a non-`None`
   return value is dumped to `retval.{json|npy}`, and the caller gets a
   `Run`.

Config resolution belongs entirely to step 1. The decorator and its wrapper
never see a yaml, a partial dict, or a precedence rule — they are handed a
finished `Config`. Which is why calling a decorated `run()` from python has no
layering at all: you build the `Config` yourself and pass it.

Both of the outcome writers are best-effort, for a sharper reason than the
retval dump: they run with the experiment's exception in flight, and must never
replace it with one of runkit's own.

For `eval` and `viz` there is nothing to create: argv names a run (`RUN`), the
wrapper picks the run dir (`select_run`), rebuilds its `Run` (`load_run`) and
calls the body with its config and context. Nothing is recorded; the body's
output goes under `ctx.out`.

### Run dir naming

```
{root}/{name}/{date}_{time}_{hex8}[_{tag}]/
{root}/{name}/latest -> the latest started run dir
```

```
runs/
└── baseline/
    ├── 2026-06-26_15-40-12_a3f9c1e7_abl-a/
    ├── 2026-06-26_15-40-12_0e4fb4b9_abl-b-longer-label/
    ├── 2026-06-27_09-02-55_77c1d2aa/
    └── latest -> 2026-06-27_09-02-55_77c1d2aa
```

Runs are grouped by experiment: `ls runs/baseline` is one experiment's history,
and anything that compares runs works on one folder. There is no date level
below that — the date leads the dir name, so a glob already is a date folder
(`ls runs/baseline/2026-06-*`), and a single run dir copied elsewhere keeps its
date. What will eventually produce too many runs for one folder is sweeps, and
those get grouped by sweep, not by day.

`{time}` is `HH-MM-SS`, so runs started in the same minute still sort in the
order they started (within one second, runkit breaks the tie on when each run's
`run_context.yaml` was written); no counter, which would need a scan of the
folder at creation and races between parallel launches. `{hex8}` is the run
id's hex, so the directory never collides, and the id is in the path — split
across it: the experiment folder is the name, the dir's `{hex8}` the rest.

```
runs/baseline/2026-06-26_15-40-12_a3f9c1e7_abl-a/
     ^^^^^^^^                     ^^^^^^^^
     name                         hex8          -> id = baseline_a3f9c1e7
```

Everything fixed-width comes first and the one free-form part last:
`{date}_{time}_{hex8}` is always 28 characters, and the tag — dropped when unset,
underscores and all — is the remainder. So the hex sits in the same column of
every `ls`, where it is easy to find and copy (`viz a3f9`), a long tag only
lengthens its own line, and the name parses without guessing. `run_context.yaml`
and `meta.yaml` stay the source of truth all the same. Built in one place
(`_resolve_dir`) so the scheme is easy to change.

`latest` is a shortcut for people — `cd runs/baseline/latest`, `tail -f
runs/baseline/latest/...` — not a record. It points at the latest *started*
run, so while a run is going, or after one fails, that is where it points. It is
a relative link (it survives moving the root), swapped in atomically (parallel
starts cannot break it), and best-effort: if it cannot be made, or goes stale
because its run was deleted, nothing is harmed. Nothing in runkit reads it; a
tool that needs "the latest run" (e.g. the latest `ok` one) computes it from the
dirs.

### The `Experiment`

```python
exp = Experiment("baseline")
```

Shaped like a `typer.Typer` app: one object, functions registered on it by
decorator, and `exp.main()` as the entry point. `name` is its only argument,
and it is required. It describes the *experiment* — it travels with the code —
so it takes only definition-level arguments. Everything that stages an attempt
is a flag instead.

Every role body has the same `(cfg, ctx)` signature; the decorator is what
changes the calling convention:

| decorator | body you write | decorated callable |
| --------- | -------------- | ------------------ |
| `@exp.run` | `run(cfg, ctx)` | `run(cfg, *, tag=..., root=...) -> Run` |
| `@exp.eval` | `evaluate(cfg, ctx)` | `evaluate(run=None, *, root=...)` |
| `@exp.viz` | `show(cfg, ctx)` | `show(run=None, *, root=...)` |

`main(run, eval=..., viz=...)` is the same dispatcher as a function: it picks up
whatever is registered on `run`'s experiment, and `eval=` / `viz=` take plain
`(cfg, ctx)` bodies.

---

## Open

- **retval dispatch.** `.npy` for an array, `.json` otherwise. Widen it (`.npz`
  for a dict of arrays?) and keep it best-effort, so a return value that will
  not serialize never fails a run that already did its work.
- **`script:` is absolute**, so a run dir stops resolving if the repo moves.
  Whatever reads it should take an explicit override and treat the recorded
  value as a default. `module:`, where there is one, does not have that
  problem -- it resolves wherever the package is importable.
- **provenance.** `meta.yaml` is the open bag it goes in: git sha, dirty flag,
  host, python version. A dirty-git gate would sit alongside it.
- **captured stdout.** For a run that died on a cluster three days ago, the log
  is often the only thing you want, and the run dir has no equivalent today.
- **the overrides, separately.** `config.yaml` records what the values *were*,
  not which came from a file and which from the command line.
- **`config=` on the decorator.** A config the experiment ships with, as a path
  or a factory, loaded beneath `config.yaml`. Note the `Config` dataclass
  already *is* the shipped default, so this needs a real motivating case first.
  If it happens it must be a stored attribute read during resolution, never a
  wrapper argument.
