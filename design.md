# runkit — design

Lightweight, reproducible experiment runs. You write a `Config` and a
`run(cfg, ctx)`; runkit turns the script into a CLI, gives every run a fresh
self-describing directory, and records what the run was and how it went.

It is deliberately small. Features get added when a real need shows up, not
before — there is no config composition, no server, no database, and no tracking
UI. The run directory is the whole interface.

Designs that are proposed but not built live in `proposals.md`.

---

## Usage

### An experiment

An experiment is a python file with a `Config` dataclass and a `run(cfg, ctx)`:

```python
# experiment.py
from dataclasses import dataclass
from runkit import RunContext, experiment, main


@dataclass
class Config:
    lr: float = 3e-4
    seed: int = 1
    steps: int = 1000


@experiment(name="baseline")
def run(cfg: Config, ctx: RunContext):
    ckpt = ctx.out / "checkpoints"       # ctx.out:  {run dir}/out -- write everything here
    ckpt.mkdir()                         # ctx.id:   unique run id, "baseline_a3f9c1e7"
    ...                                  # ctx.name: "baseline"
    return {"loss": 0.31}                # optional -> retval.json


if __name__ == "__main__":
    main(run)
```

Two things are required and checked: `Config` must be a dataclass, and the
`cfg` parameter must carry it as a type annotation — that annotation is how
runkit knows what to build from the command line.

### Running it

```bash
python experiment.py                                  # dataclass defaults
python experiment.py lr=1e-4 seed=7                   # + overrides
python experiment.py config.yaml                      # + a config yaml
python experiment.py config.yaml lr=1e-4 --tag=abl-a  # + a variant label
python experiment.py --help                           # fields and flags for this experiment
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
--root=DIR       where run dirs are created (default: ./runs)
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

### What lands on disk

```
runs/baseline/2026-06-26_15-40-12_abl-a_a3f9c1e7/
├── config.yaml        the resolved config -- what it ran with
├── run_context.yaml   id + name -- what it ran as
├── meta.yaml          tag, script -- how the attempt was staged
├── status.yaml        running | ok | failed | interrupted, and how long
├── traceback.txt      only if the run raised
├── retval.json        the return value, if there was one (.npy for an array)
└── out/               ctx.out -- everything the body writes
    └── checkpoints/
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
```

```yaml
# status.yaml -- the lifecycle; the one file that changes
status: failed
started: '2026-06-26T15:40:12'
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
that about itself. The `error` line stays a one-liner so a root full of runs is
scannable:

```bash
grep -l "status: failed" runs/*/*/status.yaml
```

The full traceback is what you actually debug from and does not belong in a
yaml, so it goes to `traceback.txt`, present only when there is one.

If the body raises, the exception propagates untouched — a failed experiment
still exits non-zero with its own traceback. runkit records the outcome; it does
not handle it. A `KeyboardInterrupt` is recorded as `interrupted` rather than
`failed`, because you stopping it is not the same fact as it breaking.

### What you get back

Calling the experiment from python — a sweep, a notebook — returns a `Run`:

```python
from experiment import Config, run

runs = [run(Config(lr=lr), tag=f"lr{lr}") for lr in (1e-3, 3e-4, 1e-4)]

[(r.config.lr, r.retval["loss"]) for r in runs]   # pair each config with its result
[r.context.dir for r in runs]                     # where they landed
```

The same staging flags are available as keyword arguments:
`run(cfg, tag=..., root=...)`.

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
*other* processes to read, not for the caller who was there.

`tag` is likewise absent from `RunContext`. It stages an attempt; it is not the
run's identity. It lives in `meta.yaml`.

---

## How it works

The call at the bottom of the script, `main(run)`, is the handoff into runkit:

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

### Run dir naming

```
{root}/{name}/{date}_{time}[_{tag}]_{hex8}/
{root}/{name}/latest -> the latest started run dir
```

```
runs/
└── baseline/
    ├── 2026-06-26_15-40-12_abl-a_a3f9c1e7/
    ├── 2026-06-26_15-40-12_abl-b_0e4fb4b9/
    ├── 2026-06-27_09-02-55_77c1d2aa/
    └── latest -> 2026-06-27_09-02-55_77c1d2aa
```

Runs are grouped by experiment: `ls runs/baseline` is one experiment's history,
and anything that compares runs works on one folder. There is no date level
below that — the date leads the dir name, so a glob already is a date folder
(`ls runs/baseline/2026-06-*`), and a single run dir copied elsewhere keeps its
date. What will eventually produce too many runs for one folder is sweeps, and
those get grouped by sweep, not by day.

`{time}` is `HH-MM-SS`, so runs started in the same minute (any sweep) still
sort in the order they started; no counter, which would need a scan of the
folder at creation and races between parallel launches. `{tag}` is dropped when
unset. `{hex8}` is the run id's hex (`id = {name}_{hex8}`), so the directory is
self-identifying and never collides. Built in one place (`_resolve_dir`) so the
scheme is easy to change.

With `name` in the folder, the dir name parses: date and time have fixed shapes,
`{hex8}` is last, and the tag — underscores and all — is whatever sits between.
`run_context.yaml` and `meta.yaml` stay the source of truth all the same.

`latest` is a shortcut for people — `cd runs/baseline/latest`, `tail -f
runs/baseline/latest/...` — not a record. It points at the latest *started*
run, so while a run is going, or after one fails, that is where it points. It is
a relative link (it survives moving the root), swapped in atomically (parallel
starts cannot break it), and best-effort: if it cannot be made, or goes stale
because its run was deleted, nothing is harmed. Nothing in runkit reads it; a
tool that needs "the latest run" (e.g. the latest `ok` one) computes it from the
dirs.

### The decorator

```python
@experiment(name="baseline")
```

`name` is the only argument, and it is required. The decorator describes the
*experiment* — it travels with the code — so it takes only definition-level
arguments. Everything that stages an attempt is a flag instead.

---

## Open

- **retval dispatch.** `.npy` for an array, `.json` otherwise. Widen it (`.npz`
  for a dict of arrays?) and keep it best-effort, so a return value that will
  not serialize never fails a run that already did its work.
- **`script:` is absolute**, so a run dir stops resolving if the repo moves.
  Whatever reads it should take an explicit override and treat the recorded
  value as a default.
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
