# Experiment runner — design

A lighter, crisper take on the experiment runner, trimmed to what we actually
need. We add features when a real need shows up, not before.

(Provenance, the dirty-gate, and a context yaml are out of scope for this first pass — only config
yamls are loaded. Both are specced under *Visualization* below, as
`run_context.yaml` and `meta.yaml`, which is what finally needs them.)

## Desired experience and usage

An experiment is a python file with a `Config` class and a `run(cfg, ctx)` method.
The `@experiment` decorator makes the script callable and hands `run` a prepared
`RunContext` containing the path to a run directory and a run id.

```python
# experiment.py
from dataclasses import dataclass
from runkit import experiment, RunContext, main


@dataclass
class Config:
    ... 


@experiment(name="baseline")
def run(cfg: Config, ctx: RunContext):
    ckpt = ctx.out / "checkpoints"   # ctx.out: the run dir, everything writes here
    ...                              # ctx.id:  stable unique run id, e.g. "baseline_a3f9c1e7"
                                     # ctx.name: experiment name, e.g. "baseline"


if __name__ == "__main__":
    main(run)
```

`RunContext` is a small dataclass containing run-relevant information that the
experiment needs, e.g. the path to a run dir to store results.

```python
@dataclass
class RunContext:
    out: Path                # the run dir; everything the experiment writes goes here
    id: str                  # stable unique run id, independent of `out` (for search)
    name: str | None = None  # the experiment's identity, as given to @experiment
```

`Run` is the mirror image: `RunContext` is what the body is handed, `Run` is what
the *caller* gets back, so a script driving experiments knows where they landed.

```python
@dataclass
class Run:
    context: RunContext
    retval: object = None    # whatever the body returned
```

```python
runs = [run(Config(lr=lr), tag=f"lr{lr}") for lr in (1e-3, 3e-4)]
[r.context.out for r in runs]        # where they landed
```

`retval` is held in memory rather than left to `results/retval.json`, because
that dump is best-effort — a value that will not serialize lives only here. And
there is deliberately no `status` field: a failed run raises, so a caller holding
a `Run` always has one that finished.

You can run the experiment file from the command line as follows:

```bash
python experiment.py                                    # dataclass defaults
python experiment.py lr=3e-4 seed=7                     # + overrides
python experiment.py config.yaml                        # + a config yaml
python experiment.py config.yaml lr=3e-4 --tag=ablation-a
```

It will create a self-contained run directory describing the run:

```
{run_dir}/
├── config.yaml        the resolved config -- what ran
├── run_context.yaml   id + name -- the ctx the body was handed
├── meta.yaml          tag, script -- how it was staged
├── status.yaml        running | ok | failed | interrupted, and how long
├── checkpoints/       whatever the body wrote
└── results/
    └── retval.json    the return value, if it returned one
```

What happens, and who does it. The CLI call is what starts it — `main(run)` at
the bottom of the script hands argv to runkit:

1. **`autocli` resolves the config** — defaults, then the yaml, then `key=value`
   (see *CLI*). The wrapper is handed a finished `Config`.
2. **the wrapper creates the run** — a fresh run dir and the `RunContext`,
   frozen into `config.yaml` / `run_context.yaml` / `meta.yaml` — then calls
   `run(cfg, ctx)`. From here the body owns `ctx.out`; runkit writes nothing
   more until it returns.
3. **the wrapper dumps the return value** — non-`None` to
   `ctx.out / "results" / "retval.{json|npy}"`, format chosen by type; `None`
   stores nothing. Best-effort: a value that will not serialize is skipped with
   a warning rather than failing a run that has already done its work. The
   caller gets back a `Run` — the `RunContext` plus that return value.

TODO — settle the type→format dispatch (`.npz` for a dict of arrays?).

If `run` raises, the exception propagates untouched — a failed experiment still
exits non-zero with its own traceback; runkit records the outcome, it does not
handle it. What it records is `status.yaml` (`failed`, or `interrupted` for a
Ctrl-C) plus `{out}/traceback.txt`, so a browsable `runs/` tells a crash from a
success without the terminal scrollback. Both writes are best-effort: they
happen while an exception is in flight, and must never replace it with one of
runkit's own.

## CLI

Two entry points, identical behavior:

```
python     experiment.py [config.yaml] [key=value ...] [--flag ...]
runkit run experiment.py [config.yaml] [key=value ...] [--flag ...]
```

- optional positional: a config yaml (or `--config=PATH`).
- bare `key=value`  → overrides into **config**   (the "what")
- `--flag=value`    → sets **context**            (the "how/where")

Two disjoint namespaces: flags never touch config, `key=value` never touches context.

### Config resolution is a CLI concern

Three layers, last wins:

```
dataclass defaults  →  config.yaml  →  key=value
```

The yaml is optional, and so is any key in it — a partial config is normal, and
most runs pass none at all. Defaults are not a merge step: `build_cfg` passes
*only* the keys somebody actually set, so every unmentioned field falls through
to the dataclass constructor. That one property is why no yaml, a partial yaml,
and `field(default_factory=...)` all work without a special case — and why a
field with no default fails with Python's own message (`missing 1 required
positional argument: 'lr'`) rather than a runkit one.

All of this lives in `autocli`. The decorator and its wrapper never see a yaml,
a partial dict, or a precedence rule: they are handed a finished `Config`.
`--config` is consumed by `autocli` and never forwarded; only the staging flags
reach the wrapper. So calling a decorated `run()` from Python
has no layering at all — you build the `Config` yourself and pass it.

Two consequences worth stating, since they are easy to get wrong later:

- a `config=` decorator default (TODO below) must be stored as an attribute for
  `autocli` to read as a fourth layer *under* `config.yaml`. It cannot be a
  wrapper kwarg — merging a partial default there would teach `exp.py` about
  dicts and precedence and break the split above.
- anything that wants to know how a run was configured reads `{out}/config.yaml`,
  which is the resolved result, not the layers that produced it.

Config path resolution:

```
/abs/path         →  used as-is
NAME or cwd:NAME  →  relative to cwd (no prefix = cwd)
exp:NAME          →  relative to the experiment's dir
SCHEME:NAME       →  relative to $RUNKIT_PATH_<SCHEME> (user-defined base)
```

`cwd:` and `exp:` are built in. Any other scheme is user-defined: `ctk:x.yaml`
resolves against the env var `RUNKIT_PATH_CTK` (scheme uppercased). `~` in the
base is expanded. If the env var is unset the token is left as a literal path,
with a warning to catch typo'd schemes. Pairs well with `direnv` for per-project
bases (put `export RUNKIT_PATH_CTK=...` in an `.envrc`).

## Flags (context)

```
--tag=TAG       variant label, part of the run dir name
--config=PATH   config yaml to load (cwd:/exp:/abs, resolved as above)
--runs-dir=DIR  root dir for run folders (default: ./runs)
--out=DIR       use this exact run dir (collision-safe: appends a timestamp)
```

`name` is decorator-only (the experiment's identity, no CLI override). `id` is
the auto uid `{name}_{hex8}`, not settable.

## Run dir naming

default `--out`:

```
{runs_dir}/{date}_{time}_{name}_{tag}_{hex8}/
```

example: `runs/2026-06-26_15-40_baseline_ablation-a_a3f9c1e7/`

Date first so a listing sorts chronologically; `{time}` is `HH-MM`.

- `{hex8}` is the run id's hex (`id = {name}_{hex8}`), so the dir is self-identifying.
- `{tag}` is dropped when no tag is set.
- built in one place so the scheme is easy to change later (`_resolve_out`).

TODO — counter scheme (group by date, sequential within a day):

```
{runs_dir}/{date}/{name}_{tag}_{counter}_{hex8}/
```

if `{out}` already exists, increase `{counter}`.

## Decorator

```python
@experiment(name="baseline")
```

The decorator describes the *experiment* (it travels with the code), so it takes
only definition-level args — an explicit, small signature, no arbitrary kwargs.

- `name : str`  required; the experiment's identity (no CLI override)

Everything else — `tag`, `runs_dir`, `out` — is a staging flag, not a decorator
arg. For now the decorator is just `name`; the only
planned overlap with flags is `config` (below).

TODO — `config=` decorator default: a config the experiment ships with, given
as a path (e.g. `exp:default.yaml`) or a factory `Callable[[], Config]`. Loaded
as the base config layer; `--config` and `key=value` still override it.

## Visualization / analysis (proposed)

A run writes; something else has to look at what it wrote. The convention:

```
experiment.py    Config + run(cfg, ctx)          -- produces a run dir
visualize.py     visualize(cfg, ctx)             -- consumes one
```

Small experiments keep both in `experiment.py`; once the plots grow, `visualize.py`
imports the `Config` from next door. Either way the entry point is a decorated
function, so the CLI can find it the same way it finds `run`.

### Signature: the same `(cfg, ctx)` as `run`

```python
@visualization
def visualize(cfg: Config, ctx: RunContext):
    fig = plot(json.loads((ctx.out / "results" / "retval.json").read_text()))
    fig.savefig(ctx.out / "figures" / "loss.png")
```

`@experiment` creates a run dir and freezes the config into it; `@visualization`
is the inverse -- it opens an existing run dir and thaws the config back out.
Same two arguments, same `ctx.out / ...` idiom in both bodies, one place
(the decorator) that knows the on-disk layout.

As with `@experiment`, the decorator changes the calling convention: the body you
write takes `(cfg, ctx)`, the decorated callable takes a run dir.

| decorator        | body you write        | decorated callable                    |
| ---------------- | --------------------- | ------------------------------------- |
| `@experiment`    | `run(cfg, ctx)`       | `run(cfg, *, tag=..., runs_dir=..., out=..., force=...)` |
| `@visualization` | `visualize(cfg, ctx)` | `visualize(run_dir)`                  |

So the wrapper takes the run dir, reads `config.yaml` + `run_context.yaml` out of it,
and hands the body a `(cfg, ctx)` pair. That is what the CLI passes, and it makes
the function callable as-is from a notebook: `visualize("runs/2026-08-19_...")`.

The alternative -- no decorator, a plain `visualize(run_dir)` that opens the dir
itself -- is smaller, but pushes the same four lines of yaml-loading into every
analysis and leaves the body holding an untyped dict instead of its `Config`.
That is the string-surgery we already rejected for `ctx.name`. The cfg round-trip
is lossless -- `asdict` -> `config.yaml` -> `build_cfg`, with `_cast` repairing
the scalars yaml mangles (`1e-4`) -- so the body may as well get the real thing.

### Prerequisite: what a run dir records

Reconstructing `ctx` needs `id`/`name`, and today nothing records them. The dir
name can't be parsed back: `2026-08-19_11-42_example_demo_0e4fb4b9` is either
`name=example, tag=demo` or `name=example_demo, tag=None` -- `_` is both the
separator and legal inside each part.

The split: **config, run context and meta are what it takes to recreate the run;
status is how that run went.**

```
{out}/config.yaml        -> cfg      closed schema (the Config dataclass)
{out}/run_context.yaml   -> ctx      closed schema (RunContext)
{out}/meta.yaml          -> nobody   open bag: the circumstances
{out}/status.yaml        -> nobody   the lifecycle, rewritten when the run ends
{out}/traceback.txt                  written only if the run raised
```

```yaml
# {out}/run_context.yaml     -- exactly RunContext, nothing more
id:   example_0e4fb4b9
name: example
```

```yaml
# {out}/meta.yaml            -- how this attempt was staged
tag:    demo
script: /abs/path/to/experiment.py
# later: git sha, dirty flag, host, python version
```

```yaml
# {out}/status.yaml          -- running | ok | failed | interrupted
status:     failed
started:    2026-08-19T11:42:03
ended:      2026-08-19T11:42:07
duration_s: 3.71
error:      "ValueError: bad shape"
```

`out` is stored in none of them: it is the directory the files sit in.

The first two are typed round-trips -- they rebuild exactly the pair `run` was
handed, which is exactly what `@visualization` hands to `visualize`. Their shape
is pinned to a dataclass and moves only when that dataclass moves. `meta.yaml`
has no such contract: it is meant to grow (git sha, dirty flag, host, python
version) and a reader that doesn't know a key ignores it. Keeping them apart
stops the stable files inheriting the open one's churn.

That is also why `tag` lands in meta rather than in `RunContext`: it stages an
attempt, it isn't the run's identity (see *Flags* above). Recording it at all
closes a real gap -- today `tag` exists nowhere but the dir name.

`status.yaml` is separate from `meta.yaml` because it is the one file that
*changes*. It is written `running` before the body starts and rewritten with the
verdict when it ends, so flipping the outcome never rewrites the immutable
record next to it. Three states are honest rather than two: a run still sitting
at `running` with no process behind it was killed hard (`kill -9`, OOM, the node
went away), which no end-of-run hook can report about itself.

The error line in `status.yaml` is a one-liner so `runs/` stays scannable
(`grep -l "status: failed" runs/*/status.yaml`). The full traceback is what you
debug from and does not belong in a yaml, so it goes to `{out}/traceback.txt`,
present only when there is one.

TODO -- `script` is the one meta key runkit itself reads, so that `runkit viz
<run_dir>` can find the `@visualization` from the dir alone. Keep meta advisory
by making `--script` the real mechanism and the recorded value its default; that
also covers the case where the repo moved and the recorded absolute path dangles.

### Loading, and the notebook path

The plumbing is public, mirroring `init_run`:

```python
init_run(cfg, name=..., ...) -> ctx          # create a run dir
load_run(run_dir, Config)    -> (cfg, ctx)   # open one
```

The decorator just calls `load_run`, taking `Config` from the `cfg` annotation
(the `_cfg_type` trick `autocli` already uses). In a notebook, skip the decorator
and call `load_run` directly -- one line, no framework.

### CLI

```
runkit viz <run_dir> [key=value ...]        # script comes from meta.yaml
runkit viz <run_dir> --script=visualize.py  # or say it explicitly
python visualize.py <run_dir>
```

`key=value` overrides the loaded cfg the same way it overrides defaults in `run`
-- useful for plot knobs that live in the config, and harmless because
visualization doesn't write `config.yaml`.

### Where figures go

Into `{out}/figures/`, alongside `results/`, so everything about a run stays in
one directory. This does mean a finished run dir is no longer immutable; the
containment rule is that visualization writes *only* under `figures/`, and
re-running it overwrites rather than accumulates.

### Out of scope for now

Cross-run comparison (sweeps, ablations) is a different signature -- it takes
many run dirs, not one, and there is no single `cfg` to hand it. Naming it here
so the single-run form above isn't mistaken for the general answer:

```python
@comparison
def compare(runs: list[Run]): ...    # runkit compare 'runs/*_baseline_*'
```

`Run` is the type `run()` already returns, so a sweep driven from python has the
list in hand; pointed at a glob, `runkit compare` would rebuild it from disk.

Add it when a real need shows up.
