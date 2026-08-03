# Experiment runner — design

A lighter, crisper take on the experiment runner, trimmed to what we actually
need. We add features when a real need shows up, not before. (Provenance, the
dirty-gate, and a context yaml are out of scope for this first pass — only config
yamls are loaded.)

## Desired experience and usage

An experiment is a `Config` + a `run(cfg, ctx)`. The `@experiment` decorator
makes the script callable and hands `run` a prepared `RunContext`.

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


if __name__ == "__main__":
    main(run)
```

`RunContext` is exactly two fields:

```python
@dataclass
class RunContext:
    out: Path   # the run dir; everything the experiment writes goes here
    id: str     # stable unique run id, independent of `out` (for search)
```

What happens:
- before `run`: create a fresh run dir (`ctx.out`) and dump the resolved config into it.
- during: the body writes whatever it wants under `ctx.out` (e.g. `ctx.out / "checkpoints"`).
- after `run`: if it returns non-`None`, best-effort dump the value to
  `ctx.out / "results" / "retval.{json|npy|npz}"` (format chosen by type); `None`
  stores nothing. TODO: settle the type→format dispatch, and keep it best-effort
  so a non-serializable return never fails an otherwise-good run.

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
--dry-run       resolve config, print, and exit (no run dir)
```

`name` is decorator-only (the experiment's identity, no CLI override). `id` is
the auto uid `{name}_{hex8}`, not settable.

## Run dir naming

default `--out`:

```
{runs_dir}/{name}_{tag}_{date}_{time}_{hex8}/
```

example: `runs/baseline_ablation-a_2026-06-26_15-40-50_a3f9c1e7/`

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

Everything else — `tag`, `runs_dir`, `out` (staging) and `--dry-run` (action) —
is a flag, not a decorator arg. For now the decorator is just `name`; the only
planned overlap with flags is `config` (below).

TODO — `config=` decorator default: a config the experiment ships with, given
as a path (e.g. `exp:default.yaml`) or a factory `Callable[[], Config]`. Loaded
as the base config layer; `--config` and `key=value` still override it.
