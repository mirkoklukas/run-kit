# todo

- [x] **run dir organization** — grouping (`{root}/{name}/...`), the `latest`
  link, and the root per experiment (`experiment.toml`, `runkit root`) are done.
  Left: captured stdout as a runkit-owned file in the run dir.
- [x] **checkpoint callback and organization** — built: `with ctx.checkpoint(name=None)
  as ckpt:` -> `{run dir}/checkpoints/<name>/` plus `checkpoint.yaml` and a `latest`
  link; `checkpoint: checkpoints/<name>` in `status.yaml`; `ctx.checkpoints()` /
  `load_checkpoints` read them back; `ctx.live`. A checkpoint's `state/` (the
  body's) and `eval/` (an eval's). Open: retention (proposals.md, "Checkpoints:
  open").
- [x] **eval and visualization** — built as `@exp.eval` / `@exp.viz`; eval takes a
  checkpoint, `evaluate(ckpt)`, writing into its `eval/`, and `runkit eval
  <checkpoint>` finds the experiment itself. Left: eval's own parameters, and
  whether a loaded `Run` carries `status` (proposals.md, "Open between the
  proposals").
- [x] **branching** — `runkit run EXP --branch RUN[:CHECKPOINT]`: a new run from a
  checkpoint, on the parent's config, lineage in `meta.yaml`; the body gets it as
  `run(cfg, ctx, branch=None)`. `runkit.record(path, ...)`, `save_config` /
  `load_config`. Open: lineage plots, an unread-branch warning, a round-trip
  check (proposals.md, "Branching and checkpoint eval: open"). control-kit's
  `test_policy` still saves at the checkpoint's top and evaluates `(cfg, ctx)`:
  moving it to `ckpt.state`, `branch` and `evaluate(ckpt)` is its own change.
- [x] **progress and metrics** — built: `ctx.progress(n=None, /, *, total=None)` ->
  `progress` / `total` in `status.yaml`; `ctx.record(stream=None, /, **values)` ->
  `metrics/<stream>.jsonl`, read with `load_metrics`; `ctx.live`; `ctx.note(message,
  **values)` -> a printed line and `metrics/notes.jsonl`. Open: marking
  eval re-runs, showing progress (proposals.md, "Progress and metrics: open").
- [ ] **detached runs** — start a run in the background, follow its log, stop it
  cleanly (`runkit run --detach experiment.py ...`; the slot between verb and
  experiment is kept free for it), with `tail` / `stop` verbs. Needs captured
  stdout (above) so there is a log to follow. See proposals.md, "What is left of
  the verbs".
- [ ] **Parameter sweeps** — Run multiple experiments sweeping over one or more arguments. what if we declare ranges for two params. should that be 2d sweep or 1d.
  Leanings so far: a grid (every combination) by default, pairing on request; an
  explicit flag so `a=1,2` is not confused with a list value (Hydra's
  `--multirun`); a sweep id in `meta.yaml`; group a sweep's runs by sweep, not by
  date, if volume demands it. From python a sweep is already a list comprehension.
- [x] **nested defaults lost on override** — a nested config whose default the parent
  sets via `field(default_factory=lambda: Model(pad_cells=1))` was rebuilt from the
  *class* defaults as soon as one nested field was overridden (`model.kp=12` ->
  `pad_cells` back to 3), silently. Fixed in c463d87: `build_cfg` applies nested
  overrides to the field's own default. control-kit's workaround subclass
  (`PolicyModelCfg(ModelCfg)` with `pad_cells: int = 1`) can go back to a plain
  `default_factory`.

- [x] **experiment.toml extras and vars** — `runkit <verb> <experiment>` reads
  them before the import and relaunches under `uv run --project <dir> --extra ...`
  once; vars alone are set in-process; `meta.yaml` records `launch`. Open: the
  `project` key, a check under `python -m` (proposals.md).

- [x] **dict fields: merge overrides into the default** -- a `dict` config field was
  *replaced* by a CLI override (`_schedule.env.w_support.start=2e6` left a dict with
  only that leaf), unlike nested dataclasses, where c463d87 applies overrides to the
  field's default. Fixed: overrides are deep-merged into the dict the field would
  otherwise hold. control-kit's `PolicyCfg._schedule` deep-merge in `__post_init__`
  can go.

## soft

- [ ] **path schemes: where they apply** — `exp:` / `cwd:` / `SCHEME:` prefixes
  resolve for `--config` and for `root` in `experiment.toml`, but not for the
  `--root` flag (a plain path, relative to cwd). Decide whether `--root` takes
  them too, and say which inputs do in design.md.
- [ ] **bad `experiment.toml` during a run** — a toml syntax error gives a clean
  message from `runkit root`, but a python traceback from a run (the ValueError
  from `settings.resolve_root` is not caught in the run wrapper / dispatcher).
- [ ] **progress in the terminal** — a completed checkpoint prints a line now;
  `ctx.progress` (3.2M / 10M) could too, as an occasional line. The body prints
  what it wants today, so it may not be needed.
