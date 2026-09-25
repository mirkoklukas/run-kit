# todo

- [x] **run dir organization** — grouping (`{root}/{name}/...`), the `latest`
  link, and the root per experiment (`experiment.toml`, `runkit root`) are done.
  Left: captured stdout as a runkit-owned file in the run dir.
- [x] **checkpoint callback and organization** — built: `with ctx.checkpoint(name=None)
  as ckpt:` -> `{run dir}/checkpoints/<name>/` plus `checkpoint.yaml` and a `latest`
  link; `ctx.checkpoints()` / `load_checkpoints` read them back; `ctx.live`. Open:
  retention, a record in `status.yaml`, eval of a not-`ok` run from its latest
  checkpoint (proposals.md, "Checkpoints: open").
- [x] **eval and visualization** — built as `@exp.eval` / `@exp.viz`. Left: eval's
  own parameters, and whether a loaded `Run` carries `status` (proposals.md,
  "Open between the proposals").
- [ ] **progress and metrics** — `ctx.live`, `ctx.progress(n=None, *, total=None)`,
  `ctx.record(stream=None, /, **values)` -> `metrics/<stream>.jsonl`. Designed;
  see proposals.md, "Live runs: progress and metrics". Built so far: `updated`,
  `host`, `pid` in `status.yaml`, atomic yaml writes.
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

## soft

- [ ] **path schemes: where they apply** — `exp:` / `cwd:` / `SCHEME:` prefixes
  resolve for `--config` and for `root` in `experiment.toml`, but not for the
  `--root` flag (a plain path, relative to cwd). Decide whether `--root` takes
  them too, and say which inputs do in design.md.
- [ ] **bad `experiment.toml` during a run** — a toml syntax error gives a clean
  message from `runkit root`, but a python traceback from a run (the ValueError
  from `settings.resolve_root` is not caught in the run wrapper / dispatcher).
- [ ] **relative paths in the banner and closing line** — both print absolute run
  dir paths, which wrap in narrow terminals; show them relative to cwd when
  that is shorter.
