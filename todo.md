# todo

- [x] **run dir organization** — grouping (`{root}/{name}/...`), the `latest`
  link, and the root per experiment (`experiment.toml`, `runkit root`) are done.
  Left: captured stdout as a runkit-owned file in the run dir.
- [ ] **checkpoint callback and organization** — versioned checkpoints inside the run
  dir (e.g. `checkpoints/<step>/`, `latest`), and a callback that writes them.
- [x] **eval and visualization** — built as `@exp.eval` / `@exp.viz`. Left: eval's
  own parameters, and whether a loaded `Run` carries `status` (proposals.md,
  "Open between the proposals").
- [ ] **Parameter sweeps** — Run multiple experiments sweeping over one or more arguments. what if we declare ranges for two params. should that be 2d sweep or 1d.

## soft

- [ ] **path schemes: where they apply** — `exp:` / `cwd:` / `SCHEME:` prefixes
  resolve for `--config` and for `root` in `experiment.toml`, but not for the
  `--root` flag (a plain path, relative to cwd). Decide whether `--root` takes
  them too, and say which inputs do in design.md.
