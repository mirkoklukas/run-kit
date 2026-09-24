# todo

- [x] **run dir organization** — grouping (`{root}/{name}/...`) and the `latest`
  link are done. Left: the default root per experiment (e.g. via
  `experiment.toml`), and captured stdout as a runkit-owned file in the run dir.
- [ ] **checkpoint callback and organization** — versioned checkpoints inside the run
  dir (e.g. `checkpoints/<step>/`, `latest`), and a callback that writes them.
- [ ] **eval and visualization** — `@evaluation` / `@visualization`: open an existing
  run dir, write only inside it; `runkit eval <module> [run dir]`. See proposals.md,
  "Verbs, roles and `experiment.toml`" and "Open between the proposals".    
- [ ] **Parameter sweeps** — Run multiple experiments sweeping over one or more arguments. what if we declare ranges for two params. should that be 2d sweep or 1d.

## soft

- [ ] **path schemes: where they apply** — `exp:` / `cwd:` / `SCHEME:` prefixes
  resolve for `--config` only; `--root` is a plain path, relative to
  cwd. Decide which inputs take schemes (and say so in design.md). proposals.md's
  `root = "ctk:runs/rl_env"` in `experiment.toml` already assumes `--root` does.
