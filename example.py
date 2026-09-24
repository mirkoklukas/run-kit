"""Minimal runkit example.

    uv run python example.py seed=7 --tag=demo      # run (the default verb)
    uv run python example.py viz                    # look at the latest run
    uv run runkit viz example.py                    # same, via runkit

Runs land in ./runs/example/<date>_<time>[_<tag>]_<hex8>/ (gitignored).
"""
import json
from dataclasses import dataclass

from runkit import Experiment, RunContext

exp = Experiment("example")


@dataclass
class Config:
    seed: int = 1
    lr: float = 3e-4
    steps: int = 1000


@exp.run
def run(cfg: Config, ctx: RunContext):
    print(f"[example] id={ctx.id}  dir={ctx.dir}")
    ckpt = ctx.out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "ckpt.txt").write_text(f"seed={cfg.seed} steps={cfg.steps}")
    return {"seed": cfg.seed, "final_loss": 1.0 / (cfg.steps + 1)}   # -> {run dir}/retval.json


@exp.viz
def show(cfg: Config, ctx: RunContext):
    """The first thing to look at: what the run returned, and what it wrote."""
    print(f"{ctx.id}  (seed={cfg.seed}, steps={cfg.steps})")
    print(json.loads((ctx.dir / "retval.json").read_text()))
    for p in sorted(ctx.out.rglob("*")):
        print(" ", p.relative_to(ctx.out))


if __name__ == "__main__":
    exp.main()
