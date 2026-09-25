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
    for i in range(3):
        with ctx.checkpoint() as ckpt:                    # checkpoints/000001/, ...
            (ckpt.dir / "state.txt").write_text(f"seed={cfg.seed} part={i}")
            ckpt.info["part"] = i
    return {"seed": cfg.seed, "final_loss": 1.0 / (cfg.steps + 1)}   # -> {run dir}/retval.json


@exp.viz
def show(cfg: Config, ctx: RunContext):
    """The first thing to look at: what the run returned, and what it wrote."""
    print(f"{ctx.id}  (seed={cfg.seed}, steps={cfg.steps})")
    print(json.loads((ctx.dir / "retval.json").read_text()))
    for c in ctx.checkpoints():
        print(f"  checkpoint {c.name}  {c.info}")


if __name__ == "__main__":
    exp.main()
