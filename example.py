"""Minimal runkit example.

    uv run python example.py seed=7 --tag=demo      # run (the default verb)
    uv run python example.py viz                    # look at the latest run
    uv run runkit viz example.py                    # same, via runkit

Runs land in ./runs/example/<date>_<time>[_<tag>]_<hex8>/ (gitignored).
"""
import json
from dataclasses import dataclass

from runkit import Experiment, RunContext, load_metrics

exp = Experiment("example")


@dataclass
class Config:
    seed: int = 1
    lr: float = 3e-4
    steps: int = 1000


@exp.run
def run(cfg: Config, ctx: RunContext):
    print(f"[example] id={ctx.id}  dir={ctx.dir}")
    ctx.progress(total=3)                                 # status.yaml: progress / total
    for i in range(3):
        loss = 1.0 / (i + 1)
        ctx.record(part=i, loss=loss)                     # metrics/run.jsonl
        with ctx.checkpoint() as ckpt:                    # checkpoints/000001/, ...
            (ckpt.dir / "state.txt").write_text(f"seed={cfg.seed} part={i}")
            ckpt.info["loss"] = loss
        ctx.progress(i + 1)
    return {"seed": cfg.seed, "final_loss": 1.0 / (cfg.steps + 1)}   # -> {run dir}/retval.json


@exp.viz
def show(cfg: Config, ctx: RunContext):
    """The first thing to look at: what the run returned, and what it wrote."""
    print(f"{ctx.id}  (seed={cfg.seed}, steps={cfg.steps})")
    print(json.loads((ctx.dir / "retval.json").read_text()))
    for row in load_metrics(ctx.dir):
        print(f"  part {row['part']}  loss {row['loss']:.3f}")
    for c in ctx.checkpoints():
        print(f"  checkpoint {c.name}  {c.info}")


if __name__ == "__main__":
    exp.main()
