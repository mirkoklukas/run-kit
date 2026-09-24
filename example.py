"""Minimal runkit example.

    uv run python example.py seed=7 --tag=demo
    uv run runkit run example.py seed=7 --tag=demo

Runs land in ./runs/<name>[_<tag>]_<date>_<time>_<hex8>/ (gitignored).
"""
from dataclasses import dataclass

from runkit import RunContext, experiment, main


@dataclass
class Config:
    seed: int = 1
    lr: float = 3e-4
    steps: int = 1000


@experiment(name="example")
def run(cfg: Config, ctx: RunContext):
    print(f"[example] id={ctx.id}  dir={ctx.dir}")
    ckpt = ctx.out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "ckpt.txt").write_text(f"seed={cfg.seed} steps={cfg.steps}")
    return {"seed": cfg.seed, "final_loss": 1.0 / (cfg.steps + 1)}   # -> {run dir}/retval.json


if __name__ == "__main__":
    main(run)
