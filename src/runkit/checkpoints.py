"""Checkpoints: named, recorded snapshots a live run saves along the way.

    with ctx.checkpoint() as ckpt:              # {run dir}/checkpoints/000003/
        model.save(ckpt.dir / "model.zip")

    with ctx.checkpoint("best") as ckpt:        # {run dir}/checkpoints/best/
        model.save(ckpt.dir / "model.zip")
        ckpt.info["ep_return"] = ret            # saved with it

runkit owns the folder structure -- names, `checkpoint.yaml`, the `latest` link
-- and the body owns the files inside each `ckpt.dir`, as it owns `ctx.out`.

A checkpoint is complete once its `checkpoint.yaml` exists; that file is written
last, and only when the block exits cleanly. The block writes into a hidden
staging folder that is swapped into place at the end, so a failed or killed save
never replaces an earlier checkpoint of the same name, and leaves at most a
dot-folder that everything ignores. Order is runkit's `index`, never the name.
"""
import dataclasses
import datetime
import os
import pathlib
import shutil
import time

import yaml

from .utils import plain, point_latest

FOLDER = "checkpoints"
RECORD = "checkpoint.yaml"
_RESERVED = ("latest",)


@dataclasses.dataclass
class Checkpoint:
    name: str                 # the folder name; the counter ("000003") when none is given
    dir: pathlib.Path         # {run dir}/checkpoints/<name>/ (a staging folder during the block)
    index: int                # runkit's counter: the order of checkpoints, whatever their names
    info: dict = dataclasses.field(default_factory=dict)   # yours; saved to checkpoint.yaml


class _Saving:
    """The context manager `RunContext.checkpoint` returns."""

    def __init__(self, ctx, name):
        self.ctx, self.name = ctx, name

    def __enter__(self):
        ctx = self.ctx
        index = ctx._live.checkpoints + 1
        name = self.name if self.name is not None else f"{index:06d}"
        _check_name(name)
        folder = ctx.dir / FOLDER
        folder.mkdir(exist_ok=True)
        self.final = folder / name
        staging = folder / f".{name}.staging-{index}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        self.ckpt = Checkpoint(name=name, dir=staging, index=index)
        return self.ckpt

    def __exit__(self, exc_type, exc, tb):
        ckpt, staging = self.ckpt, self.ckpt.dir
        if exc_type is not None:              # not complete: nothing recorded, nothing replaced
            shutil.rmtree(staging, ignore_errors=True)
            return False
        ctx = self.ctx
        record = {
            "index": ckpt.index,
            "name": ckpt.name,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(time.monotonic() - ctx._live.t0, 3),
            "run": ctx.id,
            "info": plain(ckpt.info),
        }
        (staging / RECORD).write_text(yaml.safe_dump(record, sort_keys=False))
        _swap_in(staging, self.final)
        ctx._live.checkpoints = ckpt.index
        ckpt.dir = self.final
        point_latest(self.final)
        # status.yaml names it at once (not throttled like progress): what a
        # resume or an eval of an unfinished run would read
        ctx._live.checkpoint = f"{FOLDER}/{ckpt.name}"
        ctx._write_running()
        return False


def _check_name(name):
    if (not isinstance(name, str) or not name or name in _RESERVED
            or name.startswith(".") or "/" in name or os.sep in name):
        raise ValueError(f"bad checkpoint name {name!r}: a plain folder name, "
                         f"not starting with '.', and not {', '.join(_RESERVED)}")


def _swap_in(staging, final):
    """Put `staging` where `final` is. An old checkpoint of that name is moved
    aside first and removed only after the new one is in place."""
    old = None
    if final.exists():
        old = final.with_name(f".{final.name}.old-{os.getpid()}")
        os.replace(final, old)
    os.replace(staging, final)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def load_checkpoints(run_dir):
    """The complete checkpoints of a run, oldest first (by `index`).

    A folder counts only if its `checkpoint.yaml` is there; staging leftovers,
    the `latest` link and anything unreadable are skipped.
    """
    folder = pathlib.Path(run_dir) / FOLDER
    if not folder.is_dir():
        return []
    found = []
    for d in folder.iterdir():
        if d.name.startswith(".") or d.is_symlink() or not (d / RECORD).is_file():
            continue
        try:
            rec = yaml.safe_load((d / RECORD).read_text())
            found.append(Checkpoint(name=rec["name"], dir=d.resolve(), index=rec["index"],
                                    info=rec.get("info") or {}))
        except Exception:                                    # noqa: BLE001
            continue
    return sorted(found, key=lambda c: c.index)
