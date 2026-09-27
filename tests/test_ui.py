"""What runkit prints around a run: order, paths, the banner."""
import json
import os
import pathlib
import subprocess
import sys

from runkit import ui


def test_short_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ui.short_path(tmp_path / "runs" / "a") == os.path.join("runs", "a")   # below cwd
    home = pathlib.Path.home()
    assert ui.short_path(home / "x" / "y") in ("~/x/y", str(home / "x" / "y"))   # under home
    assert ui.short_path("/definitely/elsewhere") == "/definitely/elsewhere"


_EXP = '''
import sys
from dataclasses import dataclass
from runkit import Experiment, RunContext

exp = Experiment("order")

@dataclass
class Cfg:
    n: int = 1

@exp.run
def run(cfg: Cfg, ctx: RunContext):
    print("BODY-1")
    print("BODY-2")

if __name__ == "__main__":
    exp.main()
'''


def _run_piped(tmp_path, env=None):
    """stdout and stderr into one pipe, as in a log file: stdout is buffered."""
    (tmp_path / "exp.py").write_text(_EXP)
    return subprocess.run([sys.executable, "exp.py", "--root=runs"], cwd=tmp_path,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          env={**os.environ, "COLUMNS": "200", **(env or {})}).stdout


def test_a_log_keeps_the_order_things_happened_in(tmp_path):
    out = _run_piped(tmp_path)
    at = [out.index(s) for s in ("▶ runkit · order", "BODY-1", "BODY-2", "✓ order_")]
    assert at == sorted(at), out


def test_banner_has_no_frame_short_paths_and_the_launch(tmp_path):
    launch = {"extras": ["mjx", "sb3"], "vars": {"A": "1"}, "project": "/p"}
    out = _run_piped(tmp_path, {"RUNKIT_LAUNCH": json.dumps(launch)})
    assert "╭" not in out and "│" not in out                      # no panel
    assert " dir  runs/order/" in out                             # relative to the cwd
    assert "launch  uv · extras mjx, sb3 · vars A" in out
    assert "all 1 fields at their defaults" in out and "[dim]" not in out
    assert "→ runs/order/" in out                                 # the closing line too
