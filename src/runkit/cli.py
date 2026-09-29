"""The `runkit` console script.

Usage:
    runkit <verb> [runkit options] <experiment> [args ...]
    runkit root [FOLDER]
    runkit metrics [info|follow|plot] [PATH] [KEY ...] [options]

verbs: run, eval, viz (as registered on the experiment), root, latest

`<experiment>` is a file (`experiment.py`) or a dotted module
(`lab.rl_env.test_policy`). For eval and viz it may be left out when a run dir
or a checkpoint is named instead (`runkit eval runs/x/latest/checkpoints/best`):
the run's meta.yaml names its experiment. `args` are what `python experiment.py <verb> ...`
takes after the verb. The slot between the verb and the experiment is for
runkit's own options (none yet).

`root` and `latest` print a path, for `cd "$(runkit latest experiment.py)"`:
the experiment's runs folder ({root}/{name}) and its latest run dir. `latest`
also re-points a wrong `latest` link; `runkit latest --fix EXP` does only that. Without an
experiment, `runkit root [FOLDER]` prints the root itself, resolved from FOLDER
(default: the current one): the nearest `experiment.toml`'s `[env] root`, else
`./runs`.

`metrics` reads a run's metrics without importing its experiment:

    runkit metrics [info] [PATH] [KEY ...]    keys: rows, last, min, max, mean, std, trend
                                              (no KEY: every stream, run first)
    runkit metrics follow [PATH] [KEY ...]    rows as they are written, the run's
                                              checkpoints, and its end
    runkit metrics plot [PATH] [KEY ...]      to a PNG in the run's metrics/ (no KEY:
                                              every key, a subplot each)

PATH is a run dir, its metrics/ folder, or a stream file (metrics/eval.jsonl);
or any folder of .jsonl files, or one such file (an eval's
checkpoints/best/eval/episodes.jsonl) -- each file a stream. Default: the
current folder. KEY is `key` (of the current stream: `run`, or the
file's), `stream:key`, `loss/` or 'loss/*' (every loss/... key), or a lone `stream:` that
switches the stream for the keys after it -- or, with none after it, means all
of it: `eval: ret len`, `loss eval: ret`, `eval:`.

Options: `--x KEY` (plot, info: the x axis), `--rows A:B` (a python slice of
line numbers, negatives from the end), `--start V` / `--end V` (an inclusive
window on the x axis: a value, negative counting back from the end, or a
checkpoint name), `--out FILE` (plot), `--sort COLUMN` (info: a numeric column
by absolute value, largest first; or `key`, `rows`).
"""
import importlib
import importlib.util
import pathlib
import sys

from . import ui
from .autocli import PATH_VERBS, VERBS, dispatch
from .exp import SCRIPT_MODULE, Experiment
from .settings import resolve_root

HELP = __doc__[__doc__.index("Usage:"):].strip()


def app():
    """Console-script entry point. Returns None: whatever the verb returned (a
    `Run`, say) must not become the process's exit status."""
    main()


def main(argv=None):
    """`runkit <verb> [options] <experiment> ...` -> the verb's return value."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"runkit — lightweight, reproducible experiment runs.\n\n{HELP}")
        return
    verb, rest = argv[0], argv[1:]
    if verb == "metrics":                    # read a run dir; no experiment import
        return metrics_cmd(rest)
    if verb == "plot":
        sys.exit("`runkit plot ...` is now `runkit metrics plot [PATH] [KEY ...]`")
    if verb not in VERBS + PATH_VERBS:
        hint = ""
        if verb.endswith(".py") or "." in verb:      # the old order: runkit exp.py viz
            v = rest[0] if rest and rest[0] in VERBS + PATH_VERBS else "run"
            hint = f" -- the verb comes first: runkit {v} {verb} ..."
        sys.exit(f"unknown verb {verb!r}{hint}\n\n{HELP}")

    # runkit's own options sit between the verb and the experiment
    passed = []                              # ... some belong to the verb itself
    while rest and rest[0].startswith("-"):
        opt = rest.pop(0)
        if opt in ("-h", "--help"):
            print(HELP)
            return
        if verb == "latest" and opt == "--fix":
            passed.append(opt)
            continue
        sys.exit(f"unknown runkit option {opt!r} for {verb} "
                 f"(experiment arguments go after the experiment)")

    if verb == "root" and (not rest or pathlib.Path(rest[0]).is_dir()):
        return root_cmd(rest)
    if not rest:
        sys.exit(f"usage: runkit {verb} <experiment> ...")
    target, args = rest[0], [*rest[1:], *passed]
    if verb in ("eval", "viz") and pathlib.Path(target).is_dir():
        # a run dir or a checkpoint, and no experiment: the run says which
        target, args = _experiment_of(target), rest
    from .launch import prepare
    prepare(target, argv)                   # experiment.toml's extras / vars, before the import
    return dispatch(_find_experiment(_import(target), target), [verb, *args])


def _experiment_of(path):
    """The experiment a run dir (or a checkpoint of one) was made by, as a target
    to import: its recorded module when it can be found from here -- so its
    package's relative imports work -- else its script."""
    from .launch import experiment_file
    from .runs import run_dir_of
    from .utils import load_yaml
    run_dir = run_dir_of(path)
    if run_dir is None:
        sys.exit(f"{path} is not in a run dir (no run_context.yaml above it); "
                 f"name the experiment: runkit eval <experiment> [CHECKPOINT]")
    try:
        meta = load_yaml(run_dir / "meta.yaml")
    except ValueError:
        meta = {}
    module, script = meta.get("module"), meta.get("script")
    if module and experiment_file(module) is not None:
        return module
    if script and pathlib.Path(script).is_file():
        return script
    sys.exit(f"{ui.short_path(run_dir)}: its experiment ({module or script or 'unrecorded'}) "
             f"is not found from here; name it: runkit eval <experiment> {path}")


def root_cmd(argv):
    """`runkit root [FOLDER]`: print the root resolved from FOLDER.

    Printing is all it can do: a process cannot change its parent shell's
    directory. Only the path goes to stdout, so `cd "$(runkit root)"` is clean.
    """
    if len(argv) > 1:
        sys.exit("usage: runkit root [FOLDER | <experiment>]")
    start = pathlib.Path(argv[0]) if argv else pathlib.Path.cwd()
    try:
        root = resolve_root(start).resolve()
    except ValueError as e:
        sys.exit(str(e))
    if not root.is_dir():
        ui.warn(f"{root} does not exist yet (no runs made there)")
    print(root)
    return root


def _options(argv, allowed, usage):
    """Split `--name value` / `--name=value` options (from `allowed`) off argv.
    A value may start with '-' (`--start -1000`)."""
    opts, positional, it = {}, [], iter(argv)
    for a in it:
        if a.startswith("--") and a not in ("--help",):
            name, eq, value = a[2:].partition("=")
            if name not in allowed:
                sys.exit(f"unknown option {a!r}\n\n{usage}")
            opts[name] = value if eq else next(it, None)
            if opts[name] is None:
                sys.exit(f"--{name} needs a value\n\n{usage}")
        else:
            positional.append(a)
    return opts, positional


def _selection(opts):
    """--rows / --start / --end / --x -> keyword arguments for runkit.metrics
    (named as in python: `from` is a keyword there)."""
    return {"rows": opts.get("rows"), "start": opts.get("start"), "end": opts.get("end"),
            "x": opts.get("x")}


# -- runkit metrics [info|follow|plot] [PATH] [KEY ...] ---------------------------

METRICS_ACTIONS = ("info", "follow", "plot")
_USAGE = {
    "info": ("usage: runkit metrics [info] [PATH] [KEY ...] [--sort COLUMN] [--rows A:B] "
             "[--start V] [--end V] [--x KEY]"),
    "follow": "usage: runkit metrics follow [PATH] [KEY ...] [--rows A:B]",
    "plot": ("usage: runkit metrics plot [PATH] [KEY ...] [--x KEY] [--rows A:B] "
             "[--start V] [--end V] [--out FILE]"),
}
_OPTIONS = {"info": ("rows", "start", "end", "x", "sort"), "follow": ("rows",),
            "plot": ("x", "out", "rows", "start", "end")}


def _metrics_path(argv, usage):
    """Split off a leading PATH -- a run dir, its `metrics/` folder, or a stream
    file (`metrics/eval.jsonl`) -- else the current folder, read the same way.
    -> (run dir, the stream a file names or None, the rest of argv)."""
    if any(a in ("-h", "--help") for a in argv):
        print(usage)
        sys.exit(0)
    rest = list(argv)
    if rest and pathlib.Path(rest[0]).exists():
        target = pathlib.Path(rest.pop(0))
    else:
        target = pathlib.Path.cwd()
    stream = None
    if target.is_file():
        if target.suffix != ".jsonl":
            sys.exit(f"{target} is not a stream (a .jsonl file, e.g. a run's "
                     f"metrics/<stream>.jsonl)\n\n{usage}")
        stream, run_dir = target.stem, target.parent
    else:
        run_dir = target
    if run_dir.name == "metrics" and (run_dir.parent / "run_context.yaml").is_file():
        run_dir = run_dir.parent                 # a run's metrics/: the run
    if not (run_dir / "run_context.yaml").is_file() and not any(run_dir.glob("*.jsonl")):
        sys.exit(f"{run_dir} is not a run dir (no run_context.yaml) and holds no "
                 f".jsonl streams; pass a run dir, its metrics/ folder, or a stream "
                 f"file, e.g. runs/<name>/latest\n\n{usage}")
    return run_dir, stream, rest


def parse_keys(specs, stream=None):
    """KEY tokens -> [(stream, key)], a key of "" meaning all of that stream.

    `key` is a key of the current stream -- `run`, or the stream a file PATH
    names; `stream:key` a key of that stream, leaving the current one as it is;
    a lone `stream:` switches the current stream for the keys that follow it,
    and when none follow, means all of it. `loss/` (a group -- every `loss/...`
    key; `loss/*` means the same) is kept as given.
    The split is at the first `:` -- stream names have none -- so a key
    containing one is written with its stream: `run:a:b` is key `a:b`.
    """
    current, out, open_switch = stream or "run", [], None
    for tok in specs:
        s, sep, key = tok.partition(":")
        if not sep:
            key = tok
        if key.endswith("/*"):                # `reward/*` is `reward/` (quote it in a shell)
            key = key[:-1]
        if sep and not key:                  # `eval:` -- switch
            if open_switch:
                out.append((open_switch, ""))
            current = open_switch = s or "run"
        elif sep:                            # `eval:ret` -- just this key
            out.append((s or "run", key))
        else:                                # `ret` -- the current stream
            out.append((current, key))
            open_switch = None
    if open_switch:
        out.append((open_switch, ""))
    if not out and stream:
        out = [(stream, "")]
    return out


def metrics_cmd(argv):
    """`runkit metrics [info|follow|plot] [PATH] [KEY ...] [options]`."""
    if "-f" in argv:
        sys.exit("`runkit metrics -f` is now `runkit metrics follow [PATH] [KEY ...]`")
    action = "info"
    if argv and argv[0] in METRICS_ACTIONS:
        action, argv = argv[0], argv[1:]
    usage = _USAGE[action]
    opts, positional = _options(argv, _OPTIONS[action], usage)
    run_dir, stream, specs = _metrics_path(positional, usage)
    series = parse_keys(specs, stream)
    return {"info": _metrics_info, "follow": _metrics_follow,
            "plot": _metrics_plot}[action](run_dir, series, opts, usage)


def _stream_keys(run_dir, stream, keys, usage):
    """Expand a stream's requested keys: "" -> all of them (None), `loss/` ->
    its group; a key the stream does not have is an error."""
    from .metrics import _columns, load_metrics
    if "" in keys:
        return None
    have = [k for k in _columns(load_metrics(run_dir, stream)) if k != "_line"]
    out = []
    for k in keys:
        found = [h for h in have if h.startswith(k)] if k.endswith("/") else \
                [k] if k in have else []
        if not found:
            sys.exit(f"no key {k!r}{'...' if k.endswith('/') else ''} in stream {stream!r} "
                     f"(keys: {', '.join(have) or 'none'})\n\n{usage}")
        out += [f for f in found if f not in out]
    return out


def _by_stream(series):
    grouped = {}
    for stream, key in series:
        grouped.setdefault(stream, []).append(key)
    return grouped


def _metrics_info(run_dir, series, opts, usage):
    """Tables of keys: every stream (`run` first) with nothing named, else the
    streams named, each with its keys."""
    from .metrics import stream_folder, stream_label, streams, summarize_metrics
    have = streams(run_dir)
    wanted = _by_stream(series) or {s: [""] for s in have}
    order = sorted(wanted, key=lambda s: (s != "run", s))
    ui.out.print(f"[bold]{run_dir.resolve().name}[/bold]")
    if not have:
        ui.out.print("[dim]no metrics yet (nothing recorded with ctx.record)[/dim]")
        return {}
    missing = [s for s in order if s not in have]
    if missing:
        sys.exit(f"no metrics stream {missing[0]!r} in {stream_folder(run_dir)} "
                 f"(streams: {', '.join(have)})")
    tables, failed = {}, {}
    for s in order:
        keys = _stream_keys(run_dir, s, wanted[s], usage)
        try:
            rows = summarize_metrics(run_dir, s, **_selection(opts))
        except ValueError as e:              # e.g. a checkpoint that did not count this stream
            failed[s] = e
            continue
        tables[s] = rows if keys is None else [r for r in rows if r["key"] in keys]
    if failed and not tables:                # a bad selection: nothing to show at all
        sys.exit(str(next(iter(failed.values()))))
    if opts.get("sort"):
        tables = {s: _sorted_rows(rows, opts["sort"], usage) for s, rows in tables.items()}
    for s in order:
        if s in tables:
            _print_stream(stream_label(run_dir, s), tables[s])
        else:
            ui.out.print(f"\n{stream_label(run_dir, s)}  [dim]{failed[s]}[/dim]")
    return tables


def _print_stream(label, rows):
    """One stream's keys as a table under a `metrics/<stream>.jsonl  N rows` line."""
    from rich.markup import escape
    n = max((r["rows"] for r in rows), default=0)
    ui.out.print(f"\n{label}  {n} rows")
    names = ("key", "rows", "last", "min", "max", "mean", "std", "trend")
    cells = [[r["key"], str(r["rows"]), _fmt(r["last"]), _fmt(r["min"]), _fmt(r["max"]),
              _fmt(r["mean"]), _fmt(r["std"]), _trend_text(r)] for r in rows]
    # laid out by hand, every column as wide as its widest cell, and printed
    # uncropped: a key or a value is never cut -- a table wider than the
    # terminal wraps there instead
    widths = [max([len(n), *(len(c[i]) for c in cells)]) for i, n in enumerate(names)]
    def line(values):
        return "  " + "  ".join(v.ljust(w) if i == 0 else v.rjust(w)
                                for i, (v, w) in enumerate(zip(values, widths))).rstrip()
    ui.out.print(f"[bold]{escape(line(names))}[/bold]", crop=False, soft_wrap=True,
                 highlight=False)
    for c in cells:
        ui.out.print(line(c), markup=False, crop=False, soft_wrap=True, highlight=False)


def _fmt(v, sign=False):
    """One number style for the table: whole numbers in full (60000, not
    6e+04), others of 10000 or more to the unit (12346), smaller ones to 4
    significant digits (0.4123, -0.0001234)."""
    if v is None:
        return ""
    if not isinstance(v, float):
        return str(v)
    s = "+" if sign and v > 0 else "-" if v < 0 else ""
    a = abs(v)
    if a.is_integer() and a < 1e15:
        return f"{s}{int(a)}"
    if 1e4 <= a < 1e15:
        return f"{s}{round(a)}"
    return f"{s}{a:.4g}"


SORT_COLUMNS = ("key", "rows", "last", "min", "max", "mean", "std", "trend")


def _sorted_rows(rows, column, usage):
    """A stream's table rows by `column`: `key` alphabetically, `rows` most
    first, a numeric column by absolute value, largest first -- a large cost
    ranks with a large reward. Rows without a number go last."""
    if column not in SORT_COLUMNS:
        sys.exit(f"--sort takes one of {', '.join(SORT_COLUMNS)} (got {column!r})\n\n{usage}")
    if column == "key":
        return sorted(rows, key=lambda r: r["key"])
    if column == "rows":
        return sorted(rows, key=lambda r: -r["rows"])
    number = lambda r: isinstance(r.get(column), (int, float)) and not isinstance(r[column], bool)
    return (sorted([r for r in rows if number(r)], key=lambda r: -abs(r[column]))
            + [r for r in rows if not number(r)])


def _trend_text(r):
    """+0.08 ↑ / -0.02 ↓ / 0.001 →: flat when the change is under 5% of the
    key's typical size, so noise does not read as a trend. The arrow last, so
    the arrows line up at the column's right edge."""
    trend = r.get("trend")
    if trend is None:
        return ""
    scale = max(abs(r["min"]), abs(r["max"])) if r["mean"] is None else \
        max(abs(r["mean"]), r["std"] or 0.0, 1e-12)
    arrow = "→" if abs(trend) < 0.05 * scale else ("↑" if trend > 0 else "↓")
    return f"{_fmt(float(trend), sign=True)} {arrow}"


def _metrics_follow(run_dir, series, opts, usage):
    """Rows of one stream as they are written, until the run ends."""
    from .follow import follow
    from .metrics import FOLDER, parse_rows
    if not (run_dir / "run_context.yaml").is_file():
        sys.exit(f"follow needs a run dir: {run_dir} is a folder of streams\n\n{usage}")
    wanted = _by_stream(series) or {"run": [""]}
    if len(wanted) > 1:
        sys.exit(f"follow takes one stream (got {', '.join(wanted)})\n\n{usage}")
    (stream, keys), = wanted.items()
    if "" in keys:
        keys = None                          # all of the stream
    elif (run_dir / FOLDER / f"{stream}.jsonl").is_file():
        # a named key must exist; a group (`reward/`) may still be empty -- its
        # keys become columns as they are recorded -- but only while the run goes
        _stream_keys(run_dir, stream, [k for k in keys if not k.endswith("/")], usage)
        from .follow import _status
        from .metrics import _columns, load_metrics
        have = [k for k in _columns(load_metrics(run_dir, stream)) if k != "_line"]
        empty = [g for g in keys if g.endswith("/") and not any(h.startswith(g) for h in have)]
        if empty and _status(run_dir).get("status") != "running":
            sys.exit(f"no keys {', '.join(g + '...' for g in empty)} in stream {stream!r} "
                     f"(keys: {', '.join(have) or 'none'})\n\n{usage}")
        for g in empty:
            ui.line(f"[dim]no {g}... keys yet in {stream!r} -- waiting for them[/dim]")
    try:
        first = parse_rows(opts["rows"]) if opts.get("rows") else None
    except ValueError as e:
        sys.exit(str(e))
    ui.line(f"[dim]following {FOLDER}/{stream}.jsonl of {ui.short_path(run_dir)} "
            f"-- Ctrl-C stops following, not the run[/dim]")
    try:
        return follow(run_dir, stream, keys, first=first)
    except KeyboardInterrupt:
        return None


def _metrics_plot(run_dir, series, opts, usage):
    """Keys to a PNG in the run's metrics/: every key of one stream in its own
    subplot when a whole stream is asked for alone, else the keys on one axis."""
    from .metrics import _columns, load_metrics, plot_metrics
    if not series:
        ys = ["run:"]
    elif len(series) == 1 and series[0][1] == "":
        ys = [f"{series[0][0]}:"]            # the overview of one stream
    else:
        ys = []
        for stream, key in series:
            if key == "":                    # a whole stream among other keys: its keys
                cols = _columns(load_metrics(run_dir, stream))
                ys += [k if stream == "run" else f"{stream}:{k}" for k, c in cols.items()
                       if not k.startswith("_") and k != opts.get("x") and c.dtype == float]
            else:
                ys.append(key if stream == "run" else f"{stream}:{key}")
    sel = _selection(opts)
    try:
        out = plot_metrics(run_dir, ys, x=sel["x"], out=opts.get("out"), rows=sel["rows"],
                           start=sel["start"], end=sel["end"])
    except ValueError as e:
        sys.exit(str(e))
    print(out)
    return out


def _import(target):
    """Import a file-path or dotted-module target the way `python` would run it."""
    path = pathlib.Path(target)
    if target.endswith(".py") or path.is_file():
        if not path.is_file():
            sys.exit(f"experiment file not found: {target}")
        # like `python file.py`: the file's own folder is importable
        sys.path.insert(0, str(path.resolve().parent))
        spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[SCRIPT_MODULE] = mod
        spec.loader.exec_module(mod)
        return mod
    # like `python -m pkg.mod`: importable from the current directory
    sys.path.insert(0, str(pathlib.Path.cwd()))
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as e:
        sys.exit(f"cannot import {target!r} (not a file, and not a module: {e})")


def _find_experiment(mod, target):
    """The one `Experiment` in `mod` -- an `Experiment(...)` object, or the one
    behind an `@experiment`-decorated function. One experiment per file."""
    found = {}
    for value in vars(mod).values():
        exp = value if isinstance(value, Experiment) else getattr(
            value, "_runkit_experiment", None)
        if isinstance(exp, Experiment):
            found[id(exp)] = exp
    if not found:
        sys.exit(f"{target}: no experiment found (an `Experiment(...)` or an "
                 f"@experiment function at module level)")
    if len(found) > 1:
        names = sorted(e.name for e in found.values())
        sys.exit(f"{target}: {len(found)} experiments ({', '.join(names)}); "
                 f"one experiment per file")
    return next(iter(found.values()))
