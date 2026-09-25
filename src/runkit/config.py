"""argv parsing and dataclass instantiation.

Two namespaces, strictly disjoint:
  - bare `key=value`  -> cfg overrides    (the experiment definition)
  - `--flag[=value]`  -> ctx overrides    (how this attempt is run)

Pure functions; no IO, no globals. Tested in isolation.
"""
import dataclasses
import inspect
import types
import typing


def split_argv(tokens):
    """Split tokens into (cfg_overrides, ctx_flags, positionals).

    cfg_overrides: list of "key=value" strings (the cfg layer).
    ctx_flags: dict of {flag_name: value} parsed from --flag / --flag=value /
               --flag value pairs. Bare --flag (and any `-x` short flag) becomes
               True, so a stray `-x` is rejected as a flag rather than
               mistaken for a positional.
    positionals: bare tokens with no leading dash and no '=' (e.g. a config
                 file path). Callers decide what they mean.
    """
    cfg, flags, positionals = [], {}, []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            body = t[2:]
            if "=" in body:
                k, v = body.split("=", 1)
                flags[k.replace("-", "_")] = _coerce(v)
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--") \
                    and "=" not in tokens[i + 1]:
                flags[body.replace("-", "_")] = _coerce(tokens[i + 1])
                i += 1
            else:
                flags[body.replace("-", "_")] = True
        elif len(t) > 1 and t[0] == "-" and not t[1].isdigit():
            body = t[1:]
            flags[body.replace("-", "_")] = True
        elif "=" in t:
            cfg.append(t)
        else:
            positionals.append(t)
        i += 1
    return cfg, flags, positionals


def deep_merge(base, over):
    """Recursively merge `over` into `base`; `over` wins. Returns a new dict."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(tokens):
    """Parse ["a=1", "b.c=hi"] -> {"a": 1, "b": {"c": "hi"}}.

    Dotted keys become nested dicts. Values are type-coerced.
    """
    out = {}
    for t in tokens:
        if "=" not in t:
            raise ValueError(f"override {t!r} must be key=value")
        key, raw = t.split("=", 1)
        _set_dotted(out, key.split("."), _coerce(raw))
    return out


def _set_dotted(d, path, value):
    for p in path[:-1]:
        d = d.setdefault(p, {})
        if not isinstance(d, dict):
            raise ValueError(f"override path collides with scalar at {p!r}")
    d[path[-1]] = value


def _coerce(s):
    """Cheap YAML-ish scalar coercion. No external dep."""
    if not isinstance(s, str):
        return s
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "None", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def annotated_cfg(fn):
    """The annotation on `fn`'s `cfg` parameter, or None if it has none.

    Follows `functools.wraps` to the body, and `eval_str=True` so a string
    annotation (`from __future__ import annotations`) resolves to the class.
    """
    params = inspect.signature(fn, eval_str=True).parameters
    if "cfg" not in params or params["cfg"].annotation is inspect.Parameter.empty:
        return None
    return params["cfg"].annotation


def build_cfg(cls, overrides, base=None):
    """Instantiate dataclass `cls` with `overrides` applied to defaults.

    Supports nested dataclasses: a dict value overrides fields *of the nested
    config the field would otherwise hold* -- its own default (`default_factory()`
    or `default`), or, below the top, the parent's value -- so every field the
    overrides do not mention keeps that value, not the nested class's defaults.
    A subclass default stays that subclass. Only a field with no default is
    built fresh from its annotated class.

    `base`: an instance to override instead of `cls`'s defaults (used for the
    nesting; the result is `dataclasses.replace(base, ...)`).
    """
    if base is not None:
        cls = type(base)
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    field_types = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in overrides.items():
        if k not in field_types:
            raise ValueError(
                f"unknown field {k!r} for {cls.__name__}; "
                f"known: {sorted(field_types)}")
        t = field_types[k]
        if isinstance(v, dict) and dataclasses.is_dataclass(t):
            current = getattr(base, k) if base is not None else _field_default(fields[k])
            if _is_instance(current):
                kwargs[k] = build_cfg(type(current), v, base=current)
            else:
                kwargs[k] = build_cfg(t, v)
        else:
            kwargs[k] = _cast(t, v)
    if base is not None:
        return dataclasses.replace(base, **kwargs)
    return cls(**kwargs)


def _is_instance(x):
    return dataclasses.is_dataclass(x) and not isinstance(x, type)


def _field_default(f):
    """The value a dataclass field gets when it is not passed, or None."""
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return None


def _unwrap_optional(t):
    """`Optional[X]` / `X | None` -> X; other unions and plain types pass through."""
    if typing.get_origin(t) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(t) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return t


def _cast(t, v):
    """Coerce `v` to the field's annotated scalar type `t`, when it's safe to.

    The point of failure this fixes: YAML 1.1 only reads a float when the
    mantissa has a dot, so `1e-4` loads as the *string* "1e-4". A `float`
    annotation lets us recover the intended value. Only the scalar leaf types are
    touched; containers, dataclasses, and anything we can't convert pass through
    unchanged, so a genuinely wrong value still surfaces at `cls(**kwargs)`.
    """
    t = _unwrap_optional(t)
    if v is None or not isinstance(t, type) or isinstance(v, t):
        return v
    if t is bool:
        return _coerce(v) if isinstance(v, str) else v
    if t is int:
        return _to_int(v)
    if t in (float, str):
        try:
            return t(v)
        except (TypeError, ValueError):
            return v
    return v


def _to_int(v):
    """`int(v)`, but also accept exponent strings yaml leaves as text (`1e3`).

    Such a value is read through `float` first; it's only taken as an int when it
    lands on a whole number (`1e3` -> 1000, but `1.5e0` passes through untouched).
    """
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return v
        return int(f) if f.is_integer() else v
