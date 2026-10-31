"""Component registry.

A bundle's manifest names its building blocks by ``type`` (e.g. a
``line-chunker`` preprocessor, a ``kv-cache`` autoregressive decoder). Each
type maps to one class registered here, so supporting a new kind of model is
one new class plus one ``@register`` line -- nothing else in the runtime
changes.
"""

_REGISTRY: dict[str, dict[str, type]] = {}

KINDS = ("preprocess", "decoder", "tokenizer", "postprocess", "task")


def register(kind: str, name: str):
    if kind not in KINDS:
        raise ValueError(f"Unknown component kind '{kind}'. Known: {KINDS}")

    def decorator(cls):
        table = _REGISTRY.setdefault(kind, {})
        if name in table and table[name] is not cls:
            raise ValueError(f"{kind} '{name}' is already registered by {table[name].__name__}")
        table[name] = cls
        return cls

    return decorator


def get(kind: str, name: str) -> type:
    table = _REGISTRY.get(kind, {})
    if name not in table:
        raise KeyError(f"No {kind} component named '{name}'. Registered: {sorted(table)}")
    return table[name]


def build(kind: str, spec: dict, **context):
    """Instantiate the component a manifest ``spec`` (``{"type": ..., **options}``) describes."""
    options = dict(spec)
    cls = get(kind, options.pop("type"))
    return cls(**options, **context)


def registered() -> dict[str, list[str]]:
    return {kind: sorted(_REGISTRY.get(kind, {})) for kind in KINDS}
