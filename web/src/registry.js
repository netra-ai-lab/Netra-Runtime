// Component registry: a manifest names components by `type`; each type maps to
// one factory registered here (mirrors netra_runtime/registry.py).

const KINDS = ["preprocess", "decoder", "head", "tokenizer", "postprocess", "task"];
const registry = new Map(KINDS.map((kind) => [kind, new Map()]));

export function register(kind, name, factory) {
  if (!registry.has(kind)) throw new Error(`Unknown component kind '${kind}'. Known: ${KINDS}`);
  registry.get(kind).set(name, factory);
}

export function get(kind, name) {
  const factory = registry.get(kind)?.get(name);
  if (!factory) {
    throw new Error(`No ${kind} component named '${name}'. Registered: ${[...(registry.get(kind)?.keys() ?? [])]}`);
  }
  return factory;
}

/** Build the component a manifest spec ({type, ...options}) describes. */
export function build(kind, spec, context = {}) {
  const { type, ...options } = spec;
  return get(kind, type)(options, context);
}
