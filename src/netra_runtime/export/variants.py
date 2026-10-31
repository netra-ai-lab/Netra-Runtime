"""Add reduced-precision variants to an existing bundle.

    netra-runtime make-variant BUNDLE fp16 [--prefer-on cuda tensorrt]
    netra-runtime make-variant BUNDLE int8 [--prefer-on cpu]
    netra-runtime make-variant BUNDLE int8 --remove

fp16   all float weights/activations in half precision, graph inputs and
       outputs kept float32 so the task's contract doesn't change. For GPUs.
int8   dynamic quantization: weights stored as int8, activations quantized on
       the fly. Only MatMul/Gemm by default -- Transformer layers gain the most,
       while ConvInteger is usually *slower* than float Conv on CPU (conv-heavy
       models need static quantization with calibration data instead).

A new variant is only *available* (``load(..., variant="int8")``) until
``--prefer-on`` makes it the default for those devices -- do that after
checking accuracy on a real test set, since quantization can change outputs.
"""

import json
from pathlib import Path

from ..bundle import MANIFEST_NAME, load_bundle, write_manifest

DEFAULT_INT8_OPS = ("MatMul", "Gemm")


def _convert_fp16(src: Path, dst: Path):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    model = convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    # When two nodes share a float32 constant (e.g. two Resize ops with the
    # same scales), the converter inserts one identical Cast per consumer, all
    # writing the same tensor -- an invalid graph. Keep the first of each.
    produced, kept = set(), []
    for node in model.graph.node:
        if node.output and all(out in produced for out in node.output):
            continue
        produced.update(node.output)
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    onnx.save(model, str(dst))


def _convert_int8(src: Path, dst: Path, op_types=DEFAULT_INT8_OPS):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=list(op_types))


CONVERTERS = {"fp16": _convert_fp16, "int8": _convert_int8}


def make_variant(bundle_dir: str | Path, variant: str, prefer_on: list[str] | None = None,
                 roles: list[str] | None = None) -> dict:
    """Convert the bundle's default graphs (all roles, or only ``roles``) and register the variant."""
    if variant not in CONVERTERS:
        raise ValueError(f"Unknown variant '{variant}'. Available: {sorted(CONVERTERS)}")
    bundle_dir = Path(bundle_dir)
    bundle = load_bundle(bundle_dir)
    graphs = {}
    for role, path in bundle.graph_paths().items():
        if roles and role not in roles:
            continue
        out = path.with_name(f"{path.stem}.{variant}{path.suffix}")
        CONVERTERS[variant](path, out)
        graphs[role] = out.name

    manifest = json.loads((bundle_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    for key in ("format", "format_version", "files"):
        manifest.pop(key, None)
    variants = manifest.setdefault("variants", {})
    variants[variant] = {"graphs": graphs, "devices": list(prefer_on or [])}
    write_manifest(bundle_dir, manifest)
    return variants[variant]


def remove_variant(bundle_dir: str | Path, variant: str):
    """Delete a variant's graph files and its manifest entry."""
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    for key in ("format", "format_version", "files"):
        manifest.pop(key, None)
    entry = manifest.get("variants", {}).pop(variant, None)
    if entry is None:
        raise ValueError(f"Bundle has no variant '{variant}'")
    default_files = set(manifest["graphs"].values())
    for file in entry.get("graphs", {}).values():
        if file not in default_files:
            (bundle_dir / file).unlink(missing_ok=True)
    write_manifest(bundle_dir, manifest)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="netra-runtime make-variant")
    parser.add_argument("bundle")
    parser.add_argument("variant", choices=sorted(CONVERTERS))
    parser.add_argument("--prefer-on", nargs="*", default=[],
                        help="devices that should load this variant by default (e.g. cuda tensorrt)")
    parser.add_argument("--roles", nargs="*", help="only convert these graph roles")
    parser.add_argument("--remove", action="store_true", help="delete this variant instead")
    args = parser.parse_args(argv)
    if args.remove:
        remove_variant(args.bundle, args.variant)
        print(f"{args.variant}: removed")
        return
    result = make_variant(args.bundle, args.variant, args.prefer_on, args.roles)
    print(f"{args.variant}: {result}")
