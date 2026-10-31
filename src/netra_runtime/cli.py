"""netra-runtime command line.

    netra-runtime devices                       devices the installed ONNX Runtime can use
    netra-runtime info BUNDLE                   manifest summary + file integrity check
    netra-runtime run BUNDLE INPUT... [--device D]
    netra-runtime export-yolo WEIGHTS OUT --name NAME   (needs the export extra + the model's library)
    netra-runtime make-variant BUNDLE fp16|int8 [--prefer-on DEVICE...]
"""

import argparse
import json
import sys
import time

from . import available_devices, load, load_bundle, registry


def _devices(args):
    print("available devices:", ", ".join(available_devices()))
    print("components:", json.dumps(registry.registered(), indent=2))


def _info(args):
    bundle = load_bundle(args.bundle)
    print(f"{bundle.name} {bundle.version}  task={bundle.task}")
    for role, path in bundle.graph_paths().items():
        print(f"  graph {role}: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    for name, variant in bundle.variants.items():
        print(f"  variant {name}: devices={variant.get('devices', [])}")
    for kind, spec in bundle.components.items():
        types = [s["type"] for s in spec] if isinstance(spec, list) else [spec["type"]]
        print(f"  {kind}: {', '.join(types)}")
    bad = bundle.verify()
    print("  files: OK" if not bad else f"  files: CHANGED OR MISSING: {bad}")
    return 1 if bad else 0


def _run(args):
    task = load(args.bundle, device=args.device, variant=args.variant)
    print(task, file=sys.stderr)
    for item in args.inputs:
        start = time.perf_counter()
        result = task(item)
        ms = 1000 * (time.perf_counter() - start)
        output = [vars(r) for r in result] if isinstance(result, list) else vars(result)
        print(json.dumps({"input": item, "output": output, "ms": round(ms, 1)}, ensure_ascii=False))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["export-yolo"]:
        from .export.yolo import main as export_yolo
        return export_yolo(argv[1:])
    if argv[:1] == ["make-variant"]:
        from .export.variants import main as make_variant
        return make_variant(argv[1:])

    parser = argparse.ArgumentParser(prog="netra-runtime", description="Run Netra model bundles.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="List usable devices and registered components").set_defaults(func=_devices)

    info = sub.add_parser("info", help="Describe a bundle and verify its files")
    info.add_argument("bundle")
    info.set_defaults(func=_info)

    run = sub.add_parser("run", help="Run a bundle on inputs")
    run.add_argument("bundle")
    run.add_argument("inputs", nargs="+")
    run.add_argument("--device", default=None, help="auto (default), cpu, cuda[:N], tensorrt[:N], openvino")
    run.add_argument("--variant", default=None)
    run.set_defaults(func=_run)

    args = parser.parse_args(argv)
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
