"""Netra Runtime: run exported Netra model bundles on CPU or GPU.

    import netra_runtime
    recognizer = netra_runtime.load("hf://Netra-Lab/Netra-OCR/recognition", device="auto")
    print(recognizer("line.png").text)
"""

import json
from pathlib import Path

from . import components, registry, tasks  # noqa: F401  (registers built-in components)
from .backends import available_devices, resolve_device
from .bundle import MANIFEST_NAME, Bundle, BundleError, load_bundle, pick_variant

__version__ = "0.1.1"


def resolve(source: str | Path, device: str | None = None, variant: str | None = None) -> Path:
    """A local bundle folder for ``source``: a path, or
    ``hf://<org>/<repo>[/<folder>][@revision]`` -- a bundle at the root of a
    Hugging Face model repo, or in one of its folders (one repo can hold a
    project's bundles side by side). Files are downloaded once into the Hugging
    Face cache, so it also works offline afterwards. Given a ``device`` or
    ``variant``, only the graphs that will run there are downloaded (e.g. not
    the fp16 copies on a CPU); otherwise the whole bundle is."""
    text = str(source)
    if not text.startswith("hf://"):
        return Path(source)
    path, _, revision = text[len("hf://"):].partition("@")
    org, repo, *folder = path.strip("/").split("/")
    folder = "/".join(folder)
    prefix = f"{folder}/" if folder else ""
    from huggingface_hub import snapshot_download

    def fetch(patterns):
        return Path(snapshot_download(repo_id=f"{org}/{repo}", revision=revision or None,
                                      allow_patterns=patterns)) / folder

    if device is None and variant is None:
        return fetch([f"{folder}/*"] if folder else None)
    manifest = json.loads((fetch([prefix + MANIFEST_NAME]) / MANIFEST_NAME).read_text(encoding="utf-8"))
    if variant is None:
        variant = pick_variant(manifest.get("variants", {}), resolve_device(device).split(":")[0])
    graphs = dict(manifest.get("graphs", {}))
    every_graph = set(graphs.values())
    for spec in manifest.get("variants", {}).values():
        every_graph.update(spec.get("graphs", {}).values())
    if variant is not None and variant in manifest.get("variants", {}):
        graphs.update(manifest["variants"][variant].get("graphs", {}))
    side_files = [name for name in manifest.get("files", {}) if name not in every_graph]
    return fetch([prefix + name for name in (MANIFEST_NAME, *side_files, *graphs.values())])


def load(bundle: str | Path | Bundle, device: str | None = None, variant: str | None = None,
         threads: int | None = None):
    """Load a bundle as its task (e.g. a ``TextRecognizer``).

    bundle: a folder, ``hf://<org>/<repo>[/<folder>][@revision]``, or a loaded ``Bundle``.
    device: "auto" (default; or $NETRA_DEVICE), "cpu", "cuda[:N]", "tensorrt[:N]", "openvino".
    variant: force a precision variant listed in the manifest (e.g. "fp16");
             by default the manifest's preference for the device is used.
    """
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(resolve(bundle, device=resolve_device(device), variant=variant))
    task_cls = registry.get("task", bundle.task)
    return task_cls(bundle, device=device, variant=variant, threads=threads)


__all__ = ["Bundle", "BundleError", "available_devices", "load", "load_bundle", "resolve", "__version__"]
