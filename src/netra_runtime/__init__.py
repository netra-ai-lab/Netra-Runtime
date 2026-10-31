"""Netra Runtime: run exported Netra model bundles on CPU or GPU.

    import netra_runtime
    recognizer = netra_runtime.load("hf://Netra-Lab/Netra-OCR/recognition", device="auto")
    print(recognizer("line.png").text)
"""

from pathlib import Path

from . import components, registry, tasks  # noqa: F401  (registers built-in components)
from .backends import available_devices
from .bundle import Bundle, BundleError, load_bundle

__version__ = "0.1.0"


def resolve(source: str | Path) -> Path:
    """A local bundle folder for ``source``: a path, or
    ``hf://<org>/<repo>[/<folder>][@revision]`` -- a bundle at the root of a
    Hugging Face model repo, or in one of its folders (one repo can hold a
    project's bundles side by side). Only that bundle's files are downloaded,
    once, into the Hugging Face cache, so it also works offline afterwards."""
    text = str(source)
    if not text.startswith("hf://"):
        return Path(source)
    path, _, revision = text[len("hf://"):].partition("@")
    org, repo, *folder = path.strip("/").split("/")
    folder = "/".join(folder)
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=f"{org}/{repo}", revision=revision or None,
                             allow_patterns=[f"{folder}/*"] if folder else None)
    return Path(root) / folder


def load(bundle: str | Path | Bundle, device: str | None = None, variant: str | None = None,
         threads: int | None = None):
    """Load a bundle as its task (e.g. a ``TextRecognizer``).

    bundle: a folder, ``hf://<org>/<repo>[/<folder>][@revision]``, or a loaded ``Bundle``.
    device: "auto" (default; or $NETRA_DEVICE), "cpu", "cuda[:N]", "tensorrt[:N]", "openvino".
    variant: force a precision variant listed in the manifest (e.g. "fp16");
             by default the manifest's preference for the device is used.
    """
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(resolve(bundle))
    task_cls = registry.get("task", bundle.task)
    return task_cls(bundle, device=device, variant=variant, threads=threads)


__all__ = ["Bundle", "BundleError", "available_devices", "load", "load_bundle", "resolve", "__version__"]
