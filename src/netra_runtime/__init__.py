"""Netra Runtime: run exported Netra model bundles on CPU or GPU.

    import netra_runtime
    recognizer = netra_runtime.load("khmer-text-recognition/", device="auto")
    print(recognizer("line.png").text)
"""

from pathlib import Path

from . import components, registry, tasks  # noqa: F401  (registers built-in components)
from .backends import available_devices
from .bundle import Bundle, BundleError, load_bundle

__version__ = "0.1.0"


def load(bundle: str | Path | Bundle, device: str | None = None, variant: str | None = None,
         threads: int | None = None):
    """Load a bundle as its task (e.g. a ``TextRecognizer``).

    device: "auto" (default; or $NETRA_DEVICE), "cpu", "cuda[:N]", "tensorrt[:N]", "openvino".
    variant: force a precision variant listed in the manifest (e.g. "fp16");
             by default the manifest's preference for the device is used.
    """
    if not isinstance(bundle, Bundle):
        bundle = load_bundle(bundle)
    task_cls = registry.get("task", bundle.task)
    return task_cls(bundle, device=device, variant=variant, threads=threads)


__all__ = ["Bundle", "BundleError", "available_devices", "load", "load_bundle", "__version__"]
