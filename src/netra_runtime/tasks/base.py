from ..backends import Session, resolve_device
from ..bundle import Bundle


class Task:
    """A loaded bundle on one device. Subclasses define the task's call signature
    and output type; this base only owns the bundle and its ONNX sessions."""

    def __init__(self, bundle: Bundle, device: str | None = None, variant: str | None = None,
                 threads: int | None = None):
        self.bundle = bundle
        requested = resolve_device(device)
        self.variant = variant if variant is not None else bundle.pick_variant(requested.split(":")[0])
        self.sessions = {
            role: Session(path, requested, threads=threads)
            for role, path in bundle.graph_paths(self.variant).items()
        }
        # A session that couldn't get its GPU provider falls back to CPU (and logs it).
        self.device = next(iter(self.sessions.values())).device

    def __repr__(self):
        return (f"{type(self).__name__}({self.bundle.name} {self.bundle.version}, "
                f"device={self.device}, variant={self.variant or 'fp32'})")
