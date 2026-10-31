"""Model bundles: a folder holding a ``netra-model.json`` manifest, the ONNX
graphs and any side files (vocab, label maps). See ``spec/bundle.md`` for the
format; this module only reads and checks it.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_NAME = "netra-model.json"
FORMAT = "netra-bundle"
FORMAT_VERSION = 1


class BundleError(ValueError):
    pass


@dataclass
class Bundle:
    root: Path
    name: str
    version: str
    task: str
    graphs: dict[str, str]
    components: dict
    variants: dict[str, dict] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)

    def path(self, relative: str) -> Path:
        resolved = (self.root / relative).resolve()
        if self.root.resolve() not in resolved.parents and resolved != self.root.resolve():
            raise BundleError(f"'{relative}' points outside the bundle")
        return resolved

    def graph_paths(self, variant: str | None = None) -> dict[str, Path]:
        """Graph files for ``variant`` (None = the default fp32 graphs)."""
        graphs = dict(self.graphs)
        if variant is not None:
            if variant not in self.variants:
                raise BundleError(f"Bundle has no variant '{variant}'. Available: {sorted(self.variants)}")
            graphs.update(self.variants[variant].get("graphs", {}))
        return {role: self.path(file) for role, file in graphs.items()}

    def pick_variant(self, device: str) -> str | None:
        """The first variant the manifest marks as preferred for ``device``, else None."""
        for name, variant in self.variants.items():
            if device in variant.get("devices", []):
                return name
        return None

    def verify(self) -> list[str]:
        """Check every file listed under ``files`` against its sha256; return the mismatches."""
        bad = []
        for relative, expected in self.files.items():
            path = self.path(relative)
            if not path.exists() or _sha256(path) != expected:
                bad.append(relative)
        return bad


def load_bundle(root: str | Path) -> Bundle:
    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        raise BundleError(f"No {MANIFEST_NAME} in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("format") != FORMAT:
        raise BundleError(f"{manifest_path} is not a {FORMAT} manifest")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise BundleError(f"Unsupported format_version {manifest.get('format_version')} "
                          f"(this runtime reads version {FORMAT_VERSION})")
    for key in ("name", "version", "task", "graphs", "components"):
        if key not in manifest:
            raise BundleError(f"{manifest_path} is missing '{key}'")

    bundle = Bundle(
        root=root,
        name=manifest["name"],
        version=manifest["version"],
        task=manifest["task"],
        graphs=manifest["graphs"],
        components=manifest["components"],
        variants=manifest.get("variants", {}),
        metadata=manifest.get("metadata", {}),
        files=manifest.get("files", {}),
    )
    for path in bundle.graph_paths().values():
        if not path.exists():
            raise BundleError(f"Graph file missing: {path}")
    return bundle


def write_manifest(root: str | Path, manifest: dict) -> Path:
    """Write ``manifest`` into ``root``, filling in the format header and file hashes."""
    root = Path(root)
    out = {"format": FORMAT, "format_version": FORMAT_VERSION, **manifest}
    out["files"] = {
        str(p.relative_to(root)): _sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != MANIFEST_NAME
    }
    path = root / MANIFEST_NAME
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
