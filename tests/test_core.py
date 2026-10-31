import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import netra_runtime
from netra_runtime import registry
from netra_runtime.bundle import BundleError, load_bundle, write_manifest
from netra_runtime.components.postprocess import OCRTextCleanup
from netra_runtime.components.preprocess import LineChunker
from netra_runtime.components.tokenizer import VocabTokenizer


def make_bundle(root: Path, **overrides) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "encoder.onnx").write_bytes(b"graph")
    (root / "vocab.json").write_text(json.dumps({"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3,
                                                 "ក": 4, "ា": 5, "<mask>": 6}))
    manifest = {"name": "toy", "version": "0.0.1", "task": "text-recognition",
                "graphs": {"encoder": "encoder.onnx"},
                "components": {"tokenizer": {"type": "vocab", "vocab": "vocab.json"}},
                **overrides}
    write_manifest(root, manifest)
    return root


def test_manifest_round_trip_and_integrity(tmp_path):
    bundle = load_bundle(make_bundle(tmp_path / "b"))
    assert (bundle.name, bundle.task) == ("toy", "text-recognition")
    assert bundle.verify() == []
    (tmp_path / "b" / "encoder.onnx").write_bytes(b"tampered")
    assert bundle.verify() == ["encoder.onnx"]


def test_manifest_rejects_paths_outside_bundle(tmp_path):
    bundle = load_bundle(make_bundle(tmp_path / "b"))
    with pytest.raises(BundleError):
        bundle.path("../secret.txt")


def test_manifest_rejects_unknown_format_version(tmp_path):
    root = make_bundle(tmp_path / "b")
    manifest = json.loads((root / "netra-model.json").read_text())
    manifest["format_version"] = 99
    (root / "netra-model.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="format_version"):
        load_bundle(root)


def test_variant_selection(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "encoder.fp16.onnx").write_bytes(b"graph")
    bundle = load_bundle(make_bundle(tmp_path / "b", variants={
        "fp16": {"graphs": {"encoder": "encoder.fp16.onnx"}, "devices": ["cuda", "tensorrt"]}}))
    assert bundle.pick_variant("cuda") == "fp16"
    assert bundle.pick_variant("cpu") is None
    assert bundle.graph_paths("fp16")["encoder"].name == "encoder.fp16.onnx"


def test_registry_rejects_conflicting_names():
    with pytest.raises(ValueError):
        registry.register("preprocess", "line-chunker")(type("Other", (), {}))
    with pytest.raises(KeyError):
        registry.get("decoder", "does-not-exist")


def test_line_chunker_shapes_and_padding():
    chunker = LineChunker(height=48, chunk_width=100, chunk_overlap=16, min_width=50)
    image = Image.new("L", (200, 24), color=0)        # -> resized to 400 x 48
    chunks = chunker(image)
    # windows start every 84 px across 400 px: 0, 84, ..., 336 -> 5 chunks
    assert chunks.shape == (5, 1, 48, 100) and chunks.dtype == np.float32
    assert chunks[0].max() == -1.0                    # black, normalized
    assert chunks[-1, 0, 0, -1] == 1.0                # right padding is white
    assert chunker(np.zeros((10, 5, 3), np.uint8)).shape == (1, 1, 48, 100)  # min_width


def test_tokenizer_decode(tmp_path):
    bundle = load_bundle(make_bundle(tmp_path / "b"))
    tok = VocabTokenizer(vocab="vocab.json", bundle=bundle)
    assert tok.decode([1, 4, 5, 3, 6, 4, 2, 4]) == "កា  ក"


def test_text_cleanup():
    clean = OCRTextCleanup(max_repeat=3)
    assert clean("kkkkk") == "k"
    assert clean("H H H H") == "H"
    assert clean("P<KHMNHEM<<DARAYUT<<<<<<<<<<<<") == "P<KHMNHEM<<DARAYUT<<<<<<<<<<<<"


@pytest.mark.skipif(not os.environ.get("NETRA_TEST_BUNDLE"), reason="set NETRA_TEST_BUNDLE to a recognition bundle")
def test_real_bundle_runs():
    recognizer = netra_runtime.load(os.environ["NETRA_TEST_BUNDLE"], device="cpu")
    result = recognizer(Image.new("L", (300, 48), color=255))
    assert isinstance(result.text, str) and 0.0 <= result.confidence <= 1.0
