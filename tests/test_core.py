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


def test_letterbox_and_restore_round_trip():
    from netra_runtime.components.detection import Letterbox
    lb = Letterbox(imgsz=640, stride=32, auto=True)
    tensor, info = lb(np.zeros((300, 1000, 3), np.uint8))
    # long side -> 640, short side 192 already a multiple of 32: no padding needed
    assert tensor.shape == (1, 3, 192, 640)
    tensor, info = lb(np.zeros((310, 1000, 3), np.uint8))   # 198 -> padded to 224
    assert tensor.shape == (1, 3, 224, 640) and info.pad_y == 13
    assert tensor[0, 0, 0, 0] == np.float32(114 / 255)
    box = np.array([[100.0, 50.0, 400.0, 300.0]])
    letterboxed = box * [info.gain_x, info.gain_y, info.gain_x, info.gain_y] + [info.pad_x, info.pad_y] * 2
    np.testing.assert_allclose(lb.restore(letterboxed, info), box, atol=1e-3)


def test_numpy_resize_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    from netra_runtime.components import detection
    image = np.random.default_rng(0).integers(0, 256, (97, 203, 3), dtype=np.uint8)
    reference = cv2.resize(image, (150, 61), interpolation=cv2.INTER_LINEAR)
    saved, detection.cv2 = detection.cv2, None
    try:
        ours = detection.resize_bilinear(image, 150, 61)
    finally:
        detection.cv2 = saved
    assert np.abs(ours.astype(int) - reference).max() <= 1


def test_nms_heads():
    from netra_runtime.components.detection import YoloEnd2EndHead, YoloNMSHead, nms
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], np.float32)
    assert list(nms(boxes, np.array([0.9, 0.8, 0.7]), iou=0.5)) == [0, 2]
    # raw head: (1, 4 + 2 classes, 3 anchors) in cx, cy, w, h; anchors 0 and 1 overlap but differ in class
    raw = np.array([[[5, 5.5, 25], [5, 5.5, 25], [10, 9, 10], [10, 9, 10],
                     [0.9, 0.1, 0.1], [0.1, 0.8, 0.6]]], np.float32)
    b, s, c = YoloNMSHead(labels=["a", "b"], conf=0.25, iou=0.5)(raw)
    assert list(c) == [0, 1, 1] and len(b) == 3            # class-aware: no suppression across classes
    b, s, c = YoloNMSHead(labels=["a", "b"], conf=0.25, iou=0.5, agnostic=True)(raw)
    assert len(b) == 2
    e2e = np.array([[[0, 0, 5, 5, 0.9, 1], [0, 0, 5, 5, 0.1, 0]]], np.float32)
    b, s, c = YoloEnd2EndHead(labels=["a", "b"], conf=0.25)(e2e)
    assert list(c) == [1]


def test_make_and_remove_variants(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper
    from netra_runtime.export.variants import make_variant, remove_variant

    weight = numpy_helper.from_array(np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])], "toy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 4])], [weight])
    root = tmp_path / "b"
    root.mkdir()
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9),
              str(root / "model.onnx"))
    write_manifest(root, {"name": "toy", "version": "0", "task": "object-detection",
                          "graphs": {"model": "model.onnx"}, "components": {}})

    make_variant(root, "fp16", prefer_on=["cuda"])
    make_variant(root, "int8")
    bundle = load_bundle(root)
    assert bundle.pick_variant("cuda") == "fp16" and bundle.pick_variant("cpu") is None
    assert bundle.verify() == []

    from netra_runtime.backends import Session
    x = np.random.default_rng(1).standard_normal((3, 8)).astype(np.float32)
    want = Session(root / "model.onnx", "cpu").run({"x": x})["y"]
    for variant in ("fp16", "int8"):
        got = Session(bundle.graph_paths(variant)["model"], "cpu").run({"x": x})["y"]
        assert got.dtype == np.float32 and np.allclose(got, want, atol=0.1)   # io types stay float32

    remove_variant(root, "int8")
    assert "int8" not in load_bundle(root).variants and not (root / "model.int8.onnx").exists()


class _ScriptedStep:
    """Fake decoder_step: the 'model' always predicts ``target[pos + 1]`` and its
    proposal heads guess ``target[pos + 2 + j]``, except at positions in ``wrong``."""

    def __init__(self, target, block_size, wrong=()):
        self.target, self.block_size, self.wrong, self.vocab = target, block_size, set(wrong), 16

    def run(self, inputs, keep=()):
        tokens, past = inputs["tokens"], inputs["past_k"]
        P, n = past.shape[3], tokens.shape[1]
        logits = np.zeros((1, n, self.vocab), np.float32)
        proposals = np.zeros((1, n, self.block_size - 1), np.int64)
        for i in range(n):
            pos = P + i
            logits[0, i, self.target[min(pos + 1, len(self.target) - 1)]] = 5.0
            for j in range(self.block_size - 1):
                k = min(pos + 2 + j, len(self.target) - 1)
                proposals[0, i, j] = 9 if pos in self.wrong else self.target[k]
        present = np.zeros(past.shape[:3] + (P + n, past.shape[4]), np.float32)
        return {"logits": logits, "present_k": present, "present_v": present, "proposal_ids": proposals}


def test_blockwise_decoding_matches_greedy():
    from netra_runtime.components.decoding import BlockwiseKVDecoder, KVCacheGreedyDecoder
    target = [1, 4, 5, 6, 7, 8, 10, 11, 12, 2]          # <sos> ... <eos>
    args = dict(layers=1, heads=1, head_dim=2, max_len=50)
    mask = np.zeros((1, 3), np.float32)
    greedy = KVCacheGreedyDecoder(**args).decode(_ScriptedStep(target, 1), None, None, mask, 1, 2)
    for wrong in [(), (0, 3), (1, 2, 3, 4, 5, 6)]:
        decoder = BlockwiseKVDecoder(block_size=4, **args)
        got = decoder.decode(_ScriptedStep(target, 4, wrong), None, None, mask, 1, 2)
        assert got[0][0] == greedy[0][0] == target[:-1]
        np.testing.assert_allclose(got[0][1], greedy[0][1])
    # all guesses right: 9 tokens in ceil(9 / 4) = 3 forward passes
    decoder = BlockwiseKVDecoder(block_size=4, **args)
    decoder.decode(_ScriptedStep(target, 4), None, None, mask, 1, 2)
    assert (decoder.rounds, decoder.tokens) == (3, 9)

    capped = BlockwiseKVDecoder(block_size=4, **{**args, "max_len": 4})
    ids, _ = capped.decode(_ScriptedStep(target, 4), None, None, mask, 1, 2)[0]
    assert ids == target[:5]                           # stops at max_len tokens, like greedy
