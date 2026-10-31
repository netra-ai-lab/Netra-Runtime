# Netra Runtime

The shared inference runtime for Netra Lab projects (Netra-OCR, Netra-Lens,
...). Research code stays in PyTorch. Each model is exported once into a
**bundle**, and every app loads bundles through one small API that runs on
CPU or GPU, with no torch at runtime:

```python
import netra_runtime

recognizer = netra_runtime.load("khmer-text-recognition/", device="auto")
result = recognizer("line.png")
print(result.text, result.confidence)
```

## Install

Install exactly one ONNX Runtime flavour. The CPU and GPU builds conflict and can't be installed side by side.

```bash
pip install "netra-runtime[cpu]"        # CPU
pip install "netra-runtime[gpu]"        # NVIDIA GPU + CPU (CUDA 13, cuDNN 9)
pip install "netra-runtime[openvino]"   # Intel CPU / iGPU
```

On a GPU machine, the CUDA libraries can come from the `nvidia-*` pip wheels (the ones torch installs). ONNX Runtime also needs `nvidia-curand`, which torch doesn't install.

If a GPU provider can't load, the runtime logs a warning and runs on CPU. It never silently claims to be on the GPU.

## Devices

| `device=` | Engine |
|---|---|
| `auto` (default, or `$NETRA_DEVICE`) | `cuda` if usable, else `cpu` |
| `cpu` | ONNX Runtime CPU |
| `cuda`, `cuda:1` | CUDA |
| `tensorrt` | TensorRT, falling back to CUDA, then CPU, per op. The engine is cached in `<bundle>/.trt_cache/` after the first (slow) build. |

On GPUs, detectors pad every image to the full `imgsz` square. ONNX Runtime's CUDA convolutions redo their setup every time the input shape changes, which made varying page shapes about 6× slower. CPUs keep Ultralytics' tighter rectangle. Square padding can change borderline detections slightly: 5 of 272 line boxes on the test pages.
| `openvino` | Intel OpenVINO |

The same bundles run in the browser through [`web/`](web/README.md) (`@netra-ai-lab/runtime-web`) on WebGPU or WebAssembly. Its outputs are checked against this package: identical recognized text, and identical boxes with the same padding. Graphs only use basic opset-18 ops so they stay portable; see [spec/bundle.md](spec/bundle.md).

## How it fits together

```
research repo (torch)  --export script-->  bundle/  --netra_runtime.load()-->  Task
                                           netra-model.json
                                           *.onnx, vocab.json
```

- **Bundle:** graphs plus a JSON manifest naming the *components* that surround them: preprocessing, the decoding loop, the tokenizer, post-processing. Format: [spec/bundle.md](spec/bundle.md).
- **Task:** what an app calls, e.g. `TextRecognizer`. Each task has a fixed output type, so apps don't depend on model internals.
- **Components:** small classes registered by name (`@register("decoder", "kv-cache")`).
- **Backend:** `Session` maps a device to ONNX Runtime execution providers. Tensors like a decoder's KV cache stay on the GPU between steps.

## Adding a model

1. Write an exporter in the model's research repo. It builds the graphs and calls `netra_runtime.bundle.write_manifest()`. Example: Netra-OCR's `netra_ocr/recognition/scripts/export_runtime_bundle.py`.
2. Check parity: the runtime's output vs the PyTorch model on a test set, per device.
3. If the model only needs existing components, you're done. If it needs something new, add one component class and register it. A new kind of output (e.g. detection boxes) needs one new task class.

## Precision variants

```bash
netra-runtime make-variant BUNDLE fp16 [--prefer-on cuda]   # half precision, float32 inputs/outputs kept
netra-runtime make-variant BUNDLE int8 [--prefer-on cpu]    # dynamic int8 on MatMul/Gemm
netra-runtime make-variant BUNDLE int8 --remove
```

A variant can always be selected with `load(..., variant="fp16")`. It only becomes a device's default after `--prefer-on`, and you should only do that after measuring accuracy and speed. Results on Netra-OCR's models (RTX 3060 / i7-13700KF):

| Bundle | Variant | Result | Decision |
|---|---|---|---|
| `doclayout-docstructbench` | fp16, GPU | 65 → 39 ms/page, 115/115 boxes matched | **default on cuda** |
| `khmer-text-lines` | fp16, GPU | 8.6 → 6.4 ms/page, 269/269 boxes matched | **default on cuda** |
| `khmer-text-recognition` | fp16, GPU | identical text, but 13.1 → 16.5 ms/line | not used |
| `khmer-text-recognition` | int8, CPU | CER 1.17% → 1.26%, no speed gain | not used |

Small, latency-bound models (the recognizer's LSTM and per-token steps) don't gain from lower precision. Convolution-heavy detectors on a GPU do.

## Bundles available so far

| Bundle | Task | From |
|---|---|---|
| `khmer-text-recognition` | `text-recognition` | Netra-OCR `export_runtime_bundle.py` |
| `khmer-text-lines` | `object-detection` (`yolo-nms`) | `netra-runtime export-yolo best.pt ...` |
| `khmer-text-recognition-blockwise` | `text-recognition` (`blockwise-kv`) | the same script, on the blockwise checkpoint |
| `doclayout-docstructbench` | `object-detection` (`yolo-end2end`) | `netra-runtime export-yolo ... --loader doclayout_yolo:YOLOv10` |

Blockwise decoding gives output identical to greedy (325/325 lines). It isn't faster here, though: with a KV cache each greedy step is already cheap, the per-line encoder dominates, and blockwise can't batch across lines. It averages 2.32 tokens per forward pass. It's worth trying where each decoder call is expensive: bigger decoders, or slow per-call runtimes.

Ultralytics-based detectors carry the AGPL-3.0 license into their bundles (recorded in `metadata.license`).

## CLI

```bash
netra-runtime devices                 # usable devices + registered components
netra-runtime info BUNDLE             # manifest summary, file hash check
netra-runtime run BUNDLE IMG... --device cuda
netra-runtime export-yolo WEIGHTS OUT --name NAME [--imgsz 640] [--loader module:Class]
```

## Development

```bash
pip install -e ".[cpu,dev]"
pytest                                       # unit tests
NETRA_TEST_BUNDLE=path/to/bundle pytest      # + a real-bundle smoke test
```
