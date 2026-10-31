# @netra-ai-lab/runtime-web

The browser half of Netra Runtime. It loads the **same bundles** as the Python package and runs them with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) on **WebGPU** or **WebAssembly**. It's plain ES modules, so there's no build step.

```js
import { load } from "@netra-ai-lab/runtime-web";

const recognizer = await load("/models/khmer-text-recognition/");     // device: "auto"
const { text, confidence } = await recognizer.recognize(imgElement); // <img>, canvas, ImageBitmap, Blob, ImageData

const lines = await load("/models/khmer-text-lines/", { device: "wasm" });
const boxes = await lines.detect(imageData);   // [{box: [x1, y1, x2, y2], score, label, classId}]
```

- **`device`:** `"auto"` (WebGPU when `navigator.gpu` exists, otherwise WebAssembly), `"webgpu"` or `"wasm"`.
- **`variant`:** forces a manifest variant, e.g. `"fp16"`. By default the manifest's per-device preference applies; current bundles prefer fp16 only on `cuda`.
- **`readFile(name) → ArrayBuffer`:** reads bundle files from somewhere other than a URL (tests, user-dropped files).
- **`ort`:** the `onnxruntime-web` module to use. Defaults to the bare `onnxruntime-web` import, so in a page without a bundler, map it with an import map (see `test/browser.html`).

Pixel buffers are `{data, width, height, channels}` (uint8, interleaved), so the runtime also works off the main thread and in Node.

## Same results as Python

Preprocessing reproduces the Python runtime's arithmetic exactly:
- **Line chunker:** a port of Pillow's fixed-point bilinear resize, with every step rounded to float32.
- **Letterbox:** the spec's float64 bilinear resize.

Checked with `test/parity.mjs` (WebAssembly in Node) and `test/browser.html` (Chrome) against reference outputs from `test/make_fixtures.py`:

| Check | WebAssembly (Node) | WebGPU (Chrome, RTX 3060) |
|---|---|---|
| line-chunker inputs | 325/325 bit-identical | — |
| `khmer-text-recognition` | 325/325 identical text, ~65 ms/line | 100/100 identical, 86 ms/line |
| `khmer-text-recognition-blockwise` | 325/325 identical text, ~65 ms/line | 100/100 identical, **64 ms/line** |
| `khmer-text-lines` | 270/270 boxes identical (IoU 1.0), 117 ms/page | 269/270 (square padding), 21 ms/page |
| `doclayout-docstructbench` | 113/113 boxes identical (IoU 1.0), 1.36 s/page | 111/113 matched (square padding), 129 ms/page |

Notes on these results:
- **Blockwise is the faster decoder on WebGPU**, where every call has a high fixed cost and fewer calls win. On CUDA and CPU, plain greedy decoding is as fast or faster.
- **fp16 variants don't work on WebGPU** (onnxruntime-web 1.31): the line detector returns no boxes, and DocLayout fails in the `Expand` kernel. Use the fp32 graphs in the browser. Current manifests only prefer fp16 on `cuda`, so it's never chosen automatically here.

Image *decoding* is the browser's job. JPEG decoders can differ slightly from Pillow's, so results on JPEG files may differ marginally. PNGs decode identically.

## Notes

- Lines are decoded one at a time (`batch()` loops). Batched decoding is Python-only for now.
- On WebGPU, greedy decoding keeps the KV cache in GPU memory (`preferredOutputLocation`). Blockwise decoding trims the cache on the CPU every round, so it reads the cache back.
- Detectors pad to the full square on WebGPU (fixed shapes) and to the tighter rectangle on WebAssembly, the same as the Python runtime on CUDA vs CPU.

## Tests

```bash
npm install
npm test                                              # model-free unit tests
python test/make_fixtures.py --bundles B --lines L --pages P --out F   # needs the Python runtime
node test/parity.mjs B F                              # WebAssembly parity
# browser: serve a folder with web/, B and F, then open
#   web/test/browser.html?bundles=/B/&fixtures=/F/&device=webgpu
```
