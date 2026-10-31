# Netra bundle format, version 1

A **bundle** is a folder that fully describes one exported model: the ONNX
graphs plus everything needed to turn real inputs into real outputs. It
contains no code. Any runtime that implements the components named in the
manifest can run it: the Python runtime in this repo today, and a JavaScript
runtime on `onnxruntime-web` later. The format is written so that both can
read the same files.

```
<bundle>/
  netra-model.json      manifest (this document)
  *.onnx                graphs
  vocab.json, ...       side files referenced by components
```

## Manifest

```json
{
  "format": "netra-bundle",
  "format_version": 1,
  "name": "khmer-text-recognition",
  "version": "1.0.0",
  "task": "text-recognition",
  "graphs": {"encoder": "encoder.onnx", "decoder_step": "decoder_step.onnx"},
  "variants": {
    "fp16": {"graphs": {"encoder": "encoder.fp16.onnx"}, "devices": ["cuda", "tensorrt"]}
  },
  "components": {
    "preprocess": {"type": "line-chunker", "...": "..."},
    "decoder": {"type": "kv-cache", "...": "..."},
    "tokenizer": {"type": "vocab", "vocab": "vocab.json"},
    "postprocess": [{"type": "ocr-text-cleanup", "max_repeat": 3}]
  },
  "metadata": {"license": "MIT", "source_checkpoint": "..."},
  "files": {"encoder.onnx": "<sha256>", "...": "..."}
}
```

| Field | Meaning |
|---|---|
| `task` | Which task interface loads the bundle. Fixes the graph roles and the output type. |
| `graphs` | Graph role → file. These are the default fp32 graphs and must run on every device. |
| `variants` | Optional precision variants. A variant overrides some graph roles. `devices` lists the devices it's the default for (the first match wins). |
| `components` | Each entry is `{"type": <registered name>, ...options}`. The runtime looks up `type` in its component registry. |
| `files` | sha256 of every other file in the bundle. `netra-runtime info` checks them. |

Paths are relative to the bundle and may not leave it.

Rules for graphs:
- ONNX opset 18.
- Basic ops only: no contrib ops and no fused-attention ops. That keeps every graph runnable by every ONNX Runtime build, including `onnxruntime-web` (WASM/WebGPU).
- Optimizations such as fusion and quantization belong to variants or to the engine at load time, not to the default graphs.

## Tasks

### `text-recognition`
One text-line image in, `Text(text, confidence)` out.

- **Graphs:** `encoder`, `decoder_step`.
- **Components:** `preprocess`, `decoder`, `tokenizer`, `postprocess` (a list, optional).
- **`encoder`:** `chunks` float32 `[N, 1, H, W]` (the chunks of *one* line) → `cross_k`, `cross_v` float32 `[layers, 1, heads, S, head_dim]`.
- **`decoder_step`:** see `kv-cache` below.

### `object-detection`
One image in, a list of `Detection(box, score, label, class_id)` out. Boxes are `x1, y1, x2, y2` in original-image pixels.

- **Graph:** `model`: `images` float32 `[1, 3, H, W]` (RGB, [0, 1]) → `output0`. `H` and `W` are dynamic.
- **Components:**
  - `preprocess` must also be able to map boxes back to the original image (`letterbox` does);
  - `head` turns `output0` into boxes, scores and class ids, and lists the class `labels`.

## Components

### `preprocess: line-chunker`
Options: `height`, `chunk_width`, `chunk_overlap`, `min_width`, `pad_value`, `mean`, `std`.

1. Convert to grayscale (PIL `L`: `0.299R + 0.587G + 0.114B`).
2. Resize to `height`, keeping the aspect ratio, with bilinear interpolation. The width is `max(min_width, floor(height * w / h))`.
3. Scale to [0, 1].
4. Cut windows `chunk_width` wide, starting every `chunk_width - chunk_overlap` px. Right-pad the last window with `pad_value`.
5. Normalize: `(x - mean) / std`.

Resampling differs slightly between PIL, OpenCV and a browser canvas. A port should match on the test set, but bit-identical pixels aren't guaranteed outside PIL.

### `decoder: kv-cache`
Options: `layers`, `heads`, `head_dim`, `max_len`.

`decoder_step` takes:
- `tokens` int64 `[B, n]`;
- `past_k`, `past_v` float32 `[layers, B, heads, P, head_dim]`;
- `cross_k`, `cross_v` (from the encoder, stacked along `B`);
- `cross_mask` float32 `[B, S]`: 0 for real encoder positions, `-inf` for padding.

It returns:
- `logits` `[B, n, vocab]`;
- `present_k`, `present_v` `[layers, B, heads, P + n, head_dim]`.

The position of `tokens[:, i]` is `P + i`, and causal masking is handled inside the graph. The greedy loop works like this:
1. Start with `P = 0` (empty cache) and `<sos>` for every line.
2. Each step, feed the last token and the previous `present_*` tensors as the new `past_*`.
3. A line is finished at `<eos>`. The batch stops when every line is finished, or after `max_len` steps.

To decode lines of different encoder lengths together:
- zero-pad their `cross_k`/`cross_v` to the longest `S`;
- set `cross_mask` to `-inf` on the padding.

The result is the same as decoding each line alone.

`n > 1` is allowed, which lets blockwise/speculative verification reuse the same graph.

### `preprocess: letterbox`
Options: `imgsz`, `stride`, `auto`, `pad_value`, `scaleup`. This follows Ultralytics' predict-time letterbox.

1. Scale by `r = min(imgsz / h, imgsz / w)` (capped at 1 when `scaleup` is false).
2. The new size is `round(w * r)` × `round(h * r)`. Resize with bilinear interpolation, half-pixel centers and no antialiasing (OpenCV `INTER_LINEAR`).
3. Pad up to `imgsz`, or only up to the next multiple of `stride` when `auto` is true. The padding is split evenly: top/left get `round(pad / 2 - 0.1)`, bottom/right get `round(pad / 2 + 0.1)`. Fill with `pad_value`.
4. Scale to [0, 1].

To map boxes back: subtract the left/top padding, divide by the per-axis gain (`new_w / w`, `new_h / h`), and clip to the image.

### `head: yolo-nms`
Options: `labels`, `conf`, `iou`, `max_det` (300), `max_nms` (30000), `agnostic` (false).

`output0` is `[1, 4 + classes, anchors]`: `cx, cy, w, h`, then one score per class.
1. Each anchor takes its best class. Keep anchors with score > `conf`, at most the top `max_nms` by score.
2. Convert to `xyxy`.
3. Run greedy NMS at `iou`. Unless `agnostic`, run it per class: offset each box by `class_id * 7680` first.
4. Keep the first `max_det`.

### `head: yolo-end2end`
Options: `labels`, `conf`.

`output0` is `[1, max_det, 6]`: `x1, y1, x2, y2, score, class`, already free of duplicates (NMS-free heads, e.g. YOLOv10). Keep rows with score > `conf`.

### `tokenizer: vocab`
Options: `vocab` (a JSON file of `{token: id}`), plus `sos`, `eos`, `pad`, `unk` token names.

Decoding rules:
- Skip `sos` and `pad`.
- Stop at `eos`.
- `unk`, and any other `<...>` token, becomes a space.

### `postprocess: ocr-text-cleanup`
Options: `max_repeat`.

1. Strip U+FFFD and control characters.
2. Unless the line is a passport MRZ line, collapse:
   - one character repeated more than `max_repeat` times in a row;
   - a 2–20-character cluster repeated inside a word;
   - the same whitespace-separated token repeated.
3. Collapse runs of whitespace and trim.

## Versioning

- **Adding** a component type or task is backward compatible and doesn't bump `format_version`.
- **Changing** the meaning of an existing field or graph contract bumps it. Runtimes refuse versions they don't know.
