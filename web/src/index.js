// Netra Runtime for the browser.
//
//   import { load } from "@netra-ai-lab/runtime-web";
//   const recognizer = await load("/models/khmer-text-recognition/");
//   const { text, confidence } = await recognizer.recognize(imageElement);
//
//   const detector = await load("/models/khmer-text-lines/", { device: "wasm" });
//   const boxes = await detector.detect(imageData);

import * as defaultOrt from "onnxruntime-web";

import "./components/preprocess.js";
import "./components/text.js";
import "./components/decoding.js";
import "./components/detection.js";
import "./tasks.js";
import { get, register } from "./registry.js";
import { loadBundle } from "./bundle.js";
import { resolveDevice } from "./session.js";

/**
 * Load a bundle as its task.
 * @param source  bundle base URL, or a loaded Bundle
 * @param options.device   "auto" (WebGPU if available, else WebAssembly), "webgpu" or "wasm"
 * @param options.variant  force a manifest variant (e.g. "fp16"); default: the manifest's preference
 * @param options.readFile (name) => ArrayBuffer, to read bundle files from somewhere other than a URL
 * @param options.ort      the onnxruntime-web module to use (default: the "onnxruntime-web" import)
 */
export async function load(source, { device = "auto", variant, readFile, ort = defaultOrt } = {}) {
  const bundle = typeof source === "string" ? await loadBundle(source, { readFile }) : source;
  const resolved = resolveDevice(device);
  const chosen = variant !== undefined ? variant : bundle.pickVariant(resolved);
  return get("task", bundle.task)(bundle, { ort, device: resolved, variant: chosen });
}

export { loadBundle, register, resolveDevice };
