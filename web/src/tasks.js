import { build, register } from "./registry.js";
import { Session, release } from "./session.js";
import { toPixels } from "./image.js";

async function sessions(ort, bundle, device, variant, keep = {}) {
  const out = {};
  for (const [role, file] of Object.entries(bundle.graphFiles(variant))) {
    out[role] = await Session.create(ort, await bundle.bytes(file), device, keep[role] ?? []);
  }
  return out;
}

async function jsonFiles(bundle, names) {
  const files = {};
  for (const name of names) files[name] = await bundle.json(name);
  return files;
}

// text-recognition: image of one line -> {text, confidence}
register("task", "text-recognition", async (bundle, { ort, device, variant }) => {
  const spec = bundle.components;
  const files = await jsonFiles(bundle, [spec.tokenizer.vocab]);
  const context = { device, files };
  const preprocess = build("preprocess", spec.preprocess, context);
  const decoder = build("decoder", spec.decoder, context);
  const tokenizer = build("tokenizer", spec.tokenizer, context);
  const postprocess = (spec.postprocess ?? []).map((p) => build("postprocess", p, context));
  // Plain greedy keeps the KV cache in GPU memory between steps; blockwise
  // trims the cache on the CPU every round, so it reads it back anyway.
  const cacheOnGpu = spec.decoder.type === "kv-cache";
  const { encoder, decoder_step: step } = await sessions(ort, bundle, device, variant, {
    encoder: ["cross_k", "cross_v"],
    decoder_step: cacheOnGpu ? ["present_k", "present_v"] : [],
  });

  async function recognize(image) {
    const { data, dims } = preprocess(await toPixels(image));
    const cross = await encoder.run({ chunks: encoder.tensor("float32", data, dims) });
    const { ids, probs } = await decoder.decode(step, cross, tokenizer.sosId, tokenizer.eosId);
    release(cross.cross_k);
    release(cross.cross_v);
    let text = tokenizer.decode(ids);
    for (const clean of postprocess) text = clean(text);
    return { text, confidence: probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : 0 };
  }

  return {
    bundle, device, variant, decoder,
    recognize,
    async batch(images) {
      const results = [];
      for (const image of images) results.push(await recognize(image));
      return results;
    },
  };
});

// object-detection: image -> [{box: [x1, y1, x2, y2], score, label, classId}]
register("task", "object-detection", async (bundle, { ort, device, variant }) => {
  const spec = bundle.components;
  const context = { device };
  const preprocess = build("preprocess", spec.preprocess, context);
  const head = build("head", spec.head, context);
  const { model } = await sessions(ort, bundle, device, variant);

  return {
    bundle, device, variant, labels: head.labels,
    async detect(image, { conf } = {}) {
      const { tensor, info } = preprocess(await toPixels(image));
      const out = await model.run({ images: model.tensor("float32", tensor.data, tensor.dims) });
      const found = head.decode(out.output0, conf);
      const boxes = preprocess.restore(found.map((d) => d.box), info);
      return found.map((d, i) => ({ box: boxes[i], score: d.score, label: head.labels[d.classId], classId: d.classId }));
    },
  };
});
