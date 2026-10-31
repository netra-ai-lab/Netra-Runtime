import { register } from "../registry.js";

// See spec/bundle.md "head: yolo-nms" / "head: yolo-end2end". Each head turns
// the graph output into [{box: [x1, y1, x2, y2], score, classId}] in input pixels.

function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

register("head", "yolo-nms", ({ labels, conf = 0.25, iou: iouThreshold = 0.7, max_det = 300,
                                max_nms = 30000, agnostic = false }) => ({
  labels,
  decode(output, confOverride) {
    const threshold = confOverride ?? conf;
    const [, rows, anchors] = output.dims;
    const data = output.data;
    let candidates = [];
    for (let a = 0; a < anchors; a++) {
      let classId = 0, score = data[4 * anchors + a];
      for (let c = 1; c < rows - 4; c++) {
        const s = data[(4 + c) * anchors + a];
        if (s > score) { score = s; classId = c; }
      }
      if (score <= threshold) continue;
      const cx = data[a], cy = data[anchors + a], w = data[2 * anchors + a], h = data[3 * anchors + a];
      candidates.push({ box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], score, classId });
    }
    candidates.sort((x, y) => y.score - x.score);   // stable: ties keep anchor order
    candidates = candidates.slice(0, max_nms);
    const kept = [];
    for (const cand of candidates) {
      if (kept.length >= max_det) break;
      if (kept.every((k) => (!agnostic && k.classId !== cand.classId) || iou(k.box, cand.box) <= iouThreshold)) {
        kept.push(cand);
      }
    }
    return kept;
  },
}));

register("head", "yolo-end2end", ({ labels, conf = 0.25 }) => ({
  labels,
  decode(output, confOverride) {
    const threshold = confOverride ?? conf;
    const [, rows] = output.dims;
    const data = output.data;
    const found = [];
    for (let r = 0; r < rows; r++) {
      const o = r * 6;
      if (data[o + 4] > threshold) {
        found.push({ box: [data[o], data[o + 1], data[o + 2], data[o + 3]], score: data[o + 4], classId: data[o + 5] });
      }
    }
    return found;
  },
}));
