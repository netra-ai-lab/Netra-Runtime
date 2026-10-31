import { register } from "../registry.js";
import { resizeGrayPIL, resizeRGBLinear, toGray, toRGB } from "../image.js";

const f32 = Math.fround;

// See spec/bundle.md "preprocess: line-chunker". Arithmetic is rounded to
// float32 after every operation, matching the Python runtime's numpy float32.
register("preprocess", "line-chunker", ({ height, chunk_width, chunk_overlap, min_width,
                                         pad_value = 1.0, mean = 0.5, std = 0.5 }) => (pixels) => {
  const width = Math.max(min_width, Math.trunc((height * pixels.width) / pixels.height));
  const gray = resizeGrayPIL(toGray(pixels), pixels.width, pixels.height, width, height);
  const stride = chunk_width - chunk_overlap;
  const count = Math.ceil(width / stride);
  const data = new Float32Array(count * height * chunk_width);
  const padded = f32(f32(f32(pad_value) - f32(mean)) / f32(std));
  for (let c = 0; c < count; c++) {
    const start = c * stride;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < chunk_width; x++) {
        const sx = start + x;
        data[(c * height + y) * chunk_width + x] = sx < width
          ? f32(f32(f32(gray[y * width + sx] / 255) - f32(mean)) / f32(std))
          : padded;
      }
    }
  }
  return { data, dims: [count, 1, height, chunk_width] };
});

// See spec/bundle.md "preprocess: letterbox". On WebGPU the image is padded to
// the full square: fixed input shapes avoid per-shape kernel setup.
register("preprocess", "letterbox", ({ imgsz, stride = 32, auto = true, pad_value = 114, scaleup = true },
                                     { device }) => {
  const rect = auto && device !== "webgpu";
  const preprocess = (pixels) => {
    const { width: w, height: h } = pixels;
    let r = Math.min(imgsz / h, imgsz / w);
    if (!scaleup) r = Math.min(r, 1.0);
    const newW = Math.max(roundHalfEven(w * r), 1), newH = Math.max(roundHalfEven(h * r), 1);
    let dw = imgsz - newW, dh = imgsz - newH;
    if (rect) { dw %= stride; dh %= stride; }
    dw /= 2; dh /= 2;
    const top = roundHalfEven(dh - 0.1), bottom = roundHalfEven(dh + 0.1);
    const left = roundHalfEven(dw - 0.1), right = roundHalfEven(dw + 0.1);

    let rgb = toRGB(pixels);
    if (newW !== w || newH !== h) rgb = resizeRGBLinear(rgb, w, h, newW, newH);
    const outW = newW + left + right, outH = newH + top + bottom;
    const plane = outW * outH;
    const data = new Float32Array(3 * plane).fill(pad_value / 255);
    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) {
        const p = (y * newW + x) * 3, q = (y + top) * outW + x + left;
        data[q] = rgb[p] / 255;
        data[plane + q] = rgb[p + 1] / 255;
        data[2 * plane + q] = rgb[p + 2] / 255;
      }
    }
    const info = { origW: w, origH: h, gainX: newW / w, gainY: newH / h, padX: left, padY: top };
    return { tensor: { data, dims: [1, 3, outH, outW] }, info };
  };
  preprocess.restore = (boxes, info) => boxes.map(([x1, y1, x2, y2]) => [
    clamp((x1 - info.padX) / info.gainX, 0, info.origW), clamp((y1 - info.padY) / info.gainY, 0, info.origH),
    clamp((x2 - info.padX) / info.gainX, 0, info.origW), clamp((y2 - info.padY) / info.gainY, 0, info.origH),
  ]);
  return preprocess;
});

// Python's round(): halves go to the even neighbour.
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function clamp(v, lo, hi) {
  return Math.min(Math.max(v, lo), hi);
}
