"""Detection building blocks: letterbox preprocessing and box heads for
YOLO-family graphs (raw outputs + NMS, or NMS-free end-to-end outputs)."""

from dataclasses import dataclass

import numpy as np

from ..registry import register
from .preprocess import to_pil

try:
    import cv2
except ImportError:  # optional: a pure-numpy resize is used instead
    cv2 = None


def to_rgb_array(image) -> np.ndarray:
    """Path / PIL image / uint8 array (H x W gray or H x W x 3 RGB) -> H x W x 3 uint8 RGB."""
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image, dtype=np.uint8)
    return np.asarray(to_pil(image).convert("RGB"))


def resize_bilinear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize with half-pixel centers and no antialiasing (OpenCV INTER_LINEAR semantics)."""
    if cv2 is not None:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    src_h, src_w = image.shape[:2]

    def axis(dst, src):
        pos = np.clip((np.arange(dst) + 0.5) * (src / dst) - 0.5, 0, src - 1)
        lo = np.floor(pos).astype(np.int64)
        return lo, np.minimum(lo + 1, src - 1), (pos - lo).astype(np.float32)

    y0, y1, wy = axis(height, src_h)
    x0, x1, wx = axis(width, src_w)
    img = image.astype(np.float32)
    top = img[y0][:, x0] * (1 - wx)[None, :, None] + img[y0][:, x1] * wx[None, :, None]
    bottom = img[y1][:, x0] * (1 - wx)[None, :, None] + img[y1][:, x1] * wx[None, :, None]
    out = top * (1 - wy)[:, None, None] + bottom * wy[:, None, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


@dataclass
class LetterboxInfo:
    orig_h: int
    orig_w: int
    gain_x: float
    gain_y: float
    pad_x: int
    pad_y: int


@register("preprocess", "letterbox")
class Letterbox:
    """Resize keeping the aspect ratio so the long side is ``imgsz``, then pad
    with ``pad_value`` -- to a multiple of ``stride`` (``auto``, the minimum
    rectangle) or to ``imgsz`` x ``imgsz``. Output: ``(1, 3, H, W)`` float32 RGB in [0, 1].
    """

    def __init__(self, imgsz: int, stride: int = 32, auto: bool = True, pad_value: int = 114,
                 scaleup: bool = True, **context):
        self.imgsz = imgsz
        self.stride = stride
        self.auto = auto
        self.pad_value = pad_value
        self.scaleup = scaleup

    def __call__(self, image) -> tuple[np.ndarray, LetterboxInfo]:
        img = to_rgb_array(image)
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        if not self.scaleup:
            r = min(r, 1.0)
        new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
        dw, dh = self.imgsz - new_w, self.imgsz - new_h
        if self.auto:
            dw, dh = dw % self.stride, dh % self.stride
        dw, dh = dw / 2, dh / 2
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)

        if (new_w, new_h) != (w, h):
            img = resize_bilinear(img, new_w, new_h)
        padded = np.full((new_h + top + bottom, new_w + left + right, 3), self.pad_value, np.uint8)
        padded[top:top + new_h, left:left + new_w] = img

        tensor = (padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        return tensor, LetterboxInfo(h, w, new_w / w, new_h / h, left, top)

    @staticmethod
    def restore(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
        """Map xyxy boxes from the letterboxed input back to original pixels (clipped)."""
        boxes = boxes.astype(np.float32).copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_x) / info.gain_x
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_y) / info.gain_y
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, info.orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, info.orig_h)
        return boxes


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices in descending score order."""
    order = np.argsort(-scores, kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        overlap = inter / (areas[i] + areas[rest] - inter)
        order = rest[overlap <= iou]
    return np.array(keep, dtype=np.int64)


@register("head", "yolo-nms")
class YoloNMSHead:
    """Raw YOLO output ``(1, 4 + classes, anchors)`` (cx, cy, w, h, class scores)
    -> boxes after best-class selection and class-aware NMS."""

    def __init__(self, labels: list[str], conf: float = 0.25, iou: float = 0.7, max_det: int = 300,
                 max_nms: int = 30000, agnostic: bool = False, **context):
        self.labels = labels
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.max_nms = max_nms
        self.agnostic = agnostic

    def __call__(self, output: np.ndarray, conf: float | None = None):
        conf = self.conf if conf is None else conf
        pred = output[0].T                                   # (anchors, 4 + classes)
        scores_all = pred[:, 4:]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(pred)), cls]
        keep = scores > conf
        pred, cls, scores = pred[keep], cls[keep], scores[keep]
        if len(pred) > self.max_nms:
            top = np.argsort(-scores, kind="stable")[:self.max_nms]
            pred, cls, scores = pred[top], cls[top], scores[top]
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        # Class-aware NMS by offsetting each class into its own coordinate range.
        offset = 0 if self.agnostic else cls[:, None].astype(np.float32) * 7680
        kept = nms(boxes + offset, scores, self.iou)[:self.max_det]
        return boxes[kept], scores[kept], cls[kept]


@register("head", "yolo-end2end")
class YoloEnd2EndHead:
    """NMS-free YOLO output ``(1, max_det, 6)`` (x1, y1, x2, y2, score, class) -> boxes above ``conf``."""

    def __init__(self, labels: list[str], conf: float = 0.25, **context):
        self.labels = labels
        self.conf = conf

    def __call__(self, output: np.ndarray, conf: float | None = None):
        conf = self.conf if conf is None else conf
        pred = output[0]
        pred = pred[pred[:, 4] > conf]
        return pred[:, :4], pred[:, 4], pred[:, 5].astype(np.int64)
