from pathlib import Path

import numpy as np
from PIL import Image

from ..registry import register


def to_pil(image) -> Image.Image:
    """Accept a path, a PIL image, or a uint8 numpy array (H x W gray or H x W x 3 RGB)."""
    if isinstance(image, (str, Path)):
        if not Path(image).exists():
            raise FileNotFoundError(f"Image not found: {image}")
        return Image.open(image)
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise TypeError(f"Unsupported image input: {type(image).__name__}")


@register("preprocess", "line-chunker")
class LineChunker:
    """Text-line image -> fixed-size overlapping chunks, ``(N, 1, height, chunk_width)`` float32.

    Grayscale, resize to ``height`` keeping the aspect ratio (never narrower
    than ``min_width``), scale to [0, 1], cut ``chunk_width``-wide windows every
    ``chunk_width - chunk_overlap`` pixels (the last one padded with
    ``pad_value``), then normalize ``(x - mean) / std``.
    """

    def __init__(self, height: int, chunk_width: int, chunk_overlap: int, min_width: int,
                 pad_value: float = 1.0, mean: float = 0.5, std: float = 0.5, **context):
        self.height = height
        self.chunk_width = chunk_width
        self.stride = chunk_width - chunk_overlap
        self.min_width = min_width
        self.pad_value = pad_value
        self.mean = mean
        self.std = std

    def __call__(self, image) -> np.ndarray:
        image = to_pil(image).convert("L")
        width = max(self.min_width, int(self.height * image.width / image.height))
        image = image.resize((width, self.height), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0

        chunks = []
        for start in range(0, width, self.stride):
            chunk = pixels[:, start:start + self.chunk_width]
            if chunk.shape[1] < self.chunk_width:
                chunk = np.pad(chunk, ((0, 0), (0, self.chunk_width - chunk.shape[1])),
                               constant_values=self.pad_value)
            chunks.append(chunk)

        batch = np.stack(chunks)[:, None, :, :]
        return ((batch - self.mean) / self.std).astype(np.float32)
