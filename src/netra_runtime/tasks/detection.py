from dataclasses import dataclass

from .. import registry
from .base import Task


@dataclass
class Detection:
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in original image pixels
    score: float
    label: str
    class_id: int


@registry.register("task", "object-detection")
class ObjectDetector(Task):
    """Image -> list of ``Detection`` (any number of classes).

    Bundle graphs: ``model`` (``images`` (1, 3, H, W) -> ``output0``);
    components: ``preprocess`` (with a ``restore()`` mapping boxes back to the
    original image) and ``head`` (raw output -> boxes, scores, class ids).
    """

    def __init__(self, bundle, **kwargs):
        super().__init__(bundle, **kwargs)
        spec = bundle.components
        self.preprocess = registry.build("preprocess", spec["preprocess"], bundle=bundle, device=self.device)
        self.head = registry.build("head", spec["head"], bundle=bundle)
        self.labels = self.head.labels
        self.model = self.sessions["model"]

    def __call__(self, image, conf: float | None = None) -> list[Detection]:
        tensor, info = self.preprocess(image)
        output = self.model.run({"images": tensor})["output0"]
        boxes, scores, classes = self.head(output, conf=conf)
        boxes = self.preprocess.restore(boxes, info)
        return [Detection(tuple(float(v) for v in box), float(score), self.labels[int(c)], int(c))
                for box, score, c in zip(boxes, scores, classes)]

    def batch(self, images, conf: float | None = None) -> list[list[Detection]]:
        return [self(image, conf=conf) for image in images]
