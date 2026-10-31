# Importing the modules registers their tasks.
from .detection import Detection, ObjectDetector
from .recognition import Text, TextRecognizer

__all__ = ["Detection", "ObjectDetector", "Text", "TextRecognizer"]
