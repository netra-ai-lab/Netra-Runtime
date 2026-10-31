import re

from ..registry import register

_NOISE_CHARS = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f]")
# A 2-20 char cluster repeated inside a word, e.g. "រាំរាំ" -> "រាំ".
_REPEAT_CLUSTER = re.compile(r"(\S{2,20}?)\1{1,}")
# The same whitespace-separated token repeated, e.g. "H H H" -> "H".
_REPEAT_TOKEN = re.compile(r"(\S+)(?:\s+\1)+")
_MRZ_CHARS = re.compile(r"[A-Z0-9<]")


def _is_mrz_line(text: str) -> bool:
    stripped = text.replace(" ", "")
    if len(stripped) < 10 or "<<" not in stripped:
        return False
    return len(_MRZ_CHARS.findall(stripped)) / len(stripped) >= 0.85


@register("postprocess", "ocr-text-cleanup")
class OCRTextCleanup:
    """Removes autoregressive-decoder artifacts from recognized text: control
    characters, runaway repeats of a character / cluster / word, and extra
    whitespace. Passport MRZ lines keep their legitimate ``<<<`` filler.
    """

    def __init__(self, max_repeat: int = 3, **context):
        self.char_run = re.compile(r"(.)\1{" + str(max_repeat) + r",}")

    def __call__(self, text: str) -> str:
        text = _NOISE_CHARS.sub("", text)
        if _is_mrz_line(text):
            return re.sub(r"\s{2,}", " ", text).strip()
        text = self.char_run.sub(r"\1", text)
        for _ in range(3):
            collapsed = _REPEAT_CLUSTER.sub(r"\1", text)
            if collapsed == text:
                break
            text = collapsed
        text = _REPEAT_TOKEN.sub(r"\1", text)
        return re.sub(r"\s{2,}", " ", text).strip()
