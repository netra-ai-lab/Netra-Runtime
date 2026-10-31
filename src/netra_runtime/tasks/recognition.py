from dataclasses import dataclass

import numpy as np

from .. import registry
from .base import Task


@dataclass
class Text:
    text: str
    # Mean probability of the generated tokens (end-of-sequence included): a
    # cheap signal for flagging lines a human should double-check.
    confidence: float


@registry.register("task", "text-recognition")
class TextRecognizer(Task):
    """Image of one text line -> ``Text``.

    Bundle graphs: ``encoder`` (preprocessed input -> decoder state) and
    ``decoder_step``; components: ``preprocess``, ``decoder``, ``tokenizer``,
    optional ``postprocess`` list.
    """

    def __init__(self, bundle, **kwargs):
        super().__init__(bundle, **kwargs)
        spec = bundle.components
        self.preprocess = registry.build("preprocess", spec["preprocess"], bundle=bundle)
        self.decoder = registry.build("decoder", spec["decoder"], bundle=bundle)
        self.tokenizer = registry.build("tokenizer", spec["tokenizer"], bundle=bundle)
        self.postprocess = [registry.build("postprocess", p, bundle=bundle)
                            for p in spec.get("postprocess", [])]
        self.encoder = self.sessions["encoder"]
        self.step = self.sessions["decoder_step"]
        self._encoder_outputs = tuple(self.encoder.output_names)

    def __call__(self, image) -> Text:
        encoded = self.encoder.run({"chunks": self.preprocess(image)}, keep=self._encoder_outputs)
        cross_k, cross_v = encoded["cross_k"], encoded["cross_v"]
        mask = np.zeros((1, cross_k.shape[3]), dtype=np.float32)
        [(ids, probs)] = self.decoder.decode(self.step, cross_k, cross_v, mask,
                                             self.tokenizer.sos_id, self.tokenizer.eos_id)
        return self._finish(ids, probs)

    def batch(self, images, batch_size: int = 16) -> list[Text]:
        """Recognize many lines, decoding up to ``batch_size`` of them together.

        Each line is encoded on its own (its encoder sequence length depends on
        its width); lines are then sorted by that length and decoded in groups,
        with shorter lines' encoder outputs zero-padded and masked out. Results
        come back in input order.
        """
        encoded = []
        for image in images:
            out = self.encoder.run({"chunks": self.preprocess(image)})
            encoded.append((out["cross_k"], out["cross_v"]))
        order = sorted(range(len(encoded)), key=lambda i: encoded[i][0].shape[3])

        results: list[Text | None] = [None] * len(encoded)
        for start in range(0, len(order), batch_size):
            group = order[start:start + batch_size]
            longest = max(encoded[i][0].shape[3] for i in group)
            ks, vs = [], []
            mask = np.zeros((len(group), longest), dtype=np.float32)
            for row, i in enumerate(group):
                k, v = encoded[i]
                pad = longest - k.shape[3]
                ks.append(np.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad), (0, 0))))
                vs.append(np.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad), (0, 0))))
                mask[row, k.shape[3]:] = -np.inf
            decoded = self.decoder.decode(self.step, np.concatenate(ks, axis=1),
                                          np.concatenate(vs, axis=1), mask,
                                          self.tokenizer.sos_id, self.tokenizer.eos_id)
            for i, (ids, probs) in zip(group, decoded):
                results[i] = self._finish(ids, probs)
        return results

    def _finish(self, ids: list[int], probs: list[float]) -> Text:
        text = self.tokenizer.decode(ids)
        for step in self.postprocess:
            text = step(text)
        return Text(text=text, confidence=sum(probs) / len(probs) if probs else 0.0)
