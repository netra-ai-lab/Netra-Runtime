from dataclasses import dataclass

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
        ids, probs = self.decoder.decode(self.step, encoded,
                                         self.tokenizer.sos_id, self.tokenizer.eos_id)
        text = self.tokenizer.decode(ids)
        for step in self.postprocess:
            text = step(text)
        return Text(text=text, confidence=sum(probs) / len(probs) if probs else 0.0)

    def batch(self, images) -> list[Text]:
        # One line at a time for now; true batching needs padded encoder
        # inputs + a memory mask in the graph contract.
        return [self(image) for image in images]
