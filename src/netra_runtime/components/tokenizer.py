import json

from ..registry import register


@register("tokenizer", "vocab")
class VocabTokenizer:
    """Token id -> string table loaded from a ``{"token": id}`` JSON file.

    Decoding only: tokenization of training text belongs to the research code.
    ``<unk>`` and any other ``<...>`` special token become a space, so they can
    never leak into the output literally.
    """

    def __init__(self, vocab: str, sos: str = "<sos>", eos: str = "<eos>", pad: str = "<pad>",
                 unk: str = "<unk>", bundle=None, **context):
        token_to_id = json.loads(bundle.path(vocab).read_text(encoding="utf-8"))
        self.id_to_token = {i: t for t, i in token_to_id.items()}
        self.sos_id = token_to_id[sos]
        self.eos_id = token_to_id[eos]
        self.pad_id = token_to_id[pad]
        self.unk_id = token_to_id[unk]

    def __len__(self):
        return len(self.id_to_token)

    def decode(self, ids: list[int]) -> str:
        pieces = []
        for i in ids:
            if i in (self.sos_id, self.pad_id):
                continue
            if i == self.eos_id:
                break
            token = self.id_to_token.get(i, "")
            if i == self.unk_id or (len(token) > 1 and token.startswith("<") and token.endswith(">")):
                pieces.append(" ")
            else:
                pieces.append(token)
        return "".join(pieces)
