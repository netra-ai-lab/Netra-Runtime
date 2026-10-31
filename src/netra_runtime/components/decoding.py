import numpy as np

from ..registry import register


@register("decoder", "kv-cache")
class KVCacheGreedyDecoder:
    """Greedy autoregressive decoding over a ``decoder_step`` graph with a KV cache.

    Graph contract (see spec/bundle.md, "kv-cache decoder"):
        decoder_step(tokens [B, n] int64,
                     past_k, past_v   [layers, B, heads, P, head_dim],
                     cross_k, cross_v [layers, B, heads, S, head_dim])
            -> logits [B, n, vocab], present_k, present_v [layers, B, heads, P + n, head_dim]

    ``cross_k``/``cross_v`` come from the encoder graph and are computed once
    per input; the self-attention cache grows by one position per step, so each
    step only processes the newest token instead of the whole prefix. The cache
    tensors stay on the device between steps; only the last logits row is
    copied back to pick the next token.
    """

    def __init__(self, layers: int, heads: int, head_dim: int, max_len: int, **context):
        self.layers = layers
        self.heads = heads
        self.head_dim = head_dim
        self.max_len = max_len

    def decode(self, step, encoded: dict, sos_id: int, eos_id: int) -> tuple[list[int], list[float]]:
        """Returns the generated ids (starting with ``sos_id``) and each new token's probability."""
        empty = np.zeros((self.layers, 1, self.heads, 0, self.head_dim), dtype=np.float32)
        past_k, past_v = empty, empty
        generated, probs = [sos_id], []
        for _ in range(self.max_len):
            out = step.run(
                {"tokens": np.array([[generated[-1]]], dtype=np.int64),
                 "past_k": past_k, "past_v": past_v,
                 "cross_k": encoded["cross_k"], "cross_v": encoded["cross_v"]},
                keep=("present_k", "present_v"),
            )
            logits = out["logits"][0, -1].astype(np.float64)
            next_id = int(np.argmax(logits))
            probs.append(float(1.0 / np.exp(logits - logits[next_id]).sum()))
            if next_id == eos_id:
                break
            generated.append(next_id)
            past_k, past_v = out["present_k"], out["present_v"]
        return generated, probs
