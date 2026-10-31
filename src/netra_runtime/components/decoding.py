import numpy as np

from ..registry import register


@register("decoder", "kv-cache")
class KVCacheGreedyDecoder:
    """Greedy autoregressive decoding over a ``decoder_step`` graph with a KV cache.

    Graph contract (see spec/bundle.md, "kv-cache decoder"):
        decoder_step(tokens [B, n] int64,
                     past_k, past_v   [layers, B, heads, P, head_dim],
                     cross_k, cross_v [layers, B, heads, S, head_dim],
                     cross_mask       [B, S] float32, 0 = real, -inf = padding)
            -> logits [B, n, vocab], present_k, present_v [layers, B, heads, P + n, head_dim]

    ``cross_k``/``cross_v`` come from the encoder graph and are computed once
    per input; the self-attention cache grows by one position per step, so each
    step only processes the newest token instead of the whole prefix. The cache
    tensors stay on the device between steps; only the last logits row is
    copied back to pick the next token.

    A batch decodes all its lines in lockstep: a line that has already emitted
    ``<eos>`` keeps being fed (its output is ignored) until every line is done,
    so batches work best when their lines have similar lengths.
    """

    def __init__(self, layers: int, heads: int, head_dim: int, max_len: int, **context):
        self.layers = layers
        self.heads = heads
        self.head_dim = head_dim
        self.max_len = max_len

    def decode(self, step, cross_k, cross_v, cross_mask: np.ndarray, sos_id: int, eos_id: int
               ) -> list[tuple[list[int], list[float]]]:
        """Returns, per line, the generated ids (starting with ``sos_id``) and each new token's probability."""
        batch = cross_mask.shape[0]
        empty = np.zeros((self.layers, batch, self.heads, 0, self.head_dim), dtype=np.float32)
        past_k, past_v = empty, empty
        ids = [[sos_id] for _ in range(batch)]
        probs = [[] for _ in range(batch)]
        running = np.ones(batch, dtype=bool)
        last = np.full((batch, 1), sos_id, dtype=np.int64)

        for _ in range(self.max_len):
            out = step.run(
                {"tokens": last, "past_k": past_k, "past_v": past_v,
                 "cross_k": cross_k, "cross_v": cross_v, "cross_mask": cross_mask},
                keep=("present_k", "present_v"),
            )
            logits = out["logits"][:, -1].astype(np.float64)
            next_ids = logits.argmax(axis=1)
            next_probs = 1.0 / np.exp(logits - logits[np.arange(batch), next_ids][:, None]).sum(axis=1)
            for b in np.flatnonzero(running):
                probs[b].append(float(next_probs[b]))
                if next_ids[b] == eos_id:
                    running[b] = False
                else:
                    ids[b].append(int(next_ids[b]))
            if not running.any():
                break
            last = next_ids[:, None].astype(np.int64)
            past_k, past_v = out["present_k"], out["present_v"]
        return list(zip(ids, probs))
