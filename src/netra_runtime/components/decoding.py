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


def _probability(logits: np.ndarray, token: int) -> float:
    logits = logits.astype(np.float64)
    return float(1.0 / np.exp(logits - logits[token]).sum())


@register("decoder", "blockwise-kv")
class BlockwiseKVDecoder(KVCacheGreedyDecoder):
    """Blockwise parallel decoding (Stern et al. 2018) on top of the KV cache.

    Same graph contract as ``kv-cache`` plus one output:
        proposal_ids [B, n, block_size - 1] int64 -- head j's guess for the
        token j + 2 positions after each input position.

    Each round feeds the last accepted token followed by the previous round's
    proposals. The base model's own next-token prediction (``logits``) at each
    of those positions verifies them: proposals are accepted up to the first
    disagreement, then the base model's prediction at that point is appended
    too, and the proposal heads at that same position supply the next round's
    guesses. Every emitted token is therefore exactly what greedy decoding
    would emit; the proposals only decide how many arrive per forward pass.
    The cache is trimmed back to the accepted positions after each round.

    ``rounds`` / ``tokens`` count forward passes and emitted tokens since the
    decoder was built (``tokens / rounds`` = mean accepted block size).
    """

    def __init__(self, block_size: int, **options):
        super().__init__(**options)
        self.block_size = block_size
        self.rounds = 0
        self.tokens = 0

    def decode(self, step, cross_k, cross_v, cross_mask: np.ndarray, sos_id: int, eos_id: int):
        if cross_mask.shape[0] == 1:
            return [self._decode_line(step, cross_k, cross_v, cross_mask, sos_id, eos_id)]
        # Lines accept different numbers of tokens per round, so their caches
        # diverge in length: decode a batch one line at a time.
        results = []
        for b in range(cross_mask.shape[0]):
            length = int(np.isfinite(cross_mask[b]).sum())
            results.append(self._decode_line(
                step, cross_k[:, b:b + 1, :, :length], cross_v[:, b:b + 1, :, :length],
                cross_mask[b:b + 1, :length], sos_id, eos_id))
        return results

    def _decode_line(self, step, cross_k, cross_v, cross_mask, sos_id, eos_id):
        empty = np.zeros((self.layers, 1, self.heads, 0, self.head_dim), dtype=np.float32)
        past_k, past_v = empty, empty
        generated, probs, proposals = [sos_id], [], []

        # The cache always holds every generated token except the last one.
        while len(generated) - 1 < self.max_len:
            # Never feed a position past max_len - 1 (the position table's end).
            proposals = proposals[:self.max_len - len(generated)]
            out = step.run({"tokens": np.array([[generated[-1], *proposals]], dtype=np.int64),
                            "past_k": past_k, "past_v": past_v,
                            "cross_k": cross_k, "cross_v": cross_v, "cross_mask": cross_mask})
            logits = out["logits"][0]
            predicted = logits.argmax(axis=-1)
            accepted = 0
            while accepted < len(proposals) and predicted[accepted] == proposals[accepted]:
                accepted += 1
            new_tokens = [*proposals[:accepted], int(predicted[accepted])]
            self.rounds += 1

            cache_len = past_k.shape[3] + accepted + 1
            past_k = out["present_k"][:, :, :, :cache_len]
            past_v = out["present_v"][:, :, :, :cache_len]

            for i, token in enumerate(new_tokens):
                probs.append(_probability(logits[i], token))
                self.tokens += 1
                if token == eos_id:
                    return generated, probs
                generated.append(token)
                if len(generated) - 1 >= self.max_len:
                    return generated, probs
            proposals = [int(t) for t in out["proposal_ids"][0, accepted]]
        return generated, probs
