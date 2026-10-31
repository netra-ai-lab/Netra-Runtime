import { register } from "../registry.js";
import { release } from "../session.js";

// See spec/bundle.md "decoder: kv-cache" / "decoder: blockwise-kv". The browser
// runtime decodes one line at a time (batch size 1).

function argmaxRow(data, offset, size) {
  let best = 0;
  for (let i = 1; i < size; i++) if (data[offset + i] > data[offset + best]) best = i;
  return best;
}

function probability(data, offset, size, token) {
  let total = 0;
  const top = data[offset + token];
  for (let i = 0; i < size; i++) total += Math.exp(data[offset + i] - top);
  return 1 / total;
}

function int64(session, ids) {
  return session.tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
}

register("decoder", "kv-cache", ({ layers, heads, head_dim, max_len }) => ({
  async decode(step, cross, sosId, eosId) {
    const S = cross.cross_k.dims[3];
    const crossMask = step.tensor("float32", new Float32Array(S), [1, S]);
    let pastK = step.tensor("float32", new Float32Array(0), [layers, 1, heads, 0, head_dim]);
    let pastV = pastK;
    const ids = [sosId], probs = [];
    for (let i = 0; i < max_len; i++) {
      const out = await step.run({ tokens: int64(step, [ids.at(-1)]), past_k: pastK, past_v: pastV,
                                   cross_k: cross.cross_k, cross_v: cross.cross_v, cross_mask: crossMask });
      const vocab = out.logits.dims[2];
      const logits = out.logits.data;
      const next = argmaxRow(logits, 0, vocab);
      probs.push(probability(logits, 0, vocab, next));
      release(pastK); release(pastV);
      pastK = out.present_k; pastV = out.present_v;
      if (next === eosId) break;
      ids.push(next);
    }
    release(pastK); release(pastV);
    return { ids, probs };
  },
}));

// Keep the first `length` positions (axis 3) of a [layers, 1, heads, P, head_dim] cache.
function trimCache(session, tensor, length) {
  const [L, B, H, P, D] = tensor.dims;
  const out = new Float32Array(L * B * H * length * D);
  for (let block = 0; block < L * B * H; block++) {
    out.set(tensor.data.subarray(block * P * D, block * P * D + length * D), block * length * D);
  }
  return session.tensor("float32", out, [L, B, H, length, D]);
}

register("decoder", "blockwise-kv", ({ layers, heads, head_dim, max_len, block_size }) => {
  const stats = { rounds: 0, tokens: 0 };
  return {
    stats,
    async decode(step, cross, sosId, eosId) {
      const S = cross.cross_k.dims[3];
      const crossMask = step.tensor("float32", new Float32Array(S), [1, S]);
      let pastK = step.tensor("float32", new Float32Array(0), [layers, 1, heads, 0, head_dim]);
      let pastV = pastK;
      const ids = [sosId], probs = [];
      let proposals = [];
      while (ids.length - 1 < max_len) {
        proposals = proposals.slice(0, max_len - ids.length);
        const out = await step.run({ tokens: int64(step, [ids.at(-1), ...proposals]), past_k: pastK,
                                     past_v: pastV, cross_k: cross.cross_k, cross_v: cross.cross_v,
                                     cross_mask: crossMask });
        const vocab = out.logits.dims[2];
        const logits = out.logits.data;
        let accepted = 0;
        while (accepted < proposals.length && argmaxRow(logits, accepted * vocab, vocab) === proposals[accepted]) {
          accepted++;
        }
        const fresh = [...proposals.slice(0, accepted), argmaxRow(logits, accepted * vocab, vocab)];
        stats.rounds++;
        const cacheLength = pastK.dims[3] + accepted + 1;
        pastK = trimCache(step, out.present_k, cacheLength);
        pastV = trimCache(step, out.present_v, cacheLength);
        for (let i = 0; i < fresh.length; i++) {
          probs.push(probability(logits, i * vocab, vocab, fresh[i]));
          stats.tokens++;
          if (fresh[i] === eosId) return { ids, probs };
          ids.push(fresh[i]);
          if (ids.length - 1 >= max_len) return { ids, probs };
        }
        const n = block_size - 1;
        proposals = Array.from(out.proposal_ids.data.subarray(accepted * n, accepted * n + n), Number);
      }
      return { ids, probs };
    },
  };
});
