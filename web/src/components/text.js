import { register } from "../registry.js";

// See spec/bundle.md "tokenizer: vocab".
register("tokenizer", "vocab", (
  { vocab, sos = "<sos>", eos = "<eos>", pad = "<pad>", unk = "<unk>" }, { files },
) => {
  const tokenToId = files[vocab];
  const idToToken = new Map(Object.entries(tokenToId).map(([token, id]) => [id, token]));
  const ids = { sos: tokenToId[sos], eos: tokenToId[eos], pad: tokenToId[pad], unk: tokenToId[unk] };
  return {
    sosId: ids.sos,
    eosId: ids.eos,
    decode(tokens) {
      let text = "";
      for (const id of tokens) {
        if (id === ids.sos || id === ids.pad) continue;
        if (id === ids.eos) break;
        const token = idToToken.get(id) ?? "";
        const special = [...token].length > 1 && token.startsWith("<") && token.endsWith(">");
        text += id === ids.unk || special ? " " : token;
      }
      return text;
    },
  };
});

const NOISE = /[�\x00-\x08\x0b\x0c\x0e-\x1f]/gu;
const REPEAT_CLUSTER = /(\S{2,20}?)\1{1,}/gu;
const REPEAT_TOKEN = /(\S+)(?:\s+\1)+/gu;
const SPACES = /\s{2,}/gu;

function isMrzLine(text) {
  const stripped = [...text.replaceAll(" ", "")];
  if (stripped.length < 10 || !stripped.join("").includes("<<")) return false;
  return stripped.filter((c) => /[A-Z0-9<]/.test(c)).length / stripped.length >= 0.85;
}

// See spec/bundle.md "postprocess: ocr-text-cleanup".
register("postprocess", "ocr-text-cleanup", ({ max_repeat = 3 }) => {
  const charRun = new RegExp(`(.)\\1{${max_repeat},}`, "gu");
  return (input) => {
    let text = input.replace(NOISE, "");
    if (isMrzLine(text)) return text.replace(SPACES, " ").trim();
    text = text.replace(charRun, "$1");
    for (let i = 0; i < 3; i++) {
      const collapsed = text.replace(REPEAT_CLUSTER, "$1");
      if (collapsed === text) break;
      text = collapsed;
    }
    return text.replace(REPEAT_TOKEN, "$1").replace(SPACES, " ").trim();
  };
});
