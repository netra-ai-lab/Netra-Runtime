// Reads a bundle (spec/bundle.md) from a URL, or through any `readFile(name)`
// function (e.g. node:fs in tests, or files the user dropped into the page).

export const MANIFEST_NAME = "netra-model.json";
export const FORMAT = "netra-bundle";
export const FORMAT_VERSION = 1;

function urlReader(base) {
  const root = base.endsWith("/") ? base : `${base}/`;
  return async (name) => {
    const response = await fetch(new URL(name, new URL(root, globalThis.location?.href)));
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return response.arrayBuffer();
  };
}

export class Bundle {
  constructor(manifest, readFile) {
    this.manifest = manifest;
    this.readFile = readFile;
    this.name = manifest.name;
    this.version = manifest.version;
    this.task = manifest.task;
    this.components = manifest.components;
    this.variants = manifest.variants ?? {};
  }

  graphFiles(variant = null) {
    const graphs = { ...this.manifest.graphs };
    if (variant !== null) {
      if (!(variant in this.variants)) throw new Error(`Bundle has no variant '${variant}'`);
      Object.assign(graphs, this.variants[variant].graphs ?? {});
    }
    return graphs;
  }

  pickVariant(device) {
    for (const [name, variant] of Object.entries(this.variants)) {
      if ((variant.devices ?? []).includes(device)) return name;
    }
    return null;
  }

  async bytes(name) {
    if (name.includes("..") || name.startsWith("/")) throw new Error(`'${name}' points outside the bundle`);
    return new Uint8Array(await this.readFile(name));
  }

  async json(name) {
    return JSON.parse(new TextDecoder().decode(await this.bytes(name)));
  }
}

/** "hf://<org>/<repo>[/<folder>][@revision]" -> the bundle's file URL prefix on huggingface.co. */
export function resolveSource(source) {
  if (!source.startsWith("hf://")) return source;
  const [path, revision = "main"] = source.slice("hf://".length).split("@");
  const [org, repo, ...folder] = path.split("/").filter(Boolean);
  return `https://huggingface.co/${org}/${repo}/resolve/${revision}/${folder.length ? folder.join("/") + "/" : ""}`;
}

export async function loadBundle(source, { readFile } = {}) {
  const reader = readFile ?? urlReader(resolveSource(source));
  const manifest = JSON.parse(new TextDecoder().decode(new Uint8Array(await reader(MANIFEST_NAME))));
  if (manifest.format !== FORMAT) throw new Error(`${MANIFEST_NAME} is not a ${FORMAT} manifest`);
  if (manifest.format_version !== FORMAT_VERSION) {
    throw new Error(`Unsupported format_version ${manifest.format_version} (this runtime reads ${FORMAT_VERSION})`);
  }
  for (const key of ["name", "version", "task", "graphs", "components"]) {
    if (!(key in manifest)) throw new Error(`${MANIFEST_NAME} is missing '${key}'`);
  }
  return new Bundle(manifest, reader);
}
