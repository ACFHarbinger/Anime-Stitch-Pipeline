// Bundles every markdown file under docs/ (one level up from website/) as a
// raw-string import, lazily loaded per-route. Vite resolves this glob at
// build time, so adding/removing docs just works without touching this file.
const rawDocs = import.meta.glob("../../**/*.md", { query: "?raw", import: "default" }) as Record<
  string,
  () => Promise<string>
>;
const rawNotebooks = import.meta.glob("../../**/*.ipynb", { query: "?raw", import: "default" }) as Record<
  string,
  () => Promise<string>
>;

function resolveKey(dict: Record<string, unknown>, source: string): string | undefined {
  // nav.generated.ts sources are relative to docs/ (e.g. "tutorials/index.md");
  // glob keys are relative to this file (e.g. "../../tutorials/index.md").
  return Object.keys(dict).find((k) => k.endsWith("/" + source) || k === "../../" + source);
}

export async function loadDoc(source: string): Promise<string | null> {
  const key = resolveKey(rawDocs, source);
  if (!key) return null;
  return rawDocs[key]();
}

export async function loadNotebook(source: string): Promise<string | null> {
  const key = resolveKey(rawNotebooks, source);
  if (!key) return null;
  return rawNotebooks[key]();
}

export function docExists(source: string): boolean {
  return resolveKey(rawDocs, source) !== undefined;
}
