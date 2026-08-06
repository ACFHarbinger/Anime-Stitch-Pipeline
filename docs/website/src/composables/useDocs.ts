// Bundles every markdown file under docs/ as a raw-string import, lazily
// loaded per-route. Vite resolves glob patterns relative to *this file's*
// directory (src/composables/), not to the website/ project root -- three
// ".." steps (composables/ -> src/ -> website/ -> docs/) are required to
// reach docs/, where every source in nav.generated.ts is rooted.
const rawDocs = import.meta.glob("../../../**/*.md", { query: "?raw", import: "default" }) as Record<
  string,
  () => Promise<string>
>;
const rawNotebooks = import.meta.glob("../../../**/*.ipynb", { query: "?raw", import: "default" }) as Record<
  string,
  () => Promise<string>
>;

function resolveKey(dict: Record<string, unknown>, source: string): string | undefined {
  // nav.generated.ts sources are relative to docs/ (e.g. "tutorials/index.md" or
  // "index.md"); glob keys are relative to this file, with a leading run of
  // "../" segments (e.g. "../../../tutorials/index.md"). Matching by
  // endsWith("/" + source) is ambiguous whenever two files share a basename
  // at different depths (e.g. docs/index.md vs docs/api/kotlin/index.md) --
  // strip the leading ".." run and require an exact match instead.
  return Object.keys(dict).find((k) => k.replace(/^(\.\.\/)+/, "") === source);
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
