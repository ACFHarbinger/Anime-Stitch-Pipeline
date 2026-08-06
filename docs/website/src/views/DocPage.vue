<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted } from "vue";
import { useRoute } from "vue-router";
import { searchIndex } from "../nav.generated";
import { loadDoc } from "../composables/useDocs";
import { renderMarkdown, extractTitle, extractToc, type TocEntry } from "../composables/useMarkdown";
import NotebookView from "../components/NotebookView.vue";

const route = useRoute();

const state = ref<"loading" | "ok" | "notfound">("loading");
const html = ref("");
const rawSource = ref("");
const title = ref("");
const toc = ref<TocEntry[]>([]);
const notebookSource = ref<string | null>(null);
const editSource = ref("");

const REPO_EDIT_BASE = "https://github.com/ACFHarbinger/Anime-Stitch-Pipeline/blob/main/docs/";

const currentPath = computed(() => {
  const p = "/" + (Array.isArray(route.params.pathMatch) ? route.params.pathMatch.join("/") : "");
  return p === "//" ? "/" : p.replace(/\/$/, "") || "/";
});

async function load() {
  state.value = "loading";
  notebookSource.value = null;
  const entry = searchIndex.find((p) => p.path === currentPath.value);
  if (!entry) {
    state.value = "notfound";
    return;
  }
  editSource.value = entry.source;

  if (entry.kind === "notebook") {
    notebookSource.value = entry.source;
    title.value = entry.title;
    toc.value = [];
    document.title = `${title.value} — Anime-Stitch-Pipeline Docs`;
    state.value = "ok";
    return;
  }

  const raw = await loadDoc(entry.source);
  if (raw === null) {
    state.value = "notfound";
    return;
  }
  rawSource.value = raw;
  title.value = extractTitle(raw) || entry.title;
  toc.value = extractToc(raw);
  html.value = renderMarkdown(raw);
  state.value = "ok";
  await nextTick();
  renderMermaid();
  document.title = `${title.value} — Anime-Stitch-Pipeline Docs`;
}

async function renderMermaid() {
  const nodes = document.querySelectorAll<HTMLElement>(".markdown-body .mermaid");
  if (!nodes.length) return;
  const mermaid = (await import("mermaid")).default;
  mermaid.initialize({
    startOnLoad: false,
    theme: document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "default",
    securityLevel: "loose",
  });
  try {
    await mermaid.run({ nodes: Array.from(nodes) });
  } catch (e) {
    console.warn("mermaid render failed", e);
  }
}

watch(() => route.fullPath, load);
onMounted(load);
</script>

<template>
  <div class="doc-page">
    <div v-if="state === 'loading'" class="doc-loading">Loading…</div>

    <div v-else-if="state === 'notfound'" class="doc-notfound">
      <h1>404</h1>
      <p>No page found for <code>{{ currentPath }}</code>.</p>
      <router-link to="/">← Back home</router-link>
    </div>

    <template v-else>
      <NotebookView v-if="notebookSource" :source="notebookSource" :title="title" />
      <article v-else class="doc-content">
        <div class="markdown-body" v-html="html"></div>
        <a class="edit-link" :href="REPO_EDIT_BASE + editSource" target="_blank" rel="noopener noreferrer">
          Edit this page on GitHub ↗
        </a>
      </article>

      <aside v-if="toc.length" class="doc-toc">
        <div class="doc-toc-title">On this page</div>
        <nav>
          <a
            v-for="entry in toc"
            :key="entry.id"
            :href="'#' + entry.id"
            :class="['toc-link', `toc-level-${entry.level}`]"
          >
            {{ entry.text }}
          </a>
        </nav>
      </aside>
    </template>
  </div>
</template>

<style scoped>
.doc-page {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 220px;
  gap: 2.5rem;
  align-items: start;
  max-width: 1100px;
  margin: 0 auto;
  padding: 2.5rem 2rem 6rem;
}

.doc-content {
  min-width: 0;
}

.doc-loading,
.doc-notfound {
  padding: 4rem 2rem;
  text-align: center;
  color: var(--text-muted);
}

.edit-link {
  display: inline-block;
  margin-top: 3rem;
  padding-top: 1.25rem;
  border-top: 1px solid var(--border);
  font-size: 0.875rem;
  color: var(--text-muted);
}
.edit-link:hover {
  color: var(--accent);
}

.doc-toc {
  position: sticky;
  top: 5.5rem;
  font-size: 0.8125rem;
}
.doc-toc-title {
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.6875rem;
  color: var(--text-muted);
  margin-bottom: 0.75rem;
}
.doc-toc nav {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  border-left: 1px solid var(--border);
}
.toc-link {
  color: var(--text-muted);
  padding-left: 0.875rem;
  line-height: 1.4;
}
.toc-link:hover {
  color: var(--text);
}
.toc-level-3 {
  padding-left: 1.75rem;
  font-size: 0.75rem;
}

@media (max-width: 980px) {
  .doc-page {
    grid-template-columns: 1fr;
    padding: 1.5rem 1.25rem 4rem;
  }
  .doc-toc {
    display: none;
  }
}
</style>
