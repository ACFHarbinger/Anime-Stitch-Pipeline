<script setup lang="ts">
import { ref, watch, onMounted, nextTick } from "vue";
import { loadNotebook } from "../composables/useDocs";
import { renderMarkdown } from "../composables/useMarkdown";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
hljs.registerLanguage("python", python);

const props = defineProps<{ source: string; title: string }>();

interface NBCell {
  cell_type: "markdown" | "code" | "raw";
  source: string[] | string;
  outputs?: any[];
}

const cells = ref<NBCell[]>([]);
const notFound = ref(false);

function joinSrc(s: string[] | string): string {
  return Array.isArray(s) ? s.join("") : s;
}

function highlightCode(code: string): string {
  try {
    return hljs.highlight(code, { language: "python" }).value;
  } catch {
    return code;
  }
}

function renderOutput(output: any): string | null {
  if (output.output_type === "stream") {
    return `<pre class="nb-stream">${joinSrc(output.text)}</pre>`;
  }
  if (output.output_type === "execute_result" || output.output_type === "display_data") {
    const data = output.data || {};
    if (data["image/png"]) {
      return `<img class="nb-img" src="data:image/png;base64,${data["image/png"]}" />`;
    }
    if (data["text/html"]) {
      return `<div class="nb-html">${joinSrc(data["text/html"])}</div>`;
    }
    if (data["text/plain"]) {
      return `<pre class="nb-stream">${joinSrc(data["text/plain"])}</pre>`;
    }
  }
  if (output.output_type === "error") {
    return `<pre class="nb-error">${(output.traceback || []).join("\n")}</pre>`;
  }
  return null;
}

async function load() {
  notFound.value = false;
  cells.value = [];
  const raw = await loadNotebook(props.source);
  if (!raw) {
    notFound.value = true;
    return;
  }
  try {
    const nb = JSON.parse(raw);
    cells.value = nb.cells || [];
  } catch {
    notFound.value = true;
  }
  await nextTick();
}

watch(() => props.source, load);
onMounted(load);
</script>

<template>
  <article class="notebook">
    <h1>{{ title }}</h1>
    <p class="nb-hint">Rendered from a Jupyter notebook — outputs are static (captured at commit time, not re-executed).</p>

    <div v-if="notFound" class="doc-notfound">
      <p>Could not load notebook <code>{{ source }}</code>.</p>
    </div>

    <div v-for="(cell, i) in cells" :key="i" class="nb-cell">
      <div v-if="cell.cell_type === 'markdown'" class="markdown-body" v-html="renderMarkdown(joinSrc(cell.source))" />
      <div v-else-if="cell.cell_type === 'code'" class="nb-code-cell">
        <pre class="hljs"><code v-html="highlightCode(joinSrc(cell.source))"></code></pre>
        <div v-if="cell.outputs?.length" class="nb-outputs">
          <div v-for="(out, j) in cell.outputs" :key="j" v-html="renderOutput(out)"></div>
        </div>
      </div>
    </div>
  </article>
</template>

<style scoped>
.notebook {
  min-width: 0;
}
.nb-hint {
  color: var(--text-muted);
  font-size: 0.875rem;
  margin-bottom: 2rem;
}
.nb-cell {
  margin-bottom: 1.25rem;
}
.nb-code-cell {
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}
.nb-outputs {
  border-top: 1px solid var(--border);
  padding: 0.75rem 1rem;
  background: var(--surface);
}
.nb-img {
  max-width: 100%;
  border-radius: 6px;
}
.nb-error {
  color: #ef4444;
}
</style>
