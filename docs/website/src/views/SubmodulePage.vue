<script setup lang="ts">
import { computed, ref, watch, onMounted } from "vue";
import { useRoute } from "vue-router";
import { submoduleSites } from "../submodules";

const route = useRoute();
const loaded = ref(false);
const timedOut = ref(false);
let timer: number | undefined;

const site = computed(() => submoduleSites.find((s) => s.slug === route.params.slug));

function reset() {
  loaded.value = false;
  timedOut.value = false;
  window.clearTimeout(timer);
  timer = window.setTimeout(() => {
    if (!loaded.value) timedOut.value = true;
  }, 6000);
}

watch(() => route.params.slug, reset);
onMounted(reset);
</script>

<template>
  <div class="submodule-page">
    <div v-if="!site" class="doc-notfound">
      <p>Unknown submodule <code>{{ route.params.slug }}</code>.</p>
    </div>

    <template v-else>
      <div class="submodule-bar">
        <div>
          <strong>{{ site.title }}</strong>
          <span class="submodule-desc">{{ site.description }}</span>
        </div>
        <div class="submodule-links">
          <a :href="site.url" target="_blank" rel="noopener noreferrer">Open in new tab ↗</a>
          <a :href="site.repo" target="_blank" rel="noopener noreferrer">Source ↗</a>
        </div>
      </div>

      <div v-if="timedOut && !loaded" class="submodule-fallback">
        <p>
          This submodule's site hasn't loaded — it may not be deployed yet, or this
          browser/network is blocking the embed.
          <a :href="site.url" target="_blank" rel="noopener noreferrer">Open it directly ↗</a>
        </p>
      </div>

      <iframe
        :key="site.slug"
        class="submodule-frame"
        :src="site.url"
        :title="site.title + ' documentation'"
        @load="loaded = true"
      />
    </template>
  </div>
</template>

<style scoped>
.submodule-page {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 3.5rem);
}
.submodule-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.75rem 1.5rem;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  font-size: 0.875rem;
  flex-wrap: wrap;
}
.submodule-desc {
  color: var(--text-muted);
  margin-left: 0.6rem;
}
.submodule-links {
  display: flex;
  gap: 1rem;
}
.submodule-links a {
  color: var(--accent);
}
.submodule-fallback {
  padding: 0.75rem 1.5rem;
  background: var(--accent-soft);
  color: var(--text-muted);
  font-size: 0.8125rem;
}
.submodule-frame {
  flex: 1;
  border: none;
  width: 100%;
}
.doc-notfound {
  padding: 4rem 2rem;
  text-align: center;
  color: var(--text-muted);
}
</style>
