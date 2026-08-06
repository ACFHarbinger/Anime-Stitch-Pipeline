<script setup lang="ts">
import { navTree } from "../nav.generated";
import { submoduleSites } from "../submodules";
import SidebarSection from "./SidebarSection.vue";

defineProps<{ open: boolean }>();
const emit = defineEmits<{ close: [] }>();
</script>

<template>
  <div v-if="open" class="sidebar-scrim" @click="emit('close')" />
  <nav class="sidebar" :class="{ open }">
    <SidebarSection v-for="(node, i) in navTree" :key="i" :node="node" :depth="0" />

    <template v-if="submoduleSites.length">
      <div class="related-heading">Related Projects</div>
      <router-link
        v-for="s in submoduleSites"
        :key="s.slug"
        class="related-link"
        :to="'/submodules/' + s.slug"
      >
        {{ s.title }}
      </router-link>
    </template>
  </nav>
</template>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
  padding: 1.25rem 0.75rem 3rem;
  overflow-y: auto;
}

.related-heading {
  margin-top: 1.25rem;
  padding: 0.4rem 0.75rem;
  font-weight: 600;
  font-size: 0.8125rem;
  color: var(--text);
  border-top: 1px solid var(--border);
  padding-top: 1.25rem;
}
.related-link {
  padding: 0.375rem 0.75rem;
  font-size: 0.8125rem;
  color: var(--text-muted);
  border-radius: 6px;
}
.related-link:hover {
  color: var(--text);
  background: var(--surface-hover);
}
.related-link.router-link-active {
  color: var(--accent);
  background: var(--accent-soft);
  font-weight: 600;
}

.sidebar-scrim {
  display: none;
}

@media (max-width: 980px) {
  .sidebar {
    position: fixed;
    top: 3.5rem;
    left: 0;
    bottom: 0;
    width: 280px;
    background: var(--bg);
    border-right: 1px solid var(--border);
    transform: translateX(-100%);
    transition: transform 0.2s ease;
    z-index: 40;
  }
  .sidebar.open {
    transform: translateX(0);
  }
  .sidebar-scrim {
    display: block;
    position: fixed;
    inset: 3.5rem 0 0 0;
    background: rgba(0, 0, 0, 0.4);
    z-index: 30;
  }
}
</style>
