<template>
  <button 
    @click="toggleTheme" 
    class="theme-toggle relative w-12 h-6 rounded-full transition-colors duration-300 flex items-center"
    :class="{ 'bg-gray-700': isDarkMode, 'bg-gray-300': !isDarkMode }"
    aria-label="切换主题模式"
  >
    <span 
      class="toggle-thumb absolute w-5 h-5 rounded-full bg-white flex items-center justify-center transition-transform duration-300 shadow-md"
      :class="{ 'translate-x-6': isDarkMode, 'translate-x-1': !isDarkMode }"
    >
      <i 
        :class="['text-xs', isDarkMode ? 'fas fa-moon text-gray-700' : 'fas fa-sun text-yellow-500']"
        aria-hidden="true"
      ></i>
    </span>
  </button>
</template>

<script setup>
import { inject } from 'vue';

// 从父组件注入isDarkMode和toggleDarkMode
const isDarkMode = inject('isDarkMode');
const toggleDarkMode = inject('toggleDarkMode');

// 切换主题
const toggleTheme = () => {
  toggleDarkMode();
};
</script>

<style scoped>
.theme-toggle {
  cursor: pointer;
  outline: none;
  border: none;
}

.theme-toggle:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

.toggle-thumb {
  top: 2px;
}
</style>
