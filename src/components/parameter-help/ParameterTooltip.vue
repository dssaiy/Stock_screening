<template>
  <div class="parameter-tooltip-container">
    <div ref="trigger" @mouseenter="showTooltip = true" @mouseleave="handleMouseLeave" class="tooltip-trigger">
      <slot>
        <!-- 默认内容，如果没有传递内容则显示 -->
      </slot>
      <span class="info-icon">
        <i class="fas fa-circle-info"></i>
      </span>
    </div>

    <Transition enter-active-class="transition duration-200 ease-out" enter-from-class="opacity-0 scale-95"
      enter-to-class="opacity-100 scale-100" leave-active-class="transition duration-150 ease-in"
      leave-from-class="opacity-100 scale-100" leave-to-class="opacity-0 scale-95">
      <div v-if="showTooltip" ref="tooltip" @mouseenter="isHoveringTooltip = true"
        @mouseleave="isHoveringTooltip = false" class="tooltip-content" :class="[position]">
        <div class="tooltip-inner">
          <div class="tooltip-title">{{ title }}</div>
          <div class="tooltip-description">{{ description }}</div>
          <button @click="$emit('show-tutorial', id)" class="tooltip-button">
            查看详情
          </button>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue';

const props = defineProps({
  id: {
    type: String,
    required: true
  },
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    required: true
  },
  position: {
    type: String,
    default: 'top',
    validator: (value) => ['top', 'bottom', 'left', 'right'].includes(value)
  }
});

defineEmits(['show-tutorial']);

const showTooltip = ref(false);
const isHoveringTooltip = ref(false);
const trigger = ref(null);
const tooltip = ref(null);

const handleMouseLeave = () => {
  setTimeout(() => {
    if (!isHoveringTooltip.value) {
      showTooltip.value = false;
    }
  }, 100);
};

watch(isHoveringTooltip, (newValue) => {
  if (!newValue) {
    setTimeout(() => {
      if (!isHoveringTooltip.value) {
        showTooltip.value = false;
      }
    }, 100);
  }
});

// 处理点击外部关闭
const handleClickOutside = (event) => {
  if (
    showTooltip.value &&
    tooltip.value &&
    !tooltip.value.contains(event.target) &&
    trigger.value &&
    !trigger.value.contains(event.target)
  ) {
    showTooltip.value = false;
  }
};

onMounted(() => {
  document.addEventListener('click', handleClickOutside);
});

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside);
});
</script>

<style scoped>
.parameter-tooltip-container {
  position: relative;
  display: inline-block;
}

.tooltip-trigger {
  display: inline-flex;
  align-items: center;
  cursor: help;
}

.info-icon {
  margin-left: 0.375rem;
  font-size: 1rem;
  color: hsl(var(--primary));
  opacity: 0.75;
  transition: all 0.2s;
  cursor: help;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.25rem;
  height: 1.25rem;
  border-radius: 50%;
  background-color: hsl(var(--primary) / 0.1);
  border: 1px solid hsl(var(--primary) / 0.2);
}

.info-icon:hover {
  opacity: 1;
  transform: scale(1.1);
  background-color: hsl(var(--primary) / 0.15);
  border-color: hsl(var(--primary) / 0.3);
  box-shadow: 0 0 0 2px hsl(var(--primary) / 0.1);
}

.tooltip-content {
  position: absolute;
  z-index: 50;
  width: 280px;
  filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
}

.tooltip-content.top {
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%) translateY(-8px);
}

.tooltip-content.bottom {
  top: 100%;
  left: 50%;
  transform: translateX(-50%) translateY(8px);
}

.tooltip-content.left {
  right: 100%;
  top: 50%;
  transform: translateY(-50%) translateX(-8px);
}

.tooltip-content.right {
  left: 100%;
  top: 50%;
  transform: translateY(-50%) translateX(8px);
}

.tooltip-content {
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}

.tooltip-inner {
  background-color: rgba(255, 255, 255, 0.15);
  border-left: 3px solid hsl(var(--primary));
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.dark .tooltip-inner {
  background-color: rgba(30, 41, 59, 0.75);
}

.tooltip-title {
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: hsl(var(--primary));
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.tooltip-description {
  margin-bottom: 0.75rem;
  line-height: 1.4;
  color: hsl(var(--foreground));
}

.tooltip-button {
  background-color: oklch(0.55 0.2 255);
  /* 高达蓝色 */
  color: white;
  font-size: 0.75rem;
  padding: 0.25rem 0.75rem;
  border-radius: 0.25rem;
  border: none;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.tooltip-button:hover {
  background-color: oklch(0.6 0.22 255);
  /* 高达蓝色（亮一点） */
  transform: translateY(-1px);
  box-shadow: 0 3px 6px rgba(0, 0, 0, 0.15);
}
</style>
