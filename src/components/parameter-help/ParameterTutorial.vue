<template>
  <Transition enter-active-class="transition duration-300 ease-out" enter-from-class="opacity-0 scale-95"
    enter-to-class="opacity-100 scale-100" leave-active-class="transition duration-200 ease-in"
    leave-from-class="opacity-100 scale-100" leave-to-class="opacity-0 scale-95">
    <div v-if="show" class="tutorial-overlay" @click="handleOverlayClick">
      <div class="tutorial-container" ref="tutorialRef">
        <div class="corner-tl"></div>
        <div class="corner-br"></div>
        <div class="tutorial-header">
          <h3 class="tutorial-title">{{ tutorial.title }}</h3>
          <button @click="$emit('close')" class="tutorial-close-button">
            关闭
          </button>
        </div>

        <div class="tutorial-content">
          <div v-if="tutorial.sections && tutorial.sections.length > 0">
            <div v-for="(section, index) in tutorial.sections" :key="index" class="tutorial-section">
              <h4 class="section-title">{{ section.title }}</h4>
              <div class="section-content" v-html="section.content"></div>

              <div v-if="section.examples && section.examples.length > 0" class="section-examples">
                <div v-for="(example, exIndex) in section.examples" :key="exIndex" class="example">
                  <div class="example-scenario">{{ example.scenario }}</div>
                  <div class="example-recommendation">
                    推荐值: <span class="example-value">{{ example.value }}</span>
                  </div>
                  <div v-if="example.explanation" class="example-explanation">
                    {{ example.explanation }}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div v-else class="tutorial-simple-content">
            {{ tutorial.content }}
          </div>

          <div v-if="tutorial.tips && tutorial.tips.length > 0" class="tutorial-tips">
            <h4 class="tips-title">使用技巧</h4>
            <ul class="tips-list">
              <li v-for="(tip, index) in tutorial.tips" :key="index" class="tip-item">
                {{ tip }}
              </li>
            </ul>
          </div>
        </div>


      </div>
    </div>
  </Transition>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue';

const props = defineProps({
  show: {
    type: Boolean,
    default: false
  },
  tutorial: {
    type: Object,
    required: true
  }
});

defineEmits(['close']);

const tutorialRef = ref(null);

const handleOverlayClick = (event) => {
  // 只有点击遮罩层而不是内容区域时才关闭
  if (tutorialRef.value && !tutorialRef.value.contains(event.target)) {
    event.stopPropagation();
    event.preventDefault();
  }
};

// 处理ESC键关闭
const handleKeyDown = (event) => {
  if (event.key === 'Escape' && props.show) {
    event.preventDefault();
    event.stopPropagation();
  }
};

// 禁止滚动背景
watch(() => props.show, (newValue) => {
  if (newValue) {
    document.body.style.overflow = 'hidden';
  } else {
    document.body.style.overflow = '';
  }
});

onMounted(() => {
  document.addEventListener('keydown', handleKeyDown);
});

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeyDown);
  document.body.style.overflow = '';
});
</script>

<style scoped>
.tutorial-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
  padding: 1rem;
}

/* 亮色模式下的样式 */
.tutorial-container {
  background-color: white;
  border-radius: 0.75rem;
  width: 100%;
  max-width: 650px;
  max-height: 85vh;
  overflow-y: auto;
  box-shadow:
    0 10px 25px -5px rgba(0, 0, 0, 0.2),
    0 0 15px rgba(var(--primary-rgb), 0.15);
  border: 1px solid rgba(0, 0, 0, 0.08);
  display: flex;
  flex-direction: column;
  position: relative;
  color: hsl(var(--foreground));
}

/* 暗色模式下的样式 */
.dark .tutorial-container {
  background-color: #1e293b;
  /* 固定的深色背景色 */
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow:
    0 10px 25px -5px rgba(0, 0, 0, 0.4),
    0 0 15px rgba(var(--primary-rgb), 0.25);
}

/* 高达主题渐变边框效果 - 亮色模式 */
.tutorial-container::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  border-radius: 0.75rem;
  padding: 2px;
  background: linear-gradient(135deg,
      hsl(var(--primary) / 0.6),
      transparent 40%,
      transparent 60%,
      hsl(var(--primary) / 0.6));
  -webkit-mask:
    linear-gradient(#fff 0 0) content-box,
    linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events: none;
  z-index: 1;
}

/* 高达主题渐变边框效果 - 暗色模式 */
.dark .tutorial-container::before {
  background: linear-gradient(135deg,
      hsl(var(--primary) / 0.7),
      transparent 40%,
      transparent 60%,
      hsl(var(--primary) / 0.7));
}

.tutorial-header {
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  top: 0;
  background-color: oklch(0.55 0.2 255);
  /* 纯高达蓝色背景 */
  z-index: 10;
  border-top-left-radius: 0.75rem;
  border-top-right-radius: 0.75rem;
}

/* 暗色模式下的标题栏 */
.dark .tutorial-header {
  background-color: oklch(0.3 0.15 255);
  /* 深色高达蓝色背景 */
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.tutorial-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: white;
  /* 白色文字在蓝色背景上 */
  margin: 0;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
  position: relative;
  padding-left: 1.5rem;
}

.tutorial-title::before {
  content: "";
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 0.75rem;
  height: 0.75rem;
  background-color: hsl(var(--primary));
  clip-path: polygon(0 0, 100% 0, 100% 100%);
}

.tutorial-close-button {
  background-color: rgba(255, 255, 255, 0.2);
  color: white;
  cursor: pointer;
  padding: 0.5rem 1.25rem;
  border-radius: 0.375rem;
  border: 1px solid rgba(255, 255, 255, 0.3);
  font-size: 0.875rem;
  font-weight: 500;
  transition: all 0.2s;
  position: relative;
}

.tutorial-close-button:hover {
  background-color: rgba(255, 255, 255, 0.3);
  transform: translateY(-1px);
}

.tutorial-content {
  padding: 1.75rem;
  padding-bottom: 2.5rem;
  /* 增加底部内边距 */
  flex: 1;
  overflow-y: auto;
  color: hsl(var(--foreground));
  line-height: 1.6;
  border-bottom-left-radius: 0.75rem;
  border-bottom-right-radius: 0.75rem;
}

/* 确保深色模式下文本颜色有足够对比度 */
.dark .tutorial-content {
  color: rgba(255, 255, 255, 0.9);
}

.tutorial-section {
  margin-bottom: 2rem;
  position: relative;
}

.tutorial-section:last-child {
  margin-bottom: 0;
}

.section-title {
  font-size: 1.125rem;
  font-weight: 600;
  margin-top: 0;
  margin-bottom: 1rem;
  color: hsl(var(--primary));
  display: flex;
  align-items: center;
  position: relative;
}

.section-title::before {
  content: "";
  display: inline-block;
  width: 0.5rem;
  height: 1rem;
  background-color: hsl(var(--primary));
  margin-right: 0.75rem;
  clip-path: polygon(0 0, 100% 50%, 0 100%);
}

.section-content {
  margin-bottom: 1.25rem;
  line-height: 1.7;
  color: hsl(var(--foreground));
  padding-left: 1.25rem;
  border-left: 1px dashed hsl(var(--primary) / 0.3);
}

.dark .section-content {
  color: rgba(255, 255, 255, 0.9);
}

.section-examples {
  background-color: hsl(var(--muted) / 0.2);
  border-radius: 0.75rem;
  padding: 1.25rem;
  margin-top: 1rem;
  margin-left: 1.25rem;
  border: 1px solid hsl(var(--border) / 0.5);
  position: relative;
  overflow: hidden;
}

.dark .section-examples {
  background-color: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.section-examples::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 4px;
  height: 100%;
  background: linear-gradient(to bottom, hsl(var(--primary)), transparent);
}

.example {
  margin-bottom: 1.25rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid hsl(var(--border) / 0.3);
  position: relative;
}

.example:last-child {
  margin-bottom: 0;
  padding-bottom: 0;
  border-bottom: none;
}

.example-scenario {
  font-weight: 600;
  margin-bottom: 0.75rem;
  color: hsl(var(--foreground));
  display: flex;
  align-items: center;
}

.example-scenario::before {
  content: "●";
  color: hsl(var(--primary));
  margin-right: 0.5rem;
  font-size: 0.75rem;
}

.example-recommendation {
  margin-bottom: 0.75rem;
  color: hsl(var(--foreground));
  padding-left: 1rem;
}

.example-value {
  font-weight: 600;
  color: hsl(var(--primary));
  background-color: hsl(var(--primary) / 0.1);
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
  display: inline-block;
  margin-left: 0.25rem;
  border: 1px solid hsl(var(--primary) / 0.2);
}

.example-explanation {
  font-size: 0.875rem;
  color: hsl(var(--muted-foreground));
  padding-left: 1rem;
  line-height: 1.5;
}

.tutorial-simple-content {
  line-height: 1.7;
  color: hsl(var(--foreground));
  padding: 0 1rem;
}

.tutorial-tips {
  margin-top: 2rem;
  background-color: hsl(var(--primary) / 0.08);
  border-radius: 0.75rem;
  padding: 1.25rem;
  position: relative;
  overflow: hidden;
  border: 1px solid hsl(var(--primary) / 0.15);
}

.tutorial-tips::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 4px;
  background: linear-gradient(to right, hsl(var(--primary)), transparent);
}

.tips-title {
  font-size: 1.05rem;
  font-weight: 600;
  margin-top: 0;
  margin-bottom: 1rem;
  color: hsl(var(--primary));
  display: flex;
  align-items: center;
}

.tips-title::before {
  content: "✦";
  margin-right: 0.5rem;
  color: hsl(var(--primary));
}

.tips-list {
  margin: 0;
  padding-left: 1.75rem;
  color: hsl(var(--foreground));
}

.tip-item {
  margin-bottom: 0.75rem;
  position: relative;
}

.tip-item::marker {
  color: hsl(var(--primary));
}

.tip-item:last-child {
  margin-bottom: 0;
}



/* 高达风格装饰元素 */
.tutorial-container::after {
  content: "";
  position: absolute;
  top: 1rem;
  right: 1rem;
  width: 1.5rem;
  height: 1.5rem;
  background-color: transparent;
  border: 2px solid hsl(var(--primary) / 0.4);
  border-radius: 0.25rem;
  clip-path: polygon(0 0, 100% 0, 100% 70%, 70% 100%, 0 100%);
  pointer-events: none;
  opacity: 0.7;
}

/* 左上角装饰 */
.tutorial-container .corner-tl {
  position: absolute;
  top: 0.5rem;
  left: 0.5rem;
  width: 0.75rem;
  height: 0.75rem;
  border-top: 2px solid hsl(var(--primary) / 0.5);
  border-left: 2px solid hsl(var(--primary) / 0.5);
  pointer-events: none;
}

/* 右下角装饰 */
.tutorial-container .corner-br {
  position: absolute;
  bottom: 0.5rem;
  right: 0.5rem;
  width: 0.75rem;
  height: 0.75rem;
  border-bottom: 2px solid hsl(var(--primary) / 0.5);
  border-right: 2px solid hsl(var(--primary) / 0.5);
  pointer-events: none;
}



/* 暗色模式特定样式 */
.dark .tutorial-content {
  color: hsl(var(--foreground) / 0.95);
}

.dark .section-content {
  color: hsl(var(--foreground) / 0.9);
}

.dark .example-scenario {
  color: hsl(var(--foreground) / 0.95);
}

.dark .example-recommendation {
  color: hsl(var(--foreground) / 0.9);
}

.dark .example-value {
  background-color: hsl(var(--primary) / 0.15);
  border-color: hsl(var(--primary) / 0.3);
}

.dark .example-explanation {
  color: hsl(var(--muted-foreground) / 0.9);
}

.dark .tips-list {
  color: hsl(var(--foreground) / 0.9);
}
</style>
