<template>
  <div>
    <!-- 参数提示组件将被动态插入到应用中 -->
    <ParameterTutorial v-if="activeTutorial" :show="showTutorial" :tutorial="activeTutorial" @close="closeTutorial" />
  </div>
</template>

<script setup>
import { ref, inject, onMounted } from 'vue';
import ParameterTutorial from './ParameterTutorial.vue';

// 从全局注入参数数据
const parameterTooltips = inject('parameterTooltips', {});
const parameterTutorials = inject('parameterTutorials', {});

console.log('ParameterHelpManager: 注入的提示数据', Object.keys(parameterTooltips).length);
console.log('ParameterHelpManager: 注入的教程数据', Object.keys(parameterTutorials).length);

// 当前激活的教程
const activeTutorial = ref(null);
const showTutorial = ref(false);

// 打开教程
const openTutorial = (id) => {
  console.log('ParameterHelpManager: 打开教程', id);
  if (parameterTutorials[id]) {
    activeTutorial.value = parameterTutorials[id];
    showTutorial.value = true;
    console.log('教程已设置:', parameterTutorials[id].title);
  } else {
    console.warn('未找到教程:', id);
  }
};

// 关闭教程
const closeTutorial = () => {
  console.log('ParameterHelpManager: 关闭教程');
  showTutorial.value = false;
};

// 获取参数提示
const getTooltip = (id) => {
  console.log('ParameterHelpManager: 获取提示', id);
  const tooltip = parameterTooltips[id] || null;
  console.log('提示内容:', tooltip);
  return tooltip;
};

// 获取全局提供的 parameterHelp 对象
const globalParameterHelp = inject('parameterHelp');

// 在组件挂载后，更新全局 parameterHelp 对象的方法
onMounted(() => {
  if (globalParameterHelp) {
    console.log('ParameterHelpManager: 更新全局 parameterHelp 对象的方法');
    globalParameterHelp.openTutorial = openTutorial;
    globalParameterHelp.closeTutorial = closeTutorial;
    globalParameterHelp.getTooltip = getTooltip;
  } else {
    console.error('ParameterHelpManager: 无法获取全局 parameterHelp 对象');
  }
});
</script>
