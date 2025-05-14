<template>
  <div class="parameter-label">
    <div class="flex items-center justify-between w-full">
      <label :for="forId" class="block text-sm font-medium mb-1">
        <slot></slot>
      </label>
      <ParameterTooltip v-if="tooltip" :id="tooltip.id" :title="tooltip.title" :description="tooltip.description"
        :position="position" @show-tutorial="handleShowTutorial">
        <!-- 这里不需要传递内容，ParameterTooltip 组件内部已经有图标 -->
      </ParameterTooltip>
    </div>
  </div>
</template>

<script setup>
import { inject } from 'vue';
import ParameterTooltip from './ParameterTooltip.vue';

const props = defineProps({
  forId: {
    type: String,
    required: true
  },
  parameterId: {
    type: String,
    required: true
  },
  position: {
    type: String,
    default: 'top'
  }
});

const emit = defineEmits(['show-tutorial']);

// 获取参数提示
console.log('ParameterLabel: 参数ID', props.parameterId);

// 直接从全局注入参数数据
const parameterTooltips = inject('parameterTooltips', {});
const tooltip = parameterTooltips[props.parameterId] || null;

// 获取全局 parameterHelp 对象（用于打开教程）
const parameterHelp = inject('parameterHelp');

console.log('ParameterLabel: 直接获取的提示', tooltip);
console.log('ParameterLabel: 注入的 parameterHelp', parameterHelp);

// 处理显示教程事件
const handleShowTutorial = (id) => {
  console.log('ParameterLabel: 处理显示教程事件', id);
  if (parameterHelp && parameterHelp.openTutorial) {
    parameterHelp.openTutorial(id);
  } else {
    console.warn('ParameterLabel: parameterHelp.openTutorial 未提供');
    // 仍然触发事件，以防父组件有处理
    emit('show-tutorial', id);
  }
};
</script>

<style scoped>
.parameter-label {
  display: block;
  width: 100%;
}

.parameter-label label {
  margin-right: 0.5rem;
}
</style>
