<template>
  <div class="task-progress">
    <!-- Loading state with progress bar -->
    <div v-if="status === 'running' || status === 'pending'" class="card p-6 text-center my-5">
      <div class="flex flex-col items-center">
        <i class="fas fa-circle-notch fa-spin text-2xl mb-2 text-primary"></i>
        <p class="text-lg font-medium mb-2">{{ message }}</p>

        <!-- Progress bar -->
        <div class="w-full bg-muted rounded-full h-4 mb-2 max-w-md">
          <div class="bg-primary h-4 rounded-full transition-all duration-300 ease-out"
            :style="{ width: `${progress}%` }"></div>
        </div>
        <p class="text-sm text-muted-foreground">{{ progress }}% 完成</p>
      </div>
    </div>

    <!-- Error state -->
    <div v-else-if="status === 'failed'" class="card p-6 text-center my-5 border-destructive">
      <div class="flex flex-col items-center">
        <i class="fas fa-exclamation-triangle text-2xl mb-2 text-destructive"></i>
        <p class="text-lg font-medium mb-2 text-destructive">处理失败</p>
        <p class="text-sm text-muted-foreground mb-4">{{ message }}</p>
        <div v-if="error" class="bg-muted p-4 rounded-md text-left w-full max-w-2xl overflow-auto max-h-40">
          <pre class="text-xs">{{ error }}</pre>
        </div>
        <button @click="$emit('retry')" class="btn btn-primary mt-4">
          <i class="fas fa-redo mr-2"></i>重试
        </button>
      </div>
    </div>

    <!-- Completed state is handled by parent component -->
  </div>
</template>

<script setup>


const props = defineProps({
  status: {
    type: String,
    required: true,
    validator: (value) => ['pending', 'running', 'completed', 'failed'].includes(value)
  },
  progress: {
    type: Number,
    default: 0
  },
  message: {
    type: String,
    default: '处理中...'
  },
  error: {
    type: String,
    default: null
  }
});

defineEmits(['retry']);
</script>

<style scoped>
.task-progress {
  transition: all 0.3s ease;
}
</style>
