<template>
  <div class="case-list">
    <div v-if="loading" class="flex justify-center items-center py-8">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
    </div>

    <div v-else-if="error" class="bg-destructive/10 text-destructive p-4 rounded-lg">
      {{ error }}
    </div>

    <div v-else-if="cases.length === 0" class="text-center py-8 text-muted-foreground px-4">
      <i class="fas fa-folder-open text-3xl mb-2 text-gundam-blue"></i>
      <p>暂无案例，从分析结果中导出股票到案例库</p>
    </div>

    <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 px-4">
      <div v-for="caseItem in cases" :key="caseItem.id"
        class="bg-card border border-border rounded-lg overflow-hidden hover:shadow-md transition-shadow">
        <div class="p-4">
          <div class="flex justify-between items-start mb-2">
            <h3 class="text-lg font-medium truncate" :title="caseItem.title">{{ caseItem.title }}</h3>
            <div class="dropdown dropdown-end">
              <button class="btn btn-ghost btn-sm p-1 text-gundam-blue hover:text-gundam-blue/80">
                <i class="fas fa-ellipsis-v"></i>
              </button>
              <ul class="dropdown-content z-10 menu p-1 shadow bg-card rounded-md w-28 border border-border">
                <li>
                  <button @click="$emit('view-case', caseItem)" class="flex items-center text-gundam-blue text-xs py-1">
                    <i class="fas fa-eye w-3 h-3 mr-1"></i>
                    <span>查看</span>
                  </button>
                </li>
                <li>
                  <button @click="$emit('edit-case', caseItem)"
                    class="flex items-center text-gundam-yellow text-xs py-1">
                    <i class="fas fa-edit w-3 h-3 mr-1"></i>
                    <span>编辑</span>
                  </button>
                </li>
                <li>
                  <button @click="$emit('delete-case', caseItem)"
                    class="flex items-center text-gundam-red text-xs py-1">
                    <i class="fas fa-trash-alt w-3 h-3 mr-1"></i>
                    <span>删除</span>
                  </button>
                </li>
              </ul>
            </div>
          </div>

          <div class="flex items-center text-sm text-muted-foreground mb-3">
            <span class="truncate">{{ caseItem.stockName }}</span>
            <span class="mx-1">·</span>
            <span class="font-mono">{{ caseItem.stockCode }}</span>
          </div>

          <div class="flex flex-wrap gap-2 mb-3">
            <span v-for="tag in caseItem.tags" :key="tag" :class="[
              'px-2 py-0.5 text-xs rounded-full',
              tag.includes('底部') || tag.includes('低位') ? 'bg-gundam-blue/10 text-gundam-blue' :
                tag.includes('突破') ? 'bg-gundam-yellow/10 text-gundam-yellow' :
                  'bg-gundam-red/10 text-gundam-red'
            ]">
              {{ tag }}
            </span>
          </div>

          <div class="flex justify-between items-center text-xs text-muted-foreground">
            <span>创建于 {{ formatDate(caseItem.createdAt) }}</span>
            <button @click="$emit('view-case', caseItem)" class="text-gundam-blue hover:underline">
              查看详情
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>


const props = defineProps({
  cases: {
    type: Array,
    default: () => []
  },
  loading: {
    type: Boolean,
    default: false
  },
  error: {
    type: String,
    default: null
  }
});

defineEmits(['view-case', 'edit-case', 'delete-case']);

// 格式化日期
const formatDate = (dateString) => {
  const date = new Date(dateString);
  return date.toLocaleDateString();
};
</script>

<style scoped>
.dropdown {
  position: relative;
  display: inline-block;
}

.dropdown-content {
  display: none;
  position: absolute;
  right: 0;
  top: 25px;
  min-width: 110px;
  z-index: 10;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.dropdown:hover .dropdown-content,
.dropdown:focus-within .dropdown-content {
  display: block;
}

.dropdown-content li button {
  width: 100%;
  text-align: left;
  padding: 0.35rem 0.75rem;
  display: flex;
  align-items: center;
}

.dropdown-content li button:hover {
  background-color: var(--muted);
}
</style>
