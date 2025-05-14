<template>
  <div class="case-manager">
    <!-- 案例管理界面 -->
    <div v-if="!selectedCase" class="case-list-container">
      <div class="header flex justify-between items-center mb-4 p-4 border-b border-border bg-gundam-blue/10">
        <h2 class="text-xl font-semibold flex items-center">
          <i class="fas fa-book mr-2 text-gundam-blue"></i>
          案例管理
        </h2>
        <button @click="$emit('close')"
          class="px-3 py-1 rounded-md text-gundam-blue hover:bg-gundam-blue/10 transition-colors">
          关闭
        </button>
      </div>

      <!-- 搜索框 -->
      <div class="flex justify-between items-center mb-4 px-4 pt-2">
        <div class="search-box relative flex-1">
          <input v-model="searchQuery" type="text" placeholder="搜索案例..." class="input pl-8 w-full" />
          <i class="fas fa-search absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground"></i>
        </div>
      </div>

      <!-- 案例列表组件 -->
      <CaseList :cases="filteredCases" :loading="loading" :error="error" @view-case="openCase" @edit-case="editCase"
        @delete-case="confirmDeleteCase" />



      <!-- 删除确认对话框 -->
      <div v-if="showDeleteConfirm" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div class="bg-card rounded-lg shadow-lg w-full max-w-md p-6" @click.stop>
          <h3 class="text-lg font-semibold mb-2">确认删除</h3>
          <p class="text-muted-foreground mb-4">确定要删除这个案例吗？此操作无法撤销。</p>
          <div class="flex justify-end space-x-3">
            <button class="btn btn-ghost" @click="showDeleteConfirm = false">
              取消
            </button>
            <button class="btn btn-destructive" @click="deleteCase">
              删除
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- 案例详情 -->
    <CaseDetail v-else :case-id="selectedCase" @close="selectedCase = null" @delete="confirmDeleteCase" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue';
import axios from 'axios';
import CaseDetail from './CaseDetail.vue';
import CaseList from './CaseList.vue';

// 属性和事件
const props = defineProps({
  stockData: {
    type: Object,
    default: null
  }
});

const emit = defineEmits(['close', 'case-created', 'case-deleted']);

// 状态
const cases = ref([]);
const selectedCase = ref(null);
const searchQuery = ref('');
const showDeleteConfirm = ref(false);
const caseToDelete = ref(null);
const loading = ref(false);
const error = ref(null);

// 计算属性
const filteredCases = computed(() => {
  if (!searchQuery.value) return cases.value;

  const query = searchQuery.value.toLowerCase();
  return cases.value.filter(caseItem =>
    caseItem.title.toLowerCase().includes(query) ||
    caseItem.stockCode.toLowerCase().includes(query) ||
    caseItem.stockName.toLowerCase().includes(query) ||
    (caseItem.tags && caseItem.tags.some(tag => tag.toLowerCase().includes(query)))
  );
});

// 方法
const loadCases = async () => {
  loading.value = true;
  error.value = null;

  try {
    // 从API加载案例
    const response = await axios.get('/api/cases');
    cases.value = response.data.cases || [];
    loading.value = false;
  } catch (err) {
    console.error('加载案例失败:', err);
    error.value = '加载案例失败，请稍后重试';
    loading.value = false;

    // 如果API调用失败，尝试从本地文件加载
    try {
      const response = await fetch('/cases/index.json');
      const data = await response.json();
      cases.value = data.cases || [];
    } catch (e) {
      console.error('从本地文件加载案例失败:', e);

      // 使用模拟数据作为最后的后备方案
      if (process.env.NODE_ENV === 'development') {
        cases.value = [
          {
            id: 'case_1745645965',
            title: '安记食品底部横盘案例',
            stockCode: 'sh.603696',
            stockName: '安记食品',
            createdAt: '2024-04-26T10:00:00Z',
            updatedAt: '2024-04-26T10:00:00Z',
            tags: ['底部横盘', '突破确认', '低位']
          }
        ];
      }
    }
  }
};



const openCase = (caseItem) => {
  selectedCase.value = caseItem.id;
};

const editCase = (caseItem) => {
  // 打开案例详情并进入编辑模式
  selectedCase.value = caseItem.id;
};

const confirmDeleteCase = (caseItem) => {
  caseToDelete.value = caseItem.id;
  showDeleteConfirm.value = true;
};

const deleteCase = async () => {
  try {
    // 调用API删除案例
    await axios.delete(`/api/cases/${caseToDelete.value}`);

    // 从本地列表中移除
    cases.value = cases.value.filter(caseItem => caseItem.id !== caseToDelete.value);

    showDeleteConfirm.value = false;

    // 如果正在查看被删除的案例，则返回列表
    if (selectedCase.value === caseToDelete.value) {
      selectedCase.value = null;
    }

    emit('case-deleted', caseToDelete.value);
  } catch (error) {
    console.error('删除案例失败:', error);
    alert('删除案例失败，请稍后重试');
  }
};

const formatDate = (dateString) => {
  const date = new Date(dateString);
  return date.toLocaleDateString();
};

// 生命周期钩子
onMounted(() => {
  loadCases();
});
</script>

<style scoped>
.case-manager {
  height: 100%;
  overflow-y: auto;
}

.case-list-container {
  padding: 0;
}

.empty-state {
  border: 2px dashed var(--border);
  border-radius: 0.5rem;
}
</style>
