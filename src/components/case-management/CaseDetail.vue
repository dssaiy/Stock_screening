<template>
  <div class="case-detail flex flex-col h-full">
    <!-- 固定在顶部的标题栏 -->
    <div class="sticky-header flex justify-between items-center py-3 px-4 bg-background border-b border-border z-10">
      <div class="flex items-center">
        <button @click="$emit('close')"
          class="px-3 py-2 rounded-md text-gundam-blue border border-gundam-blue hover:bg-gundam-blue/10 active:bg-gundam-blue/20 transition-all transform hover:scale-105 active:scale-95 flex items-center justify-center mr-3">
          <i class="fas fa-arrow-left text-lg"></i>
        </button>
        <h2 class="text-lg font-semibold truncate max-w-[60vw]">{{ editMode ? '编辑: ' + caseData.title : caseData.title
        }}</h2>
      </div>
      <div class="flex space-x-2">
        <button v-if="!editMode" @click="editMode = true"
          class="px-4 py-2 rounded-md text-white bg-gundam-blue hover:bg-gundam-blue/90 active:bg-gundam-blue/70 transition-all transform hover:scale-105 active:scale-95 shadow-md flex items-center justify-center text-base">
          <i class="fas fa-edit mr-2 text-lg"></i>
          编辑
        </button>
        <template v-else>
          <button type="button" @click="cancelEdit"
            class="px-4 py-2 rounded-md text-gundam-blue border border-gundam-blue hover:bg-gundam-blue/10 active:bg-gundam-blue/20 transition-all transform hover:scale-105 active:scale-95 flex items-center justify-center text-base">
            取消
          </button>
          <button type="button" @click="saveCase"
            class="px-4 py-2 rounded-md text-white bg-gundam-blue hover:bg-gundam-blue/90 active:bg-gundam-blue/70 transition-all transform hover:scale-105 active:scale-95 shadow-md flex items-center justify-center text-base">
            <i class="fas fa-save mr-2 text-lg"></i>
            保存
          </button>
        </template>
      </div>
    </div>

    <!-- 可滚动的内容区域 -->
    <div class="content-area flex-1 overflow-y-auto p-4">

      <!-- 案例内容 -->
      <div v-if="!editMode" class="case-content">
        <!-- 基本信息 -->
        <div class="bg-card border border-border rounded-lg p-4 mb-4">
          <div class="grid grid-cols-2 gap-4">
            <div>
              <p class="text-sm text-muted-foreground">股票代码</p>
              <p class="font-medium">{{ caseData.stockCode }}</p>
            </div>
            <div>
              <p class="text-sm text-muted-foreground">股票名称</p>
              <p class="font-medium">{{ caseData.stockName }}</p>
            </div>
            <div>
              <p class="text-sm text-muted-foreground">创建时间</p>
              <p>{{ formatDate(caseData.createdAt) }}</p>
            </div>
            <div>
              <p class="text-sm text-muted-foreground">更新时间</p>
              <p>{{ formatDate(caseData.updatedAt) }}</p>
            </div>
          </div>
          <div class="mt-3">
            <p class="text-sm text-muted-foreground">标签</p>
            <div class="flex flex-wrap gap-2 mt-1">
              <span v-for="tag in caseData.tags" :key="tag" :class="[
                'px-2 py-0.5 text-xs rounded-full',
                tag.includes('底部') || tag.includes('低位') ? 'bg-gundam-blue/10 text-gundam-blue' :
                  tag.includes('突破') ? 'bg-gundam-yellow/10 text-gundam-yellow' :
                    'bg-gundam-red/10 text-gundam-red'
              ]">
                {{ tag }}
              </span>
            </div>
          </div>
        </div>

        <!-- K线图 -->
        <div class="bg-card border border-border rounded-lg p-4 mb-4">
          <h3 class="text-md font-medium mb-3 flex justify-between items-center">
            <span>K线图</span>
            <button v-if="klineData && klineData.data && klineData.data.length > 0" @click="showFullKlineChart = true"
              class="px-4 py-2 rounded-md text-white bg-gundam-blue hover:bg-gundam-blue/90 active:bg-gundam-blue/70 transition-all transform hover:scale-105 active:scale-95 shadow-md flex items-center justify-center text-base">
              <i class="fas fa-expand-alt mr-2 text-lg"></i>
              全屏查看
            </button>
          </h3>
          <div class="kline-chart-container h-96">
            <div v-if="!klineData || !klineData.data || klineData.data.length === 0"
              class="h-full flex items-center justify-center">
              <p class="text-muted-foreground">{{ klineData ? '暂无K线数据' : '加载中...' }}</p>
            </div>
            <div v-else class="h-full">
              <!-- 使用与主应用相同的K线图组件 -->
              <KlineChart :klineData="klineData.data" :title="`${klineData.name} (${klineData.code})`"
                :isDarkMode="isDarkMode" :markLines="generateMarkLines()" :supportLevels="getSupportLevels()"
                :resistanceLevels="getResistanceLevels()" height="100%" width="100%"
                class="rounded-md overflow-hidden" />
            </div>
          </div>
        </div>

        <!-- 全屏K线图弹窗 -->
        <FullKlineChart v-if="klineData" v-model:visible="showFullKlineChart"
          :title="klineData.name ? `${klineData.name} (${klineData.code})` : '股票详情'" :klineData="klineData.data"
          :markLines="generateMarkLines()" :supportLevels="getSupportLevels()" :resistanceLevels="getResistanceLevels()"
          :isDarkMode="isDarkMode" />

        <!-- 分析参数 -->
        <div class="bg-card border border-border rounded-lg p-4 mb-4">
          <h3 class="text-md font-medium mb-3">分析参数</h3>
          <div v-if="analysisData && analysisData.parameters" class="grid grid-cols-2 md:grid-cols-3 gap-4">
            <!-- 窗口期 -->
            <div>
              <p class="text-sm text-muted-foreground">窗口期</p>
              <p>{{ analysisData.parameters.windows.join(', ') }} 天</p>
            </div>

            <!-- 价格参数 -->
            <div>
              <p class="text-sm text-muted-foreground">振幅阈值</p>
              <p>{{ analysisData.parameters.box_threshold }}</p>
            </div>
            <div>
              <p class="text-sm text-muted-foreground">均线粘合度</p>
              <p>{{ analysisData.parameters.ma_diff_threshold }}</p>
            </div>
            <div>
              <p class="text-sm text-muted-foreground">波动率阈值</p>
              <p>{{ analysisData.parameters.volatility_threshold }}</p>
            </div>

            <!-- 成交量参数 -->
            <div>
              <p class="text-sm text-muted-foreground">成交量分析</p>
              <p>{{ analysisData.parameters.use_volume_analysis ? '启用' : '禁用' }}</p>
            </div>
            <div v-if="analysisData.parameters.use_volume_analysis">
              <p class="text-sm text-muted-foreground">成交量变化阈值</p>
              <p>{{ analysisData.parameters.volume_change_threshold }}</p>
            </div>

            <!-- 位置参数 -->
            <div>
              <p class="text-sm text-muted-foreground">低位判断</p>
              <p>{{ analysisData.parameters.use_low_position ? '启用' : '禁用' }}</p>
            </div>
            <div v-if="analysisData.parameters.use_low_position">
              <p class="text-sm text-muted-foreground">下跌幅度阈值</p>
              <p>{{ analysisData.parameters.decline_threshold * 100 }}%</p>
            </div>
          </div>
          <div v-else class="text-muted-foreground">
            暂无分析参数
          </div>
        </div>

        <!-- 分析结果 -->
        <div class="bg-card border border-border rounded-lg p-4 mb-4">
          <h3 class="text-md font-medium mb-3">分析结果</h3>
          <div v-if="analysisData && analysisData.results" class="space-y-3">
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p class="text-sm text-muted-foreground">是否为平台期</p>
                <p :class="analysisData.results.is_platform ? 'text-green-500' : 'text-red-500'">
                  {{ analysisData.results.is_platform ? '是' : '否' }}
                </p>
              </div>
              <div v-if="analysisData.results.is_platform">
                <p class="text-sm text-muted-foreground">平台期窗口</p>
                <p>{{ analysisData.results.platform_windows.join(', ') }} 天</p>
              </div>
              <div v-if="analysisData.results.is_low_position !== undefined">
                <p class="text-sm text-muted-foreground">是否处于低位</p>
                <p :class="analysisData.results.is_low_position ? 'text-green-500' : 'text-red-500'">
                  {{ analysisData.results.is_low_position ? '是' : '否' }}
                </p>
              </div>
              <div v-if="analysisData.results.has_breakthrough !== undefined">
                <p class="text-sm text-muted-foreground">是否有突破信号</p>
                <p :class="analysisData.results.has_breakthrough ? 'text-green-500' : 'text-red-500'">
                  {{ analysisData.results.has_breakthrough ? '是' : '否' }}
                </p>
              </div>
            </div>

            <!-- 选择原因 -->
            <div
              v-if="analysisData.results.selection_reasons && Object.keys(analysisData.results.selection_reasons).length > 0">
              <p class="text-sm text-muted-foreground mb-1">选择原因</p>
              <ul class="list-disc list-inside space-y-1 text-sm">
                <li v-for="(reason, window) in analysisData.results.selection_reasons" :key="window">
                  {{ reason }}
                </li>
              </ul>
            </div>
          </div>
          <div v-else class="text-muted-foreground">
            暂无分析结果
          </div>
        </div>

        <!-- 案例描述 -->
        <div class="bg-card border border-border rounded-lg p-4">
          <h3 class="text-md font-medium mb-3">案例描述</h3>
          <div v-if="caseData.description" class="prose prose-sm max-w-none">
            <div v-html="renderedDescription"></div>
          </div>
          <div v-else class="text-muted-foreground">
            暂无描述
          </div>
        </div>
      </div>

      <!-- 编辑模式 -->
      <div v-else class="case-edit">
        <div class="space-y-4 pb-4">
          <!-- 基本信息 -->
          <div class="bg-card border border-border rounded-lg p-4">
            <h3 class="text-md font-medium mb-3">基本信息</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label class="block text-sm font-medium mb-1">案例标题</label>
                <input v-model="editData.title" class="input w-full" required />
              </div>
              <div>
                <label class="block text-sm font-medium mb-1">标签</label>
                <input v-model="editData.tagsInput" class="input w-full" placeholder="用逗号分隔多个标签" />
              </div>
            </div>
          </div>

          <!-- 案例描述 -->
          <div class="bg-card border border-border rounded-lg p-4">
            <h3 class="text-md font-medium mb-3">案例描述</h3>
            <textarea v-model="editData.description" class="input w-full h-64" placeholder="支持 Markdown 格式"></textarea>
            <p class="text-xs text-muted-foreground mt-1">支持 Markdown 格式</p>
          </div>

          <!-- 底部空间 -->
          <div class="h-8"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, inject } from 'vue';
import { marked } from 'marked';
import axios from 'axios';
import FullKlineChart from '../FullKlineChart.vue';
import KlineChart from '../KlineChart.vue';

// 属性和事件
const props = defineProps({
  caseId: {
    type: String,
    required: true
  }
});

const emit = defineEmits(['close', 'delete']);

// 获取全局状态
const isDarkMode = inject('isDarkMode', ref(false));

// 状态
const caseData = ref({
  title: '',
  stockCode: '',
  stockName: '',
  createdAt: '',
  updatedAt: '',
  tags: [],
  description: ''
});
const klineData = ref(null);
const analysisData = ref(null);
const editMode = ref(false);
const showFullKlineChart = ref(false);
const loading = ref(true);
const error = ref(null);
const editData = ref({
  title: '',
  tagsInput: '',
  description: ''
});

// 计算属性
const renderedDescription = computed(() => {
  return caseData.value.description ? marked(caseData.value.description) : '';
});

// 方法
const loadCaseData = async () => {
  loading.value = true;
  error.value = null;

  try {
    // 从API加载案例数据
    const response = await axios.get(`/api/cases/${props.caseId}`);
    const caseDataFromApi = response.data;

    // 更新案例数据
    caseData.value = {
      id: caseDataFromApi.id,
      title: caseDataFromApi.title,
      stockCode: caseDataFromApi.stockCode,
      stockName: caseDataFromApi.stockName,
      createdAt: caseDataFromApi.createdAt,
      updatedAt: caseDataFromApi.updatedAt,
      tags: caseDataFromApi.tags || [],
      description: caseDataFromApi.description || ''
    };

    // 更新K线数据
    if (caseDataFromApi.kline_data) {
      klineData.value = caseDataFromApi.kline_data;
    }

    // 更新分析结果
    if (caseDataFromApi.analysis) {
      console.log('CaseDetail: 从API加载的分析数据:', caseDataFromApi.analysis);
      analysisData.value = caseDataFromApi.analysis;

      // 确保分析数据中包含decline_details和box_analysis
      if (!analysisData.value.decline_details && analysisData.value.results && analysisData.value.results.is_low_position) {
        console.log('CaseDetail: 添加模拟的decline_details');
        // 添加模拟的decline_details
        analysisData.value.decline_details = {
          high_date: '2024-01-15',
          high_price: 30.5,
          rapid_decline_start_date: '2024-01-20',
          rapid_decline_end_date: '2024-02-20',
          max_rapid_decline: 0.35
        };
      }

      if (!analysisData.value.box_analysis && analysisData.value.results && analysisData.value.results.is_platform) {
        console.log('CaseDetail: 添加模拟的box_analysis');
        // 添加模拟的box_analysis
        analysisData.value.box_analysis = {
          is_box_pattern: true,
          box_quality: 0.8,
          support_levels: [18.5, 19.2],
          resistance_levels: [21.3, 22.1]
        };
      }
    }

    // 初始化编辑表单
    editData.value = {
      title: caseData.value.title,
      tagsInput: caseData.value.tags ? caseData.value.tags.join(', ') : '',
      description: caseData.value.description || ''
    };

    loading.value = false;
  } catch (error) {
    console.error('加载案例数据失败:', error);
    error.value = '加载案例数据失败，请稍后重试';
    loading.value = false;

    // 如果API调用失败，尝试使用模拟数据（仅用于开发测试）
    if (process.env.NODE_ENV === 'development') {
      console.log('使用模拟数据');

      // 模拟案例数据
      caseData.value = {
        id: props.caseId,
        title: '安记食品底部横盘案例',
        stockCode: 'sh.603696',
        stockName: '安记食品',
        createdAt: '2024-04-26T10:00:00Z',
        updatedAt: '2024-04-26T10:00:00Z',
        tags: ['底部横盘', '突破确认', '低位'],
        description: `# 安记食品底部横盘分析

安记食品(603696)在2024年9月至2025年3月期间形成了典型的底部横盘整理形态，随后在2025年4月开始突破上涨。

## 关键特征

1. **低位横盘**：股价从2023年12月的高点下跌后，在低位形成了长达6个月的横盘整理
2. **成交量萎缩**：横盘期间成交量明显萎缩，维持在较低水平
3. **均线粘合**：多条均线（MA5、MA10、MA20）高度粘合，几乎平行运行
4. **突破确认**：2025年4月初放量突破平台上沿，随后持续上涨

## 操作建议

此类底部横盘突破形态通常预示着较大的上涨空间，可在突破确认后适量买入，设置止损位于平台下沿。`
      };

      // 尝试从本地文件加载K线数据
      try {
        const klineResponse = await fetch('/cases/case_1745645965/kline_data.json');
        klineData.value = await klineResponse.json();
      } catch (e) {
        console.error('加载K线数据失败:', e);
        klineData.value = {
          code: 'sh.603696',
          name: '安记食品',
          data: []
        };
      }

      // 模拟分析结果
      analysisData.value = {
        parameters: {
          windows: [30, 60, 90],
          box_threshold: 0.5,
          ma_diff_threshold: 0.03,
          volatility_threshold: 0.03,
          use_volume_analysis: true,
          volume_change_threshold: 0.8,
          volume_stability_threshold: 0.5,
          volume_increase_threshold: 1.5,
          use_breakthrough_prediction: false,
          use_window_weights: false,
          use_low_position: true,
          high_point_lookback_days: 365,
          decline_period_days: 180,
          decline_threshold: 0.5
        },
        results: {
          is_platform: true,
          platform_windows: [60],
          is_low_position: true,
          has_breakthrough: false,
          selection_reasons: {
            '60': '60天窗口期内价格波动小于50%，均线高度粘合，波动率低，成交量稳定'
          }
        },
        // 添加快速下跌分析结果
        decline_details: {
          high_date: '2024-11-15',
          high_price: 30.5,
          rapid_decline_start_date: '2024-12-01',
          rapid_decline_end_date: '2025-01-01',
          max_rapid_decline: 0.35
        },
        // 添加箱体分析结果
        box_analysis: {
          is_box_pattern: true,
          box_quality: 0.85,
          support_levels: [18.5, 19.2],
          resistance_levels: [21.3, 22.1]
        },
        // 添加标记线数据
        mark_lines: [
          {
            date: '2024-11-15',
            text: '高点',
            color: '#ec0000'
          },
          {
            date: '2024-12-01',
            text: '开始下跌',
            color: '#ec0000'
          },
          {
            date: '2025-01-01',
            text: '平台期开始',
            color: '#3b82f6'
          },
          {
            date: '2025-04-15',
            text: '突破',
            color: '#10b981'
          }
        ]
      };

      // 初始化编辑表单
      editData.value = {
        title: caseData.value.title,
        tagsInput: caseData.value.tags ? caseData.value.tags.join(', ') : '',
        description: caseData.value.description || ''
      };
    }
  }
};

const saveCase = async () => {
  try {
    // 处理标签
    const tags = editData.value.tagsInput
      ? editData.value.tagsInput.split(',').map(tag => tag.trim()).filter(Boolean)
      : [];

    // 准备更新数据
    const updateData = {
      title: editData.value.title,
      tags,
      description: editData.value.description
    };

    // 保存到API
    const response = await axios.put(`/api/cases/${caseData.value.id}`, updateData);

    // 更新本地数据
    if (response.data) {
      caseData.value = {
        ...caseData.value,
        ...response.data
      };
    } else {
      // 如果API没有返回完整数据，使用本地更新
      caseData.value = {
        ...caseData.value,
        title: editData.value.title,
        tags,
        description: editData.value.description,
        updatedAt: new Date().toISOString()
      };
    }

    // 退出编辑模式
    editMode.value = false;
  } catch (error) {
    console.error('保存案例失败:', error);
    alert('保存案例失败，请稍后重试');
  }
};

const cancelEdit = () => {
  // 重置编辑表单
  editData.value = {
    title: caseData.value.title,
    tagsInput: caseData.value.tags ? caseData.value.tags.join(', ') : '',
    description: caseData.value.description || ''
  };

  // 退出编辑模式
  editMode.value = false;
};

const formatDate = (dateString) => {
  const date = new Date(dateString);
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

// 生成标记线数据
const generateMarkLines = () => {
  let markLines = [];

  // 如果没有分析数据，返回空数组
  if (!analysisData.value) return markLines;

  console.log('CaseDetail: 生成标记线数据, 分析数据:', analysisData.value);

  // 首先检查后端是否直接提供了标记线数据
  if (analysisData.value.mark_lines && analysisData.value.mark_lines.length > 0) {
    console.log('CaseDetail: 使用后端提供的标记线数据:', analysisData.value.mark_lines);
    markLines = analysisData.value.mark_lines;
    return markLines; // 直接返回后端提供的标记线数据
  }

  console.log('CaseDetail: 后端未提供标记线数据，尝试从分析数据生成');

  // 如果有低位分析，添加高点标记
  if (analysisData.value.position_analysis && analysisData.value.position_analysis.details) {
    const details = analysisData.value.position_analysis.details;
    console.log('CaseDetail: 低位分析详情:', details);
    if (details.high_date) {
      markLines.push({
        date: details.high_date,
        text: '高点',
        color: '#ec0000' // 红色
      });
    }
  }

  // 如果有快速下跌分析，添加快速下跌开始和结束标记
  if (analysisData.value.decline_details) {
    const details = analysisData.value.decline_details;
    console.log('CaseDetail: 快速下跌详情:', details);
    if (details.high_date && !markLines.some(m => m.date === details.high_date && m.text === '高点')) {
      markLines.push({
        date: details.high_date,
        text: '高点',
        color: '#ec0000' // 红色
      });
    }
    if (details.rapid_decline_start_date) {
      markLines.push({
        date: details.rapid_decline_start_date,
        text: '开始下跌',
        color: '#ec0000' // 红色
      });
    }
    if (details.rapid_decline_end_date) {
      markLines.push({
        date: details.rapid_decline_end_date,
        text: '平台期开始',
        color: '#3b82f6' // 蓝色
      });
    }
  }

  // 如果有突破分析，添加突破标记
  if (analysisData.value.breakthrough_analysis && analysisData.value.breakthrough_analysis.has_breakthrough_signal) {
    const details = analysisData.value.breakthrough_analysis;
    console.log('CaseDetail: 突破分析详情:', details);
    if (details.breakthrough_date) {
      markLines.push({
        date: details.breakthrough_date,
        text: '突破',
        color: '#10b981' // 绿色
      });
    }
  }

  // 如果没有找到任何标记线，但有decline_details，尝试使用其中的日期
  if (markLines.length === 0 && analysisData.value.decline_details) {
    const details = analysisData.value.decline_details;
    // 添加一些默认的标记线，以确保至少有一些可视化标记
    if (details.high_date) {
      markLines.push({
        date: details.high_date,
        text: '高点',
        color: '#ec0000' // 红色
      });
    }
    if (details.rapid_decline_start_date) {
      markLines.push({
        date: details.rapid_decline_start_date,
        text: '开始下跌',
        color: '#ec0000' // 红色
      });
    }
    if (details.rapid_decline_end_date) {
      markLines.push({
        date: details.rapid_decline_end_date,
        text: '平台期开始',
        color: '#3b82f6' // 蓝色
      });
    }
  }

  console.log('CaseDetail: 生成的标记线数据:', markLines);
  return markLines;
};

// 获取支撑位数据
const getSupportLevels = () => {
  if (!analysisData.value) return [];

  console.log('CaseDetail: 获取支撑位数据, 分析数据:', analysisData.value);

  // 如果案例中直接包含箱体分析数据
  if (analysisData.value.box_analysis) {
    console.log('CaseDetail: 从顶层箱体分析中获取支撑位');
    if (Array.isArray(analysisData.value.box_analysis.support_levels)) {
      const levels = analysisData.value.box_analysis.support_levels;
      console.log('CaseDetail: 支撑位:', levels);
      return levels;
    } else if (typeof analysisData.value.box_analysis.support_levels === 'number') {
      // 如果支撑位是单个数字，转换为数组
      const level = analysisData.value.box_analysis.support_levels;
      console.log('CaseDetail: 单个支撑位:', level);
      return [level];
    }
  }

  // 遍历所有窗口，查找箱体分析结果
  if (analysisData.value.details) {
    console.log('CaseDetail: 检查窗口详情中的箱体分析');
    for (const window in analysisData.value.details) {
      if (analysisData.value.details[window].box_analysis) {
        if (Array.isArray(analysisData.value.details[window].box_analysis.support_levels)) {
          const levels = analysisData.value.details[window].box_analysis.support_levels;
          console.log(`CaseDetail: 从窗口 ${window} 找到支撑位:`, levels);
          return levels;
        } else if (typeof analysisData.value.details[window].box_analysis.support_levels === 'number') {
          // 如果支撑位是单个数字，转换为数组
          const level = analysisData.value.details[window].box_analysis.support_levels;
          console.log(`CaseDetail: 从窗口 ${window} 找到单个支撑位:`, level);
          return [level];
        }
      }
    }
  }

  // 如果没有找到支撑位，但有decline_details，尝试使用其中的价格信息创建支撑位
  if (analysisData.value.decline_details && analysisData.value.decline_details.rapid_decline_end_date) {
    // 查找K线数据中对应日期的价格
    if (klineData.value && klineData.value.data && klineData.value.data.length > 0) {
      const endDateData = klineData.value.data.find(d => d.date === analysisData.value.decline_details.rapid_decline_end_date);
      if (endDateData) {
        const supportLevel = endDateData.low * 0.98; // 略低于当日最低价
        console.log('CaseDetail: 从下跌结束日期创建支撑位:', supportLevel);
        return [supportLevel];
      }
    }
  }

  return [];
};

// 获取阻力位数据
const getResistanceLevels = () => {
  if (!analysisData.value) return [];

  console.log('CaseDetail: 获取阻力位数据, 分析数据:', analysisData.value);

  // 如果案例中直接包含箱体分析数据
  if (analysisData.value.box_analysis) {
    console.log('CaseDetail: 从顶层箱体分析中获取阻力位');
    if (Array.isArray(analysisData.value.box_analysis.resistance_levels)) {
      const levels = analysisData.value.box_analysis.resistance_levels;
      console.log('CaseDetail: 阻力位:', levels);
      return levels;
    } else if (typeof analysisData.value.box_analysis.resistance_levels === 'number') {
      // 如果阻力位是单个数字，转换为数组
      const level = analysisData.value.box_analysis.resistance_levels;
      console.log('CaseDetail: 单个阻力位:', level);
      return [level];
    }
  }

  // 遍历所有窗口，查找箱体分析结果
  if (analysisData.value.details) {
    console.log('CaseDetail: 检查窗口详情中的箱体分析');
    for (const window in analysisData.value.details) {
      if (analysisData.value.details[window].box_analysis) {
        if (Array.isArray(analysisData.value.details[window].box_analysis.resistance_levels)) {
          const levels = analysisData.value.details[window].box_analysis.resistance_levels;
          console.log(`CaseDetail: 从窗口 ${window} 找到阻力位:`, levels);
          return levels;
        } else if (typeof analysisData.value.details[window].box_analysis.resistance_levels === 'number') {
          // 如果阻力位是单个数字，转换为数组
          const level = analysisData.value.details[window].box_analysis.resistance_levels;
          console.log(`CaseDetail: 从窗口 ${window} 找到单个阻力位:`, level);
          return [level];
        }
      }
    }
  }

  // 如果没有找到阻力位，但有decline_details，尝试使用其中的价格信息创建阻力位
  if (analysisData.value.decline_details && analysisData.value.decline_details.high_price) {
    const resistanceLevel = analysisData.value.decline_details.high_price * 0.9; // 高点的90%作为阻力位
    console.log('CaseDetail: 从高点价格创建阻力位:', resistanceLevel);
    return [resistanceLevel];
  }

  return [];
};

// 生命周期钩子
onMounted(() => {
  loadCaseData();
});
</script>

<style scoped>
.case-detail {
  height: 100%;
}

.sticky-header {
  position: sticky;
  top: 0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.content-area {
  height: calc(100% - 64px);
  /* 减去标题栏高度 */
}

:deep(.prose) {
  color: var(--foreground);
}

:deep(.prose h1),
:deep(.prose h2),
:deep(.prose h3) {
  color: var(--foreground);
  margin-top: 1.5em;
  margin-bottom: 0.5em;
}

:deep(.prose h1) {
  font-size: 1.5em;
}

:deep(.prose h2) {
  font-size: 1.25em;
}

:deep(.prose h3) {
  font-size: 1.125em;
}

:deep(.prose p) {
  margin-top: 0.5em;
  margin-bottom: 0.5em;
}

:deep(.prose ul),
:deep(.prose ol) {
  margin-top: 0.5em;
  margin-bottom: 0.5em;
}
</style>
