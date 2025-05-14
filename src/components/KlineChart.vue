<template>
  <div ref="chartRef" :style="{ height: height, width: width }" class="chart-wrapper relative">
    <!-- 加载中提示 -->
    <div v-if="loading"
      class="absolute inset-0 flex items-center justify-center bg-background/50 backdrop-blur-sm z-10">
      <div class="flex flex-col items-center">
        <div class="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
        <span class="mt-2 text-xs text-muted-foreground">加载中...</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue';
import * as echarts from 'echarts/core';
import { CandlestickChart, LineChart } from 'echarts/charts';
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  DataZoomComponent,
  LegendComponent,
  MarkAreaComponent // Optional: for highlighting zones
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
// GSAP 动画库
import { gsap } from 'gsap';

echarts.use([
  TitleComponent,
  TooltipComponent,
  GridComponent,
  DataZoomComponent,
  LegendComponent,
  MarkAreaComponent,
  CandlestickChart,
  LineChart, // If you want Moving Averages overlayed
  CanvasRenderer
]);

const props = defineProps({
  klineData: {
    type: Array,
    required: true,
    default: () => []
  },
  height: {
    type: String,
    default: '300px'
  },
  width: {
    type: String,
    default: '100%'
  },
  title: {
    type: String,
    default: ''
  },
  isDarkMode: {
    type: Boolean,
    default: false
  },
  // 标记线数据
  markLines: {
    type: Array,
    default: () => []
    // 格式: [{date: '2024-11-15', text: '高点', color: 'red'}, ...]
  },
  // 支撑位数据
  supportLevels: {
    type: Array,
    default: () => []
    // 格式: [7.05, 7.17, ...]
  },
  // 阻力位数据
  resistanceLevels: {
    type: Array,
    default: () => []
    // 格式: [7.58, 7.66, ...]
  }
});

const loading = ref(true); // 添加加载状态

const chartRef = ref(null);
let chartInstance = null;

// Function to process raw data for ECharts candlestick
const processData = (rawData) => {
  if (!Array.isArray(rawData)) return { dates: [], values: [] };
  const dates = rawData.map(item => item.date);
  // ECharts candlestick format: [open, close, low, high]
  const values = rawData.map(item => [item.open, item.close, item.low, item.high]);
  return { dates, values };
};

// 计算移动平均线
const calculateMA = (dayCount, data) => {
  if (!data || !data.length) return [];

  const result = [];
  for (let i = 0; i < data.length; i++) {
    if (i < dayCount - 1) {
      // 不足dayCount天时，设为null
      result.push(null);
      continue;
    }
    let sum = 0;
    for (let j = 0; j < dayCount; j++) {
      sum += data[i - j][1]; // 使用收盘价计算
    }
    result.push(+(sum / dayCount).toFixed(2));
  }
  return result;
};

const initChart = () => {
  if (!chartRef.value) return;

  loading.value = true; // 显示加载状态

  // 添加初始透明度为0
  if (chartRef.value) {
    chartRef.value.style.opacity = '0';
  }

  try {
    // 使用渲染器选项初始化图表，提高渲染质量
    chartInstance = echarts.init(chartRef.value, props.isDarkMode ? 'dark' : null, {
      renderer: 'canvas', // 使用 canvas 渲染器
      useDirtyRect: false, // 禁用脏矩形优化，提高渲染质量
      devicePixelRatio: window.devicePixelRatio // 使用设备像素比，提高清晰度
    });
    setOptions();

    // 使用GSAP添加淡入动画
    gsap.to(chartRef.value, {
      opacity: 1,
      duration: 0.8,
      ease: "power2.out"
    });

    // Optional: Resize chart with window resize
    window.addEventListener('resize', resizeChart);
  } catch (error) {
    console.error('初始化图表失败:', error);
  } finally {
    loading.value = false; // 隐藏加载状态
  }
};

const setOptions = () => {
  if (!chartInstance || !props.klineData) return;

  const { dates, values } = processData(props.klineData);

  // 处理标记线数据
  const markLines = [];
  console.log('KlineChart: 处理标记线数据, 原始数据:', props.markLines);
  console.log('KlineChart: 日期数据:', dates);
  console.log('KlineChart: 支撑位数据:', props.supportLevels);
  console.log('KlineChart: 阻力位数据:', props.resistanceLevels);

  // 检查日期格式
  if (dates.length > 0) {
    console.log('KlineChart: 日期格式示例:', dates[0], typeof dates[0]);
  }

  // 日期匹配函数，支持多种格式
  const findDateIndex = (targetDate, dateArray) => {
    if (!targetDate) return -1;

    // 标准化目标日期
    let dateStr = targetDate;
    if (typeof dateStr === 'string') {
      // 如果日期包含时间部分，只保留日期部分
      if (dateStr.includes(' ')) {
        dateStr = dateStr.split(' ')[0];
      }
      // 确保日期格式为YYYY-MM-DD
      if (dateStr.includes('/')) {
        const parts = dateStr.split('/');
        if (parts.length === 3) {
          dateStr = `${parts[0]}-${parts[1].padStart(2, '0')}-${parts[2].padStart(2, '0')}`;
        }
      }
    }

    // 1. 尝试完全匹配
    for (let i = 0; i < dateArray.length; i++) {
      const currentDate = dateArray[i].split(' ')[0]; // 只保留日期部分
      if (currentDate === dateStr) {
        console.log(`KlineChart: 找到完全匹配日期 ${dateStr}`);
        return i;
      }
    }

    // 2. 尝试匹配年月部分
    if (dateStr.length >= 7) {
      const yearMonth = dateStr.substring(0, 7); // YYYY-MM
      for (let i = 0; i < dateArray.length; i++) {
        const currentDate = dateArray[i].split(' ')[0]; // 只保留日期部分
        if (currentDate.startsWith(yearMonth)) {
          console.log(`KlineChart: 找到年月匹配 ${yearMonth}`);
          return i;
        }
      }
    }

    // 3. 尝试匹配年部分
    if (dateStr.length >= 4) {
      const year = dateStr.substring(0, 4); // YYYY
      for (let i = 0; i < dateArray.length; i++) {
        const currentDate = dateArray[i].split(' ')[0]; // 只保留日期部分
        if (currentDate.startsWith(year)) {
          console.log(`KlineChart: 找到年份匹配 ${year}`);
          return i;
        }
      }
    }

    // 4. 尝试查找最接近的日期（向前查找）
    const targetTimestamp = new Date(dateStr).getTime();
    if (!isNaN(targetTimestamp)) {
      let closestIndex = -1;
      let minDiff = Infinity;

      for (let i = 0; i < dateArray.length; i++) {
        const currentDate = dateArray[i].split(' ')[0];
        const currentTimestamp = new Date(currentDate).getTime();
        if (!isNaN(currentTimestamp)) {
          const diff = Math.abs(targetTimestamp - currentTimestamp);
          if (diff < minDiff) {
            minDiff = diff;
            closestIndex = i;
          }
        }
      }

      if (closestIndex !== -1) {
        console.log(`KlineChart: 找到最接近的日期 ${dateArray[closestIndex]}`);
        return closestIndex;
      }
    }

    console.warn(`KlineChart: 无法找到匹配的日期 ${targetDate}`);
    return -1;
  };

  if (props.markLines && props.markLines.length > 0) {
    // 将日期转换为x轴索引
    props.markLines.forEach(mark => {
      console.log('KlineChart: 处理标记线:', mark);

      // 处理水平标记线
      if (mark.type === 'horizontal' && mark.value !== undefined) {
        console.log('KlineChart: 处理水平标记线:', mark);
        markLines.push({
          name: mark.text,
          yAxis: mark.value,
          label: {
            formatter: mark.text + ': ' + mark.value.toFixed(2),
            position: 'end',
            color: mark.color || 'auto',
            fontSize: 10,
            backgroundColor: props.isDarkMode ? 'rgba(0, 0, 0, 0.5)' : 'rgba(255, 255, 255, 0.5)',
            padding: [2, 4],
            borderRadius: 2
          },
          lineStyle: {
            color: mark.color || '#666',
            type: 'solid',
            width: 1,
            opacity: 0.8
          }
        });
        return;
      }

      // 处理垂直标记线（日期标记）
      if (!mark.date) {
        console.warn('KlineChart: 标记线缺少日期:', mark);
        return;
      }

      const dateIndex = findDateIndex(mark.date, dates);
      console.log('KlineChart: 日期索引:', dateIndex, '日期:', mark.date);

      if (dateIndex !== -1) {
        markLines.push({
          name: mark.text,
          xAxis: dateIndex,
          label: {
            formatter: mark.text,
            position: 'top',
            color: mark.color || 'auto',
            fontSize: 10,
            backgroundColor: props.isDarkMode ? 'rgba(0, 0, 0, 0.5)' : 'rgba(255, 255, 255, 0.5)',
            padding: [2, 4],
            borderRadius: 2
          },
          lineStyle: {
            color: mark.color || '#666',
            type: 'dashed',
            width: 1,
            opacity: 0.6
          }
        });
      } else {
        // 如果找不到相近日期，添加到图表中间位置
        console.error('KlineChart: 无法找到日期匹配，添加默认标记线');
        const middleIndex = Math.floor(dates.length / 2);
        markLines.push({
          name: mark.text,
          xAxis: middleIndex,
          label: {
            formatter: mark.text + '(估计位置)',
            position: 'top',
            color: mark.color || 'auto',
            fontSize: 10,
            backgroundColor: props.isDarkMode ? 'rgba(0, 0, 0, 0.5)' : 'rgba(255, 255, 255, 0.5)',
            padding: [2, 4],
            borderRadius: 2
          },
          lineStyle: {
            color: mark.color || '#666',
            type: 'dashed',
            width: 1,
            opacity: 0.6
          }
        });
      }
    });
  } else {
    console.log('KlineChart: 没有标记线数据或数组为空');
  }

  console.log('KlineChart: 生成的标记线数据:', markLines);

  // 简化版的K线图配置，作为缩略图使用
  const option = {
    animation: false,
    // 根据传入的isDarkMode属性设置背景色
    backgroundColor: 'transparent', // 使用透明背景，让容器的背景色显示出来
    boundaryGap: [0.05, 0.05], // 为图表添加边界间隙，防止内容超出可视区域
    title: {
      show: !!props.title,
      text: props.title,
      textStyle: {
        fontSize: 12,
        fontWeight: 'normal',
        color: props.isDarkMode ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.7)'
      },
      left: 'center',
      top: 5 // 增加顶部距离，使标题更靠近K线图
    },
    legend: {
      show: true,
      data: ['K线', 'MA10', 'MA30'],
      bottom: 0, // 调整底部距离，使其更靠近框的底部，只留一点空间
      left: 'center',
      itemWidth: 8,
      itemHeight: 8,
      padding: [0, 0, 0, 0], // 移除内边距
      textStyle: {
        fontSize: 10,
        color: props.isDarkMode ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.6)'
      }
    },
    tooltip: {
      show: false // 禁用tooltip，简化交互
    },
    grid: {
      left: '5%', // 减少左侧空间，使图表左移
      right: '8%', // 保持右侧空间不变
      bottom: '8%', // 大幅减少底部空间，使图表向下拉伸更多
      top: '18%', // 略微减少顶部空间，使图表向上拉伸
      containLabel: true, // 包含轴标签
      show: false, // 隐藏网格边框线
      borderWidth: 0 // 确保没有边框
    },
    xAxis: {
      type: 'category',
      data: dates,
      show: true, // 显示x轴
      boundaryGap: false, // 与完整K线图保持一致
      axisLine: {
        onZero: false,
        lineStyle: {
          color: props.isDarkMode ? '#666' : '#333'
        }
      },
      axisTick: {
        show: false // 不显示刻度线
      },
      axisLabel: {
        show: true,
        interval: function (index, value) {
          // 确保最后一天的日期始终显示
          const totalPoints = dates.length;
          if (index === totalPoints - 1) return true; // 始终显示最后一天

          // 根据日期总数选择合适的间隔
          if (totalPoints <= 7) return true; // 数据点很少时全部显示
          if (totalPoints <= 14) return index % 2 === 0; // 两周内数据显示每隔一天
          if (totalPoints <= 30) return index % 5 === 0; // 一个月内数据显示每隔5天
          if (totalPoints <= 60) return index % 10 === 0; // 两个月内数据显示每隔10天
          if (totalPoints <= 90) return index % 15 === 0; // 三个月内数据显示每隔15天
          return index % 20 === 0; // 更长时间范围显示每隔20天
        },
        fontSize: 10,
        margin: 14, // 增加标签与轴的距离
        align: 'center', // 居中对齐
        color: props.isDarkMode ? 'rgba(255, 255, 255, 0.5)' : 'rgba(0, 0, 0, 0.5)',
        formatter: function (value) {
          // 更简洁的日期格式
          if (value && value.length >= 5) {
            const parts = value.split(' ')[0].split('-'); // 分割日期部分，忽略时间
            if (parts.length >= 3) {
              // 只显示月/日，不带前导零
              const month = parseInt(parts[1], 10);
              const day = parseInt(parts[2], 10);
              return month + '/' + day; // 使用斜杠分隔，更简洁
            }
          }
          return value;
        },
        rotate: 0 // 确保标签水平显示
      },
      splitLine: {
        show: false
      }
    },
    yAxis: {
      scale: true,
      show: true, // 显示y轴
      position: 'left', // 将y轴放在左侧
      splitArea: {
        show: true,
        areaStyle: {
          color: props.isDarkMode ?
            ['rgba(30, 41, 59, 0.2)', 'rgba(30, 41, 59, 0.1)'] :
            ['rgba(250, 250, 250, 0.3)', 'rgba(240, 240, 240, 0.3)']
        }
      },
      axisLine: {
        lineStyle: {
          color: props.isDarkMode ? '#666' : '#333'
        }
      },
      axisTick: {
        show: false
      },
      splitLine: {
        show: true,
        lineStyle: {
          color: props.isDarkMode ? 'rgba(100, 100, 100, 0.2)' : 'rgba(200, 200, 200, 0.2)'
        }
      },
      axisLabel: {
        show: true,
        inside: false, // 将标签放在轴外部
        fontSize: 10,
        margin: 8, // 与完整K线图保持一致
        color: props.isDarkMode ? '#ccc' : '#333', // 与完整K线图保持一致
        formatter: function (value) {
          // 简化数值显示
          if (value >= 1000) {
            return (value / 1000).toFixed(1) + 'k';
          }
          return value.toFixed(1); // 保留一位小数，使显示更一致
        },
        showMinLabel: true, // 显示最小值标签
        showMaxLabel: true, // 显示最大值标签
        rich: { // 使用富文本配置，增强可读性
          value: {
            padding: [0, 0, 0, 5],
            width: 40
          }
        }
      }
    },
    series: [
      {
        name: 'K线',
        type: 'candlestick',
        data: values,
        itemStyle: {
          // 根据主题使用不同的颜色
          color: props.isDarkMode ? '#a15c5c' : '#ec0000', // 深色模式下使用莫兰迪红色
          color0: props.isDarkMode ? '#5b7a9d' : '#60a5fa', // 深色模式下使用莫兰迪蓝色
          borderColor: props.isDarkMode ? '#a15c5c' : '#ec0000',
          borderColor0: props.isDarkMode ? '#5b7a9d' : '#60a5fa'
        },
        barWidth: '60%', // 调整蜡烛宽度，与完整K线图保持一致
        large: true, // 优化大数据量渲染
        largeThreshold: 100, // 超过100个数据点时启用大数据量优化
        emphasis: {
          itemStyle: {
            borderWidth: 2 // 与完整K线图保持一致
          }
        },
        // 添加标记线
        markLine: {
          silent: true, // 不响应鼠标事件
          symbol: ['none', 'none'], // 不显示起点和终点标记
          label: {
            show: true,
            position: 'end',
            formatter: '{b}',
            fontSize: 10,
            color: props.isDarkMode ? '#fff' : '#333'
          },
          lineStyle: {
            type: 'dashed',
            width: 1
          },
          data: [
            // 垂直标记线（日期标记）
            ...markLines,

            // 支撑位水平线
            ...props.supportLevels.map(level => ({
              name: `支撑位: ${level.toFixed(2)}`,
              yAxis: level,
              lineStyle: {
                color: props.isDarkMode ? '#8cb58c' : '#10b981', // 深色模式下使用莫兰迪绿色
                type: 'solid',
                width: 1,
                opacity: 0.8
              }
            })),

            // 阻力位水平线
            ...props.resistanceLevels.map(level => ({
              name: `阻力位: ${level.toFixed(2)}`,
              yAxis: level,
              lineStyle: {
                color: props.isDarkMode ? '#a15c5c' : '#ec0000', // 深色模式下使用莫兰迪红色
                type: 'solid',
                width: 1,
                opacity: 0.8
              }
            }))
          ]
        }
      },
      {
        name: 'MA10',
        type: 'line',
        data: calculateMA(10, values),
        smooth: true,
        showSymbol: false,
        lineStyle: {
          width: 1.5, // 减小线宽
          color: props.isDarkMode ? '#6b8cad' : '#3b82f6', // 使用与蜡烛图协调的蓝色
          opacity: 0.6 // 增加透明度
        }
      },
      {
        name: 'MA30',
        type: 'line',
        data: calculateMA(30, values),
        smooth: true,
        showSymbol: false,
        lineStyle: {
          width: 1.5, // 减小线宽
          color: props.isDarkMode ? '#8a8a99' : '#6b7280', // 使用与蜡烛图协调的灰色
          opacity: 0.6 // 增加透明度
        }
      }
    ]
  };
  chartInstance.setOption(option);
}

const resizeChart = () => {
  if (chartInstance) {
    chartInstance.resize();
  }
};

onMounted(async () => {
  await nextTick(); // Ensure DOM element is ready
  initChart();
});

onUnmounted(() => {
  window.removeEventListener('resize', resizeChart);
  if (chartInstance) {
    chartInstance.dispose();
  }
});

// 监听数据变化
watch(() => props.klineData, (newData) => {
  loading.value = true; // 显示加载状态

  try {
    if (chartInstance && newData) {
      setOptions(); // Re-render with new data
    } else if (chartInstance && !newData) {
      chartInstance.clear(); // Clear chart if data becomes null/empty
    } else if (!chartInstance && newData) {
      initChart(); // Initialize if chart wasn't ready but data arrived
    }
  } catch (error) {
    console.error('更新图表失败:', error);
  } finally {
    loading.value = false; // 隐藏加载状态
  }
}, { deep: true }); // Use deep watch if klineData structure might change internally

// 监听暗色模式变化
watch(() => props.isDarkMode, () => {
  if (chartInstance) {
    loading.value = true;
    try {
      // 销毁旧实例并重新创建
      chartInstance.dispose();
      nextTick(() => {
        initChart();
      });
    } catch (error) {
      console.error('切换主题失败:', error);
      loading.value = false;
    }
  }
});

// 删除重复的监听器，已在上面添加了暗色模式变化监听

</script>

<style scoped>
/* 防止鼠标悬停时的模糊效果 */
div {
  transform: translateZ(0);
  /* 启用硬件加速 */
  backface-visibility: hidden;
  /* 防止 3D 变换时的模糊 */
  perspective: 1000px;
  /* 设置透视效果 */
  will-change: transform;
  /* 告诉浏览器该元素会变化，优化渲染 */
  image-rendering: -webkit-optimize-contrast;
  /* 提高图像渲染质量 */
  image-rendering: crisp-edges;
  /* 提高边缘清晰度 */
  -webkit-font-smoothing: antialiased;
  /* 字体平滑 */
  -moz-osx-font-smoothing: grayscale;
  /* 字体平滑 */
}

/* 确保 canvas 元素清晰 */
canvas {
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
}

/* 图表容器特定样式 */
.chart-wrapper {
  position: relative;
  z-index: 1;
  isolation: isolate;
  transform: translateZ(0);
  backface-visibility: hidden;
  will-change: transform;
  box-shadow: none !important;
  border-radius: 4px;
  overflow: hidden;
}

/* 确保图表容器在所有主题下都能正确显示 */
:deep(.echarts) {
  width: 100% !important;
  height: 100% !important;
}

:deep(canvas) {
  width: 100% !important;
  height: 100% !important;
}
</style>