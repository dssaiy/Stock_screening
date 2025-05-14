<template>
  <div>
    <!-- 弹出层背景 -->
    <div v-if="visible" class="fixed inset-0 bg-black bg-opacity-70 z-40 flex items-center justify-center"
      @click="close">
      <!-- 弹出层内容 -->
      <div class="bg-card dark:bg-card rounded-lg shadow-xl w-11/12 max-w-6xl max-h-[90vh] overflow-hidden" @click.stop>
        <!-- 弹出层头部 -->
        <div class="p-4 border-b border-border flex justify-between items-center">
          <h3 class="text-lg font-semibold">{{ title }}</h3>
          <button @click="close"
            class="px-4 py-2 rounded-md text-white bg-gundam-blue hover:bg-gundam-blue/90 active:bg-gundam-blue/70 transition-all transform hover:scale-105 active:scale-95 shadow-md flex items-center justify-center text-base">
            <i class="fas fa-times mr-2 text-lg"></i>
            关闭
          </button>
        </div>
        <!-- 弹出层内容 - K线图 -->
        <div class="p-4 w-full h-full">
          <div ref="chartRef"
            class="chart-wrapper w-full h-[600px] dark:bg-opacity-10 bg-opacity-5 bg-black dark:bg-white rounded-md">
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue';
import * as echarts from 'echarts';

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  title: {
    type: String,
    default: '股票详情'
  },
  klineData: {
    type: Array,
    default: () => []
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

const emit = defineEmits(['update:visible', 'close']);

const chartRef = ref(null);
let chartInstance = null;

// 关闭弹窗
const close = () => {
  emit('update:visible', false);
  emit('close');
};

// 处理原始数据
function processData (rawData) {
  if (!rawData || !Array.isArray(rawData)) return { categoryData: [], values: [], volumes: [] };

  let categoryData = [];
  let values = [];
  let volumes = [];

  for (let i = 0; i < rawData.length; i++) {
    const item = rawData[i];
    // 日期
    categoryData.push(item.date);
    // [开盘价, 收盘价, 最低价, 最高价]
    values.push([item.open, item.close, item.low, item.high]);
    // [索引, 成交量, 涨跌标记]
    volumes.push([i, item.volume || 0, item.open > item.close ? 1 : -1]);
  }

  return {
    categoryData,
    values,
    volumes
  };
}

// 计算移动平均线
function calculateMA (dayCount, data) {
  const result = [];
  for (let i = 0, len = data.values.length; i < len; i++) {
    if (i < dayCount) {
      result.push('-');
      continue;
    }
    let sum = 0;
    for (let j = 0; j < dayCount; j++) {
      sum += data.values[i - j][1];
    }
    result.push(+(sum / dayCount).toFixed(3));
  }
  return result;
}

// 初始化图表
const initChart = () => {
  console.log('初始化图表开始');
  if (!chartRef.value) {
    console.error('图表容器不存在');
    return;
  }

  // 如果已经存在图表实例，先销毁
  if (chartInstance) {
    console.log('销毁旧的图表实例');
    chartInstance.dispose();
    chartInstance = null;
  }

  try {
    // 添加初始透明度为0
    chartRef.value.style.opacity = '0';

    // 确保容器尺寸正确
    chartRef.value.style.width = '100%';
    chartRef.value.style.height = '600px';

    console.log('创建新的图表实例');
    // 使用渲染器选项初始化图表，提高渲染质量
    chartInstance = echarts.init(chartRef.value, props.isDarkMode ? 'dark' : null, {
      renderer: 'canvas',
      useDirtyRect: false,
      devicePixelRatio: window.devicePixelRatio
    });

    // 设置图表选项
    setOptions();

    // 添加淡入动画
    chartRef.value.style.opacity = '1';
    chartRef.value.style.transition = 'opacity 0.5s ease';

    // 窗口大小变化时调整图表大小
    window.removeEventListener('resize', resizeChart); // 先移除以防重复添加
    window.addEventListener('resize', resizeChart);

    console.log('图表初始化完成');
  } catch (error) {
    console.error('图表初始化失败:', error);
  }
};

// 获取主题颜色
const getThemeColor = (varName) => {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim() || '';
};

// 设置图表选项
const setOptions = () => {
  if (!chartInstance || !props.klineData) return;

  const data = processData(props.klineData);

  // 处理标记线数据
  const markLines = [];
  console.log('FullKlineChart: 处理标记线数据, 原始数据:', props.markLines);
  console.log('FullKlineChart: 日期数据:', data.categoryData);

  // 日期匹配函数，支持多种格式
  const findDateIndex = (targetDate, dateArray) => {
    // 1. 尝试完全匹配
    const exactIndex = dateArray.indexOf(targetDate);
    if (exactIndex !== -1) {
      console.log(`FullKlineChart: 找到完全匹配日期 ${targetDate}`);
      return exactIndex;
    }

    // 2. 尝试匹配年月日部分（忽略时间）
    const dateOnly = targetDate.split(' ')[0]; // 获取日期部分
    const dateOnlyIndex = dateArray.findIndex(d => d.startsWith(dateOnly));
    if (dateOnlyIndex !== -1) {
      console.log(`FullKlineChart: 找到日期部分匹配 ${dateOnly}`);
      return dateOnlyIndex;
    }

    // 3. 尝试匹配年月部分
    if (dateOnly.length >= 7) {
      const yearMonth = dateOnly.substring(0, 7); // YYYY-MM
      const yearMonthIndex = dateArray.findIndex(d => d.startsWith(yearMonth));
      if (yearMonthIndex !== -1) {
        console.log(`FullKlineChart: 找到年月匹配 ${yearMonth}`);
        return yearMonthIndex;
      }
    }

    // 4. 尝试匹配年部分
    if (dateOnly.length >= 4) {
      const year = dateOnly.substring(0, 4); // YYYY
      const yearIndex = dateArray.findIndex(d => d.startsWith(year));
      if (yearIndex !== -1) {
        console.log(`FullKlineChart: 找到年份匹配 ${year}`);
        return yearIndex;
      }
    }

    // 5. 尝试查找最接近的日期（向前查找）
    const targetTimestamp = new Date(dateOnly).getTime();
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
        console.log(`FullKlineChart: 找到最接近的日期 ${dateArray[closestIndex]}`);
        return closestIndex;
      }
    }

    console.warn(`FullKlineChart: 无法找到匹配的日期 ${targetDate}`);
    return -1;
  };

  if (props.markLines && props.markLines.length > 0) {
    // 将日期转换为x轴索引
    props.markLines.forEach(mark => {
      console.log('FullKlineChart: 处理标记线:', mark);

      // 处理水平标记线
      if (mark.type === 'horizontal' && mark.value !== undefined) {
        console.log('FullKlineChart: 处理水平标记线:', mark);
        markLines.push({
          name: mark.text,
          yAxis: mark.value,
          label: {
            formatter: mark.text + ': ' + mark.value.toFixed(2),
            position: 'end',
            color: mark.color || 'auto',
            fontSize: 12,
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
        console.warn('FullKlineChart: 标记线缺少日期:', mark);
        return;
      }

      const dateIndex = findDateIndex(mark.date, data.categoryData);
      console.log('FullKlineChart: 日期索引:', dateIndex, '日期:', mark.date);

      if (dateIndex !== -1) {
        markLines.push({
          name: mark.text,
          xAxis: dateIndex,
          label: {
            formatter: mark.text,
            position: 'top',
            color: mark.color || 'auto',
            fontSize: 12,
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
        console.error('FullKlineChart: 无法找到日期匹配，添加默认标记线');
        const middleIndex = Math.floor(data.categoryData.length / 2);
        markLines.push({
          name: mark.text,
          xAxis: middleIndex,
          label: {
            formatter: mark.text + '(估计位置)',
            position: 'top',
            color: mark.color || 'auto',
            fontSize: 12,
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
  }

  console.log('FullKlineChart: 生成的标记线数据:', markLines);

  // 颜色定义 - 根据主题使用不同的颜色
  // 深色模式下使用莫兰迪色系的颜色
  const upColor = props.isDarkMode ? '#a15c5c' : getThemeColor('--chart-1');   // 上涨颜色 - 莫兰迪红色
  const downColor = props.isDarkMode ? '#5b7a9d' : getThemeColor('--chart-2'); // 下跌颜色 - 莫兰迪蓝色
  const volumeUpColor = props.isDarkMode ? '#a15c5c' : '#ec0000';   // 备用红色 - 莫兰迪红色
  const volumeDownColor = props.isDarkMode ? '#5b7a9d' : '#60a5fa'; // 备用蓝色 - 莫兰迪蓝色

  const option = {
    animation: false,
    backgroundColor: props.isDarkMode ? 'rgba(30, 41, 59, 0.95)' : 'rgba(255, 255, 255, 0.95)', // 使用深色主题背景色
    legend: {
      bottom: 10,
      left: 'center',
      data: ['K线', 'MA5', 'MA10', 'MA20', 'MA30'],
      textStyle: {
        color: props.isDarkMode ? '#eee' : '#333'
      }
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      },
      borderWidth: 1,
      borderColor: props.isDarkMode ? '#555' : '#ccc',
      padding: 10,
      backgroundColor: props.isDarkMode ? 'rgba(50, 50, 50, 0.9)' : 'rgba(255, 255, 255, 0.9)',
      textStyle: {
        color: props.isDarkMode ? '#eee' : '#333'
      },
      position: function (pos, params, el, elRect, size) {
        const obj = { top: 10 };
        obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 30;
        return obj;
      }
    },
    axisPointer: {
      link: [{ xAxisIndex: 'all' }],
      label: {
        backgroundColor: '#777'
      }
    },
    toolbox: {
      feature: {
        dataZoom: {
          yAxisIndex: false
        },
        restore: {},
        saveAsImage: {},
        brush: {
          type: ['lineX', 'clear']
        }
      }
    },
    brush: {
      xAxisIndex: 'all',
      brushLink: 'all',
      outOfBrush: {
        // 使用与亮色模式相同的透明度设置，确保功能一致性
        colorAlpha: 0.3 // 设置适中的透明度，使非选中区域显示为空心但仍然可见
      },
      brushStyle: {
        borderWidth: 1,
        color: 'rgba(120, 140, 180, 0.1)',
        borderColor: 'rgba(120, 140, 180, 0.3)'
      },
      throttleType: 'debounce',
      throttleDelay: 300
    },
    visualMap: {
      show: false,
      seriesIndex: 5,
      dimension: 2,
      pieces: [
        {
          value: 1,
          color: volumeDownColor
        },
        {
          value: -1,
          color: volumeUpColor
        }
      ]
    },
    grid: [
      {
        left: '10%',
        right: '8%',
        height: '50%'
      },
      {
        left: '10%',
        right: '8%',
        top: '63%',
        height: '16%'
      }
    ],
    xAxis: [
      {
        type: 'category',
        data: data.categoryData,
        boundaryGap: false,
        axisLine: {
          onZero: false,
          lineStyle: {
            color: props.isDarkMode ? '#666' : '#333'
          }
        },
        splitLine: { show: false },
        axisLabel: {
          color: props.isDarkMode ? '#ccc' : '#333',
          interval: function (index, value) {
            // 确保最后一天的日期始终显示
            const totalPoints = data.categoryData.length;
            if (index === totalPoints - 1) return true; // 始终显示最后一天

            // 根据日期总数选择合适的间隔
            if (totalPoints <= 7) return true; // 数据点很少时全部显示
            if (totalPoints <= 14) return index % 2 === 0; // 两周内数据显示每隔一天
            if (totalPoints <= 30) return index % 5 === 0; // 一个月内数据显示每隔5天
            if (totalPoints <= 60) return index % 10 === 0; // 两个月内数据显示每隔10天
            if (totalPoints <= 90) return index % 15 === 0; // 三个月内数据显示每隔15天
            return index % 20 === 0; // 更长时间范围显示每隔20天
          },
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
          fontSize: 10,
          margin: 14, // 增加标签与轴的距离
          align: 'center', // 居中对齐
          rotate: 0 // 确保标签水平显示
        },
        min: 'dataMin',
        max: 'dataMax',
        axisPointer: {
          z: 100
        }
      },
      {
        type: 'category',
        gridIndex: 1,
        data: data.categoryData,
        boundaryGap: false,
        axisLine: {
          onZero: false,
          lineStyle: {
            color: props.isDarkMode ? '#666' : '#333'
          }
        },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        min: 'dataMin',
        max: 'dataMax'
      }
    ],
    yAxis: [
      {
        scale: true,
        splitArea: {
          show: true,
          areaStyle: {
            color: props.isDarkMode ?
              ['rgba(30, 41, 59, 0.4)', 'rgba(30, 41, 59, 0.2)'] :
              ['rgba(250, 250, 250, 0.5)', 'rgba(240, 240, 240, 0.5)']
          }
        },
        axisLine: {
          lineStyle: {
            color: props.isDarkMode ? '#666' : '#333'
          }
        },
        axisLabel: {
          color: props.isDarkMode ? '#ccc' : '#333'
        },
        splitLine: {
          lineStyle: {
            color: props.isDarkMode ? 'rgba(100, 100, 100, 0.3)' : 'rgba(200, 200, 200, 0.3)'
          }
        }
      },
      {
        scale: true,
        gridIndex: 1,
        splitNumber: 2,
        axisLabel: { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false }
      }
    ],
    dataZoom: [
      {
        type: 'inside',
        xAxisIndex: [0, 1],
        start: 50,
        end: 100
      },
      {
        show: true,
        xAxisIndex: [0, 1],
        type: 'slider',
        top: '85%',
        start: 50,
        end: 100,
        borderColor: props.isDarkMode ? '#666' : '#ccc',
        textStyle: {
          color: props.isDarkMode ? '#ccc' : '#333'
        },
        handleStyle: {
          color: props.isDarkMode ? '#aaa' : '#666',
          borderColor: props.isDarkMode ? '#aaa' : '#666'
        },
        fillerColor: props.isDarkMode ? 'rgba(120, 120, 120, 0.2)' : 'rgba(200, 200, 200, 0.2)'
      }
    ],
    series: [
      {
        name: 'K线',
        type: 'candlestick',
        data: data.values,
        itemStyle: {
          color: upColor,
          color0: downColor,
          borderColor: upColor,
          borderColor0: downColor
        },
        emphasis: {
          itemStyle: {
            color: upColor,
            color0: downColor,
            borderColor: upColor,
            borderColor0: downColor,
            borderWidth: 2
          }
        },
        // 确保在亮暗模式下brush选择效果一致
        select: {
          disabled: false, // 启用选择
          itemStyle: {
            // 保持与普通状态相同的颜色，但增加边框宽度以突出显示
            color: upColor,
            color0: downColor,
            borderColor: upColor,
            borderColor0: downColor,
            borderWidth: 2,
            shadowBlur: 5,
            shadowColor: props.isDarkMode ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.3)'
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
            fontSize: 12,
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
        name: 'MA5',
        type: 'line',
        data: calculateMA(5, data),
        smooth: true,
        showSymbol: false, // 不显示数据点标记
        lineStyle: {
          color: props.isDarkMode ? '#b27b7b' : getThemeColor('--chart-1'), // 深色模式下使用较亮的莫兰迪红色
          opacity: 0.6, // 增加透明度
          width: 1.5 // 减小线宽
        },
        emphasis: {
          // 鼠标悬停时的样式
          lineStyle: {
            color: props.isDarkMode ? '#b27b7b' : getThemeColor('--chart-1'),
            opacity: 0.9, // 悬停时增加不透明度
            width: 2 // 悬停时增加线宽
          }
        }
      },
      {
        name: 'MA10',
        type: 'line',
        data: calculateMA(10, data),
        smooth: true,
        showSymbol: false, // 不显示数据点标记
        lineStyle: {
          color: props.isDarkMode ? '#6b8cad' : getThemeColor('--chart-2'), // 深色模式下使用较亮的莫兰迪蓝色
          opacity: 0.6, // 增加透明度
          width: 1.5 // 减小线宽
        },
        emphasis: {
          // 鼠标悬停时的样式
          lineStyle: {
            color: props.isDarkMode ? '#6b8cad' : getThemeColor('--chart-2'),
            opacity: 0.9, // 悬停时增加不透明度
            width: 2 // 悬停时增加线宽
          }
        }
      },
      {
        name: 'MA20',
        type: 'line',
        data: calculateMA(20, data),
        smooth: true,
        showSymbol: false, // 不显示数据点标记
        lineStyle: {
          color: props.isDarkMode ? '#a89064' : getThemeColor('--chart-3'), // 深色模式下使用莫兰迪黄褐色
          opacity: 0.6, // 增加透明度
          width: 1.5 // 减小线宽
        },
        emphasis: {
          // 鼠标悬停时的样式
          lineStyle: {
            color: props.isDarkMode ? '#a89064' : getThemeColor('--chart-3'),
            opacity: 0.9, // 悬停时增加不透明度
            width: 2 // 悬停时增加线宽
          }
        }
      },
      {
        name: 'MA30',
        type: 'line',
        data: calculateMA(30, data),
        smooth: true,
        showSymbol: false, // 不显示数据点标记
        lineStyle: {
          color: props.isDarkMode ? '#8a8a99' : getThemeColor('--chart-5'), // 深色模式下使用莫兰迪灰紫色
          opacity: 0.6, // 增加透明度
          width: 1.5 // 减小线宽
        },
        emphasis: {
          // 鼠标悬停时的样式
          lineStyle: {
            color: props.isDarkMode ? '#8a8a99' : getThemeColor('--chart-5'),
            opacity: 0.9, // 悬停时增加不透明度
            width: 2 // 悬停时增加线宽
          }
        }
      },
      {
        name: '成交量',
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: data.volumes,
        itemStyle: {
          color: function (params) {
            return params.data[2] > 0 ? volumeDownColor : volumeUpColor;
          }
        }
      }
    ]
  };

  chartInstance.setOption(option);
};

// 调整图表大小
const resizeChart = () => {
  if (chartInstance) {
    chartInstance.resize();
  }
};

// 监听弹窗可见性变化
watch(() => props.visible, (newVal, oldVal) => {
  console.log('弹窗可见性变化:', newVal);
  console.log('K线数据长度:', props.klineData?.length);
  console.log('当前主题模式:', props.isDarkMode ? '暗色' : '亮色');

  if (newVal) {
    // 弹窗显示时初始化图表，使用setTimeout确保DOM已完全渲染
    setTimeout(() => {
      nextTick(() => {
        console.log('nextTick 执行, chartRef 存在:', !!chartRef.value);
        // 每次显示弹窗时都重新初始化图表
        initChart();
      });
    }, 100);
  }
});

// 监听数据变化
watch(() => props.klineData, () => {
  if (chartInstance && props.visible) {
    setOptions();
  }
}, { deep: true });

// 监听暗色模式变化
watch(() => props.isDarkMode, (newVal) => {
  if (chartInstance) {
    // 销毁旧图表
    chartInstance.dispose();
    // 重新初始化图表
    nextTick(() => {
      initChart();
    });
  }
});

// 监听弹窗关闭，清理图表实例
watch(() => props.visible, (newVal, oldVal) => {
  if (!newVal && oldVal) {
    // 弹窗关闭时，延迟销毁图表实例
    setTimeout(() => {
      if (chartInstance) {
        console.log('销毁图表实例');
        chartInstance.dispose();
        chartInstance = null;
      }
    }, 300);
  }
});

// 组件卸载时清理
onUnmounted(() => {
  window.removeEventListener('resize', resizeChart);
  if (chartInstance) {
    chartInstance.dispose();
    chartInstance = null;
  }
});
</script>

<style scoped>
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
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
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
