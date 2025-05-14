<template>
  <div class="min-h-screen bg-background text-foreground">
    <!-- 参数帮助管理器 -->
    <ParameterHelpManager ref="parameterHelpManager" />

    <!-- 顶部导航栏 -->
    <header class="bg-card border-b border-border p-4 flex justify-between items-center sticky top-0 z-30">
      <div class="flex items-center">
        <img src="/gundam-logo.svg" alt="Gundam Logo" class="w-8 h-8 mr-2" />
        <h1 class="text-xl font-semibold hidden sm:block">股票平台期扫描工具</h1>
        <h1 class="text-xl font-semibold sm:hidden">平台期扫描</h1>
      </div>

      <div class="flex items-center space-x-2 sm:space-x-3">
        <!-- 案例管理入口 -->
        <button @click="showCaseManager = true"
          class="flex items-center justify-center px-2 sm:px-3 py-1.5 sm:py-2 rounded-md bg-gundam-blue text-white hover:bg-gundam-blue/80 transition-colors">
          <i class="fas fa-book mr-1 sm:mr-2"></i>
          <span class="hidden sm:inline">案例管理</span>
        </button>

        <!-- 主题切换 -->
        <ThemeToggle />
      </div>
    </header>

    <!-- 完整K线图弹窗 -->
    <FullKlineChart v-model:visible="showFullChart"
      :title="selectedStock ? `${selectedStock.name} (${selectedStock.code})` : '股票详情'"
      :klineData="selectedStock ? selectedStock.kline_data : []"
      :markLines="selectedStock ? selectedStock.markLines : []"
      :supportLevels="selectedStock ? selectedStock.supportLevels : []"
      :resistanceLevels="selectedStock ? selectedStock.resistanceLevels : []" :isDarkMode="isDarkMode" />

    <!-- 案例管理弹窗 -->
    <div v-if="showCaseManager"
      class="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center overflow-auto">
      <div class="bg-background rounded-lg shadow-xl w-full max-w-6xl h-[90vh] flex flex-col">
        <div class="flex-1 overflow-auto">
          <CaseManager @close="showCaseManager = false" />
        </div>
      </div>
    </div>

    <!-- 主内容区 -->
    <main class="p-4 sm:p-6 md:p-8">
      <div class="max-w-6xl mx-auto">

        <!-- 参数配置卡片 -->
        <div class="card p-4 sm:p-6 mb-6">
          <h2 class="text-lg font-semibold mb-4 flex items-center">
            <i class="fas fa-sliders-h mr-2 text-primary"></i>
            扫描参数配置
          </h2>
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4 mb-4">
            <!-- 基本参数 -->
            <div>
              <ParameterLabel for-id="windows" parameter-id="windows" @show-tutorial="showParameterTutorial">
                窗口期设置
              </ParameterLabel>
              <div class="flex flex-col space-y-2">
                <!-- 预设选项 -->
                <div class="flex flex-wrap gap-2">
                  <button v-for="preset in windowPresets" :key="preset.name" @click="selectWindowPreset(preset.value)"
                    :class="[
                      'px-2 py-1 text-xs rounded-md transition-colors',
                      config.windowsInput === preset.value
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted hover:bg-muted/80 text-muted-foreground'
                    ]">
                    {{ preset.name }}
                  </button>
                  <button @click="showCustomWindowInput = true" :class="[
                    'px-2 py-1 text-xs rounded-md transition-colors',
                    !isUsingPreset
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted hover:bg-muted/80 text-muted-foreground'
                  ]">
                    自定义
                  </button>
                </div>

                <!-- 自定义输入 -->
                <div v-if="showCustomWindowInput" class="flex items-center space-x-2">
                  <input v-model="config.windowsInput" class="input flex-grow" id="windows" type="text"
                    placeholder="例如: 30,60,90">
                  <button @click="validateCustomWindows" class="btn btn-primary text-xs py-1 px-2">
                    确认
                  </button>
                </div>

                <!-- 当前选择的窗口期显示 -->
                <div v-if="!showCustomWindowInput" class="text-xs text-muted-foreground">
                  当前窗口期:
                  <span v-for="(window, index) in parsedWindows" :key="window" class="font-medium">
                    {{ window }}天{{ index < parsedWindows.length - 1 ? '、' : '' }} </span>
                </div>
              </div>
            </div>
            <div>
              <ParameterLabel for-id="expectedCount" parameter-id="expected_count"
                @show-tutorial="showParameterTutorial">
                期望股票数量
              </ParameterLabel>
              <input v-model.number="config.expected_count" class="input" id="expectedCount" type="number" min="1"
                max="100" placeholder="例如: 10">
            </div>
            <div class="flex flex-wrap items-end gap-4">
              <label for="useVolumeAnalysis"
                class="flex items-center space-x-2 cursor-pointer hover:opacity-80 transition-opacity p-1 rounded-md hover:bg-muted/30">
                <input type="checkbox" v-model="config.use_volume_analysis" id="useVolumeAnalysis" class="checkbox">
                <ParameterLabel for-id="useVolumeAnalysis" parameter-id="use_volume_analysis"
                  @show-tutorial="showParameterTutorial">
                  <span class="text-sm font-medium">启用成交量分析</span>
                </ParameterLabel>
              </label>
              <label for="useBreakthroughPrediction"
                class="flex items-center space-x-2 cursor-pointer hover:opacity-80 transition-opacity p-1 rounded-md hover:bg-muted/30">
                <input type="checkbox" v-model="config.use_breakthrough_prediction" id="useBreakthroughPrediction"
                  class="checkbox">
                <ParameterLabel for-id="useBreakthroughPrediction" parameter-id="use_breakthrough_prediction"
                  @show-tutorial="showParameterTutorial">
                  <span class="text-sm font-medium">启用突破前兆识别</span>
                </ParameterLabel>
              </label>
              <label for="useBreakthroughConfirmation"
                class="flex items-center space-x-2 cursor-pointer hover:opacity-80 transition-opacity p-1 rounded-md hover:bg-muted/30">
                <input type="checkbox" v-model="config.use_breakthrough_confirmation" id="useBreakthroughConfirmation"
                  class="checkbox">
                <ParameterLabel for-id="useBreakthroughConfirmation" parameter-id="use_breakthrough_confirmation"
                  @show-tutorial="showParameterTutorial">
                  <span class="text-sm font-medium">启用突破确认</span>
                </ParameterLabel>
              </label>
              <label for="useLowPosition"
                class="flex items-center space-x-2 cursor-pointer hover:opacity-80 transition-opacity p-1 rounded-md hover:bg-muted/30">
                <input type="checkbox" v-model="config.use_low_position" id="useLowPosition" class="checkbox">
                <ParameterLabel for-id="useLowPosition" parameter-id="use_low_position"
                  @show-tutorial="showParameterTutorial">
                  <span class="text-sm font-medium">启用低位判断</span>
                </ParameterLabel>
              </label>
              <label for="useWindowWeights"
                class="flex items-center space-x-2 cursor-pointer hover:opacity-80 transition-opacity p-1 rounded-md hover:bg-muted/30">
                <input type="checkbox" v-model="config.use_window_weights" id="useWindowWeights" class="checkbox">
                <ParameterLabel for-id="useWindowWeights" parameter-id="use_window_weights"
                  @show-tutorial="showParameterTutorial">
                  <span class="text-sm font-medium">启用窗口权重</span>
                </ParameterLabel>
              </label>
            </div>
          </div>

          <!-- 价格参数 -->
          <div class="mb-4">
            <h3 class="text-sm font-medium mb-2 flex items-center">
              <i class="fas fa-chart-line mr-1 text-primary"></i>
              价格参数
            </h3>
            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4">
              <div>
                <ParameterLabel for-id="boxThreshold" parameter-id="box_threshold"
                  @show-tutorial="showParameterTutorial">
                  振幅阈值 (%)
                </ParameterLabel>
                <input v-model.number="config.box_threshold" class="input" id="boxThreshold" type="number" step="0.01"
                  placeholder="例如: 0.3">
              </div>
              <div>
                <ParameterLabel for-id="maDiffThreshold" parameter-id="ma_diff_threshold"
                  @show-tutorial="showParameterTutorial">
                  均线粘合度 (%)
                </ParameterLabel>
                <input v-model.number="config.ma_diff_threshold" class="input" id="maDiffThreshold" type="number"
                  step="0.005" placeholder="例如: 0.03">
              </div>
              <div>
                <ParameterLabel for-id="volatilityThreshold" parameter-id="volatility_threshold"
                  @show-tutorial="showParameterTutorial">
                  波动率阈值 (%)
                </ParameterLabel>
                <input v-model.number="config.volatility_threshold" class="input" id="volatilityThreshold" type="number"
                  step="0.005" placeholder="例如: 0.03">
              </div>
            </div>
          </div>

          <!-- 成交量参数 -->
          <div class="mb-4" v-if="config.use_volume_analysis">
            <h3 class="text-sm font-medium mb-2 flex items-center">
              <i class="fas fa-chart-bar mr-1 text-primary"></i>
              成交量参数
            </h3>
            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4">
              <div>
                <ParameterLabel for-id="volumeChangeThreshold" parameter-id="volume_change_threshold"
                  @show-tutorial="showParameterTutorial">
                  成交量变化阈值
                </ParameterLabel>
                <input v-model.number="config.volume_change_threshold" class="input" id="volumeChangeThreshold"
                  type="number" step="0.05" placeholder="例如: 0.8">
                <p class="text-xs text-muted-foreground mt-1">平台期内成交量变化的最大比例</p>
              </div>
              <div>
                <ParameterLabel for-id="volumeStabilityThreshold" parameter-id="volume_stability_threshold"
                  @show-tutorial="showParameterTutorial">
                  成交量稳定性阈值
                </ParameterLabel>
                <input v-model.number="config.volume_stability_threshold" class="input" id="volumeStabilityThreshold"
                  type="number" step="0.05" placeholder="例如: 0.5">
                <p class="text-xs text-muted-foreground mt-1">平台期内成交量波动的最大程度</p>
              </div>
              <div>
                <ParameterLabel for-id="volumeIncreaseThreshold" parameter-id="volume_increase_threshold"
                  @show-tutorial="showParameterTutorial">
                  成交量突破阈值
                </ParameterLabel>
                <input v-model.number="config.volume_increase_threshold" class="input" id="volumeIncreaseThreshold"
                  type="number" step="0.1" placeholder="例如: 1.5">
                <p class="text-xs text-muted-foreground mt-1">识别为突破的最小成交量放大倍数</p>
              </div>
            </div>
          </div>

          <!-- 位置参数 -->
          <div class="mb-4" v-if="config.use_low_position">
            <h3 class="text-sm font-medium mb-2 flex items-center">
              <i class="fas fa-map-marker-alt mr-1 text-primary"></i>
              位置参数
            </h3>
            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4">
              <div>
                <ParameterLabel for-id="highPointLookbackDays" parameter-id="high_point_lookback_days"
                  @show-tutorial="showParameterTutorial">
                  高点查找时间范围 (天)
                </ParameterLabel>
                <input v-model.number="config.high_point_lookback_days" class="input" id="highPointLookbackDays"
                  type="number" step="1" min="30" placeholder="例如: 365">
              </div>
              <div>
                <ParameterLabel for-id="declinePeriodDays" parameter-id="decline_period_days"
                  @show-tutorial="showParameterTutorial">
                  下跌时间范围 (天)
                </ParameterLabel>
                <input v-model.number="config.decline_period_days" class="input" id="declinePeriodDays" type="number"
                  step="1" min="30" placeholder="例如: 180">
              </div>
              <div>
                <ParameterLabel for-id="declineThreshold" parameter-id="decline_threshold"
                  @show-tutorial="showParameterTutorial">
                  下跌幅度阈值
                </ParameterLabel>
                <input v-model.number="config.decline_threshold" class="input" id="declineThreshold" type="number"
                  step="0.05" min="0.1" max="0.9" placeholder="例如: 0.3">
                <p class="text-xs text-muted-foreground mt-1">从高点下跌的最小百分比 (0.3 = 30%)</p>
              </div>
            </div>
          </div>

          <!-- 快速下跌判断参数 -->
          <div class="mb-4" v-if="config.use_low_position">
            <h3 class="text-sm font-medium mb-2 flex items-center">
              <i class="fas fa-bolt mr-1 text-primary"></i>
              快速下跌判断
            </h3>
            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4">
              <div>
                <div class="flex items-center justify-between">
                  <ParameterLabel for-id="useRapidDeclineDetection" parameter-id="use_rapid_decline_detection"
                    @show-tutorial="showParameterTutorial">
                    启用快速下跌判断
                  </ParameterLabel>
                  <label for="useRapidDeclineDetection" class="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" v-model="config.use_rapid_decline_detection" id="useRapidDeclineDetection"
                      class="sr-only peer">
                    <div
                      class="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-primary">
                    </div>
                  </label>
                </div>
                <p class="text-xs text-muted-foreground mt-1">识别类似安记食品的快速下跌后形成平台期的股票</p>
              </div>
              <!-- 只有在启用快速下跌判断时才显示这些参数 -->
              <div v-if="config.use_rapid_decline_detection">
                <ParameterLabel for-id="rapidDeclineDays" parameter-id="rapid_decline_days"
                  @show-tutorial="showParameterTutorial">
                  快速下跌时间窗口 (天)
                </ParameterLabel>
                <input v-model.number="config.rapid_decline_days" class="input" id="rapidDeclineDays" type="number"
                  step="1" min="10" max="60" placeholder="例如: 30">
                <p class="text-xs text-muted-foreground mt-1">定义快速下跌的时间窗口</p>
              </div>
              <div v-if="config.use_rapid_decline_detection">
                <ParameterLabel for-id="rapidDeclineThreshold" parameter-id="rapid_decline_threshold"
                  @show-tutorial="showParameterTutorial">
                  快速下跌幅度阈值
                </ParameterLabel>
                <input v-model.number="config.rapid_decline_threshold" class="input" id="rapidDeclineThreshold"
                  type="number" step="0.05" min="0.05" max="0.5" placeholder="例如: 0.15">
                <p class="text-xs text-muted-foreground mt-1">快速下跌的最小百分比 (0.15 = 15%)</p>
              </div>
            </div>
          </div>

          <!-- 突破确认参数 -->
          <div class="mb-4" v-if="config.use_breakthrough_confirmation">
            <h3 class="text-sm font-medium mb-2 flex items-center">
              <i class="fas fa-check-circle mr-1 text-primary"></i>
              突破确认参数
            </h3>
            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4">
              <div>
                <ParameterLabel for-id="breakthroughConfirmationDays" parameter-id="breakthrough_confirmation_days"
                  @show-tutorial="showParameterTutorial">
                  确认天数
                </ParameterLabel>
                <input v-model.number="config.breakthrough_confirmation_days" class="input"
                  id="breakthroughConfirmationDays" type="number" step="1" min="1" max="5" placeholder="例如: 1">
                <p class="text-xs text-muted-foreground mt-1">突破后需要多少天确认</p>
              </div>
            </div>
          </div>

          <!-- 箱体检测参数 -->
          <div class="mb-4">
            <h3 class="text-sm font-medium mb-2 flex items-center">
              <i class="fas fa-cube mr-1 text-primary"></i>
              箱体检测参数
            </h3>
            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4">
              <div>
                <div class="flex items-center justify-between">
                  <ParameterLabel for-id="useBoxDetection" parameter-id="use_box_detection"
                    @show-tutorial="showParameterTutorial">
                    启用箱体检测
                  </ParameterLabel>
                  <label for="useBoxDetection" class="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" v-model="config.use_box_detection" id="useBoxDetection" class="sr-only peer">
                    <div
                      class="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-primary">
                    </div>
                  </label>
                </div>
                <p class="text-xs text-muted-foreground mt-1">启用更精确的箱体形态检测</p>
              </div>
              <div v-if="config.use_box_detection">
                <ParameterLabel for-id="boxQualityThreshold" parameter-id="box_quality_threshold"
                  @show-tutorial="showParameterTutorial">
                  箱体质量阈值
                </ParameterLabel>
                <input v-model.number="config.box_quality_threshold" class="input" id="boxQualityThreshold"
                  type="number" step="0.05" min="0.1" max="0.9" placeholder="例如: 0.6">
                <p class="text-xs text-muted-foreground mt-1">箱体形态的最低质量要求 (0.6 = 60%)</p>
              </div>
            </div>
          </div>

          <!-- 基本面筛选参数 -->
          <div class="mb-4">
            <h3 class="text-sm font-medium mb-2 flex items-center">
              <i class="fas fa-chart-pie mr-1 text-secondary"></i>
              基本面筛选参数
            </h3>
            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4">
              <div>
                <div class="flex items-center justify-between">
                  <ParameterLabel for-id="useFundamentalFilter" parameter-id="use_fundamental_filter"
                    @show-tutorial="showParameterTutorial">
                    启用基本面筛选
                  </ParameterLabel>
                  <label for="useFundamentalFilter" class="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" v-model="config.use_fundamental_filter" id="useFundamentalFilter"
                      class="sr-only peer">
                    <div
                      class="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-secondary">
                    </div>
                  </label>
                </div>
                <p class="text-xs text-muted-foreground mt-1">启用基于财务指标的基本面筛选</p>
              </div>

              <div v-if="config.use_fundamental_filter">
                <ParameterLabel for-id="revenueGrowthPercentile" parameter-id="revenue_growth_percentile"
                  @show-tutorial="showParameterTutorial">
                  营收增长率百分位
                </ParameterLabel>
                <input v-model.number="config.revenue_growth_percentile" class="input" id="revenueGrowthPercentile"
                  type="number" step="0.05" min="0.1" max="0.9" placeholder="例如: 0.3">
                <p class="text-xs text-muted-foreground mt-1">要求位于行业前X%（0.3 = 前30%）</p>
              </div>

              <div v-if="config.use_fundamental_filter">
                <ParameterLabel for-id="profitGrowthPercentile" parameter-id="profit_growth_percentile"
                  @show-tutorial="showParameterTutorial">
                  净利润增长率百分位
                </ParameterLabel>
                <input v-model.number="config.profit_growth_percentile" class="input" id="profitGrowthPercentile"
                  type="number" step="0.05" min="0.1" max="0.9" placeholder="例如: 0.3">
                <p class="text-xs text-muted-foreground mt-1">要求位于行业前X%（0.3 = 前30%）</p>
              </div>

              <div v-if="config.use_fundamental_filter">
                <ParameterLabel for-id="roePercentile" parameter-id="roe_percentile"
                  @show-tutorial="showParameterTutorial">
                  ROE百分位
                </ParameterLabel>
                <input v-model.number="config.roe_percentile" class="input" id="roePercentile" type="number" step="0.05"
                  min="0.1" max="0.9" placeholder="例如: 0.3">
                <p class="text-xs text-muted-foreground mt-1">要求位于行业前X%（0.3 = 前30%）</p>
              </div>

              <div v-if="config.use_fundamental_filter">
                <ParameterLabel for-id="liabilityPercentile" parameter-id="liability_percentile"
                  @show-tutorial="showParameterTutorial">
                  资产负债率百分位
                </ParameterLabel>
                <input v-model.number="config.liability_percentile" class="input" id="liabilityPercentile" type="number"
                  step="0.05" min="0.1" max="0.9" placeholder="例如: 0.3">
                <p class="text-xs text-muted-foreground mt-1">要求位于行业后X%（0.3 = 后30%）</p>
              </div>

              <div v-if="config.use_fundamental_filter">
                <ParameterLabel for-id="pePercentile" parameter-id="pe_percentile"
                  @show-tutorial="showParameterTutorial">
                  PE百分位
                </ParameterLabel>
                <input v-model.number="config.pe_percentile" class="input" id="pePercentile" type="number" step="0.05"
                  min="0.1" max="0.9" placeholder="例如: 0.7">
                <p class="text-xs text-muted-foreground mt-1">要求不在行业前X%最高估值（0.7 = 后70%）</p>
              </div>

              <div v-if="config.use_fundamental_filter">
                <ParameterLabel for-id="pbPercentile" parameter-id="pb_percentile"
                  @show-tutorial="showParameterTutorial">
                  PB百分位
                </ParameterLabel>
                <input v-model.number="config.pb_percentile" class="input" id="pbPercentile" type="number" step="0.05"
                  min="0.1" max="0.9" placeholder="例如: 0.7">
                <p class="text-xs text-muted-foreground mt-1">要求不在行业前X%最高估值（0.7 = 后70%）</p>
              </div>

              <div v-if="config.use_fundamental_filter">
                <ParameterLabel for-id="fundamentalYearsToCheck" parameter-id="fundamental_years_to_check"
                  @show-tutorial="showParameterTutorial">
                  检查年数
                </ParameterLabel>
                <input v-model.number="config.fundamental_years_to_check" class="input" id="fundamentalYearsToCheck"
                  type="number" step="1" min="1" max="5" placeholder="例如: 3">
                <p class="text-xs text-muted-foreground mt-1">连续增长的年数要求（默认3年）</p>
              </div>
            </div>
          </div>

          <!-- 窗口权重设置 -->
          <div class="mb-4" v-if="config.use_window_weights">
            <h3 class="text-sm font-medium mb-2 flex items-center">
              <i class="fas fa-balance-scale mr-1 text-primary"></i>
              窗口权重设置
            </h3>
            <div class="bg-muted/30 p-3 rounded-md mb-3">
              <p class="text-xs text-muted-foreground mb-2">
                为不同的窗口期分配权重，权重总和将自动归一化。权重越高，该窗口期的分析结果对最终评分的影响越大。
              </p>
              <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4">
                <div v-for="window in parsedWindows" :key="window" class="flex items-center space-x-2">
                  <label class="text-sm font-medium whitespace-nowrap">{{ window }}天:</label>
                  <input v-model.number="windowWeights[window]" type="range" min="0" max="10" step="1" class="flex-grow"
                    @input="updateWindowWeights(window, $event.target.value)">
                  <span class="text-sm">{{ windowWeights[window] || 0 }}</span>
                </div>
              </div>
            </div>
          </div>
          <div class="flex items-center justify-center">
            <button @click="fetchPlatformStocks" :disabled="loading" class="btn btn-primary py-2 px-6" type="button">
              <i class="fas fa-search mr-2" v-if="!loading"></i>
              <i class="fas fa-spinner fa-spin mr-2" v-if="loading"></i>
              {{ loading ? '扫描中...' : '开始扫描' }}
            </button>
          </div>
        </div>

        <!-- 任务进度组件 -->
        <transition name="fade">
          <TaskProgress v-if="taskStatus && taskStatus !== 'completed'" :status="taskStatus" :progress="taskProgress"
            :message="taskMessage" :error="taskError" @retry="fetchPlatformStocks" />
        </transition>

        <!-- 错误提示 -->
        <transition name="fade">
          <div v-if="error && !taskStatus"
            class="bg-destructive/10 border border-destructive text-destructive px-4 py-3 rounded-md my-5" role="alert">
            <div class="flex items-center">
              <i class="fas fa-exclamation-circle mr-2"></i>
              <strong class="font-bold">发生错误:</strong>
              <span class="ml-2"> {{ error }}</span>
            </div>
          </div>
        </transition>

        <!-- 扫描结果 -->
        <transition name="slide-up">
          <div v-if="(!loading || taskStatus === 'completed') && platformStocks.length > 0"
            class="card overflow-hidden">
            <div class="p-4 border-b border-border flex justify-between items-center">
              <h2 class="text-lg font-semibold flex items-center">
                <i class="fas fa-list-ul mr-2 text-primary"></i>
                扫描结果
                <span class="ml-2 px-2 py-0.5 rounded-full text-xs bg-primary opacity-20 text-primary">{{
                  platformStocks.length
                }} 只</span>
              </h2>
              <div class="flex space-x-2">
                <button class="btn btn-secondary text-xs py-1 px-2">
                  <i class="fas fa-download mr-1"></i> 导出
                </button>
                <button class="btn btn-secondary text-xs py-1 px-2">
                  <i class="fas fa-filter mr-1"></i> 筛选
                </button>
              </div>
            </div>
            <!-- 桌面端表格视图 -->
            <div class="hidden md:block overflow-x-auto">
              <table class="w-full">
                <thead class="bg-muted/50">
                  <tr>
                    <th scope="col"
                      class="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider w-[100px]">
                      <div class="flex items-center">
                        <i class="fas fa-hashtag mr-1"></i> 代码
                      </div>
                    </th>
                    <th scope="col"
                      class="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider w-[120px]">
                      <div class="flex items-center">
                        <i class="fas fa-font mr-1"></i> 名称
                      </div>
                    </th>
                    <th scope="col"
                      class="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider w-[100px]">
                      <div class="flex items-center">
                        <i class="fas fa-tag mr-1"></i> 行业
                      </div>
                    </th>
                    <th scope="col"
                      class="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                      <div class="flex items-center">
                        <i class="fas fa-check-circle mr-1"></i> 选择理由
                      </div>
                    </th>
                    <th scope="col"
                      class="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider w-[400px]">
                      <div class="flex items-center">
                        <i class="fas fa-chart-line mr-1"></i> 近期K线
                      </div>
                    </th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-border">
                  <tr v-for="stock in platformStocks" :key="stock.code">
                    <td class="px-4 py-3 whitespace-nowrap text-sm font-medium">{{ stock.code }}</td>
                    <td class="px-4 py-3 whitespace-nowrap text-sm">{{ stock.name }}</td>
                    <td class="px-4 py-3 whitespace-nowrap text-sm">
                      <span class="px-2 py-0.5 rounded-full text-xs bg-primary/20 text-primary">
                        {{ stock.industry || '未知行业' }}
                      </span>
                    </td>
                    <td class="px-4 py-3 text-sm">
                      <!-- 选择理由标题栏（可点击） -->
                      <div @click="toggleReasonExpand(stock.code)"
                        class="flex items-center justify-between cursor-pointer p-1.5 rounded hover:bg-muted/50 transition-colors">
                        <div class="flex items-center">
                          <i class="fas fa-info-circle text-primary mr-1.5"></i>
                          <span class="font-medium">选择理由</span>
                          <span v-if="Object.keys(stock.selection_reasons || {}).length > 0"
                            class="ml-1.5 text-xs px-1.5 py-0.5 rounded-full bg-primary/10 text-primary">
                            {{ Object.keys(stock.selection_reasons || {}).length }}
                          </span>
                        </div>
                        <i :class="[
                          'fas transition-transform duration-300',
                          expandedReasons[stock.code] ? 'fa-chevron-up' : 'fa-chevron-down'
                        ]"></i>
                      </div>

                      <!-- 选择理由详情（可折叠） -->
                      <div :id="`reason-${stock.code.replace(/\./g, '_').replace(/[^\w-]/g, '')}`" :class="[
                        'overflow-hidden transition-all duration-300 mt-1',
                        expandedReasons[stock.code] === undefined || expandedReasons[stock.code] ? 'h-auto opacity-100' : 'h-0 opacity-0'
                      ]">
                        <div v-if="Object.keys(stock.selection_reasons || {}).length > 0"
                          class="p-2 bg-muted/10 rounded">
                          <div v-for="(reason, window) in stock.selection_reasons" :key="window"
                            class="mb-1 text-xs text-muted-foreground">
                            <span class="font-medium text-primary">{{ window }}天:</span>
                            {{ reason }}
                          </div>

                          <!-- 成交量分析结果 -->
                          <div v-if="stock.volume_analysis && stock.volume_analysis[window]"
                            class="mt-2 border-t border-border pt-1">
                            <div class="text-xs font-medium text-primary">成交量分析:</div>
                            <div v-if="stock.volume_analysis[window].has_consolidation_volume"
                              class="text-xs text-muted-foreground flex items-center">
                              <i class="fas fa-check-circle text-green-500 mr-1"></i>
                              成交量萎缩
                            </div>
                            <div v-if="stock.volume_analysis[window].has_breakthrough"
                              class="text-xs text-muted-foreground flex items-center">
                              <i class="fas fa-arrow-circle-up text-primary mr-1"></i>
                              成交量突破 ({{
                                stock.volume_analysis[window].breakthrough_details.volume_increase_ratio.toFixed(2) }}倍)
                            </div>
                          </div>

                          <!-- 突破前兆识别结果 -->
                          <div
                            v-if="stock.breakthrough_prediction && stock.breakthrough_prediction.has_breakthrough_signal"
                            class="mt-2 border-t border-border pt-1">
                            <div class="text-xs font-medium text-primary">突破前兆:</div>
                            <div class="text-xs text-muted-foreground">
                              <span class="flex items-center">
                                <i class="fas fa-bolt text-amber-500 mr-1"></i>
                                {{ stock.breakthrough_prediction.signal_count }}个技术指标显示突破信号
                              </span>
                              <div class="mt-1 flex flex-wrap gap-1">
                                <span v-for="(hasSignal, indicator) in stock.breakthrough_prediction.signals"
                                  :key="indicator" :class="[
                                    'px-1.5 py-0.5 rounded text-xs',
                                    hasSignal ? 'bg-amber-500/20 text-amber-700 dark:text-amber-400' : 'bg-muted text-muted-foreground'
                                  ]">
                                  {{ indicator }}
                                  <i v-if="hasSignal" class="fas fa-check-circle text-xs ml-0.5"></i>
                                </span>
                              </div>
                            </div>
                          </div>

                          <!-- 窗口权重得分 -->
                          <div v-if="stock.weighted_score !== undefined" class="mt-2 border-t border-border pt-1">
                            <div class="text-xs font-medium text-primary">加权得分:</div>
                            <div class="text-xs text-muted-foreground flex items-center">
                              <i class="fas fa-star text-yellow-500 mr-1"></i>
                              <div class="flex items-center">
                                <span class="font-medium">{{ stock.weighted_score.toFixed(2) }}</span>
                                <div class="ml-2 w-24 bg-muted rounded-full h-1.5 overflow-hidden">
                                  <div class="bg-yellow-500 h-full rounded-full"
                                    :style="{ width: `${Math.min(100, stock.weighted_score * 100)}%` }"></div>
                                </div>
                              </div>
                            </div>
                            <div v-if="stock.weight_details && stock.weight_details.window_scores" class="mt-1">
                              <div class="text-xs text-muted-foreground flex flex-wrap gap-1 mt-1">
                                <span v-for="(score, window) in stock.weight_details.window_scores" :key="window"
                                  class="px-1.5 py-0.5 rounded text-xs bg-muted/50">
                                  {{ window }}天: {{ score.toFixed(2) }}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div v-else class="text-xs text-muted-foreground italic p-2">
                          无选择理由
                        </div>
                      </div>
                    </td>
                    <td class="px-4 py-3">
                      <!-- 缩略图K线图容器 -->
                      <div class="relative group w-full">
                        <!-- K线图 -->
                        <KlineChart :klineData="stock.kline_data" height="150px" width="100%"
                          :title="`${stock.name} (${stock.code})`" :isDarkMode="isDarkMode"
                          :markLines="generateMarkLines(stock)" :supportLevels="getSupportLevels(stock)"
                          :resistanceLevels="getResistanceLevels(stock)" class="rounded-md overflow-hidden w-full" />

                        <!-- 操作按钮组 -->
                        <div
                          class="absolute top-3 right-3 flex space-x-2 opacity-0 group-hover:opacity-100 transition-all duration-300 z-10">
                          <!-- 放大按钮 -->
                          <button @click="openFullChart(stock)"
                            class="bg-gundam-blue/30 hover:bg-gundam-blue text-white rounded-md p-1.5 shadow-md backdrop-blur-sm transform hover:scale-110 transition-all"
                            title="查看完整K线图">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                              stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                              class="w-3.5 h-3.5">
                              <polyline points="15 3 21 3 21 9"></polyline>
                              <polyline points="9 21 3 21 3 15"></polyline>
                              <line x1="21" y1="3" x2="14" y2="10"></line>
                              <line x1="3" y1="21" x2="10" y2="14"></line>
                            </svg>
                          </button>

                          <!-- 导出到案例按钮 -->
                          <button @click="exportToCase(stock)"
                            class="bg-gundam-yellow/30 hover:bg-gundam-yellow text-white rounded-md p-1.5 shadow-md backdrop-blur-sm transform hover:scale-110 transition-all"
                            title="导出到案例库">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                              stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                              class="w-3.5 h-3.5">
                              <path d="M12 2v8"></path>
                              <path d="m16 6-4 4-4-4"></path>
                              <path d="M8 10H4a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-8a2 2 0 0 0-2-2h-4">
                              </path>
                            </svg>
                          </button>
                        </div>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <!-- 移动端卡片视图 -->
            <div class="md:hidden">
              <div v-for="stock in platformStocks" :key="stock.code"
                class="mb-4 border border-border rounded-lg overflow-hidden bg-card">
                <!-- 股票基本信息 -->
                <div class="p-3 border-b border-border bg-muted/20">
                  <div class="flex justify-between items-center">
                    <div>
                      <div class="font-medium">{{ stock.name }} <span class="text-muted-foreground">{{ stock.code
                      }}</span></div>
                      <div class="mt-1">
                        <span class="px-2 py-0.5 rounded-full text-xs bg-primary/20 text-primary">
                          {{ stock.industry || '未知行业' }}
                        </span>
                      </div>
                    </div>
                    <div class="flex space-x-2">
                      <button @click="openFullChart(stock)"
                        class="bg-gundam-blue/20 hover:bg-gundam-blue/30 text-gundam-blue rounded-md p-1.5 transition-colors"
                        title="查看完整K线图">
                        <i class="fas fa-expand-alt text-sm"></i>
                      </button>
                      <button @click="exportToCase(stock)"
                        class="bg-gundam-yellow/20 hover:bg-gundam-yellow/30 text-gundam-yellow rounded-md p-1.5 transition-colors"
                        title="导出到案例库">
                        <i class="fas fa-download text-sm"></i>
                      </button>
                    </div>
                  </div>
                </div>

                <!-- K线图 -->
                <div class="relative">
                  <KlineChart :klineData="stock.kline_data" height="180px" width="100%"
                    :title="`${stock.name} (${stock.code})`" :isDarkMode="isDarkMode"
                    :markLines="generateMarkLines(stock)" :supportLevels="getSupportLevels(stock)"
                    :resistanceLevels="getResistanceLevels(stock)" class="rounded-md overflow-hidden w-full" />
                </div>

                <!-- 选择理由 -->
                <div class="p-3 border-t border-border">
                  <!-- 选择理由标题栏（可点击） -->
                  <div @click="toggleReasonExpand(stock.code)"
                    class="flex items-center justify-between cursor-pointer p-1.5 rounded hover:bg-muted/50 transition-colors">
                    <div class="flex items-center">
                      <i class="fas fa-info-circle text-primary mr-1.5"></i>
                      <span class="font-medium">选择理由</span>
                      <span v-if="Object.keys(stock.selection_reasons || {}).length > 0"
                        class="ml-1.5 text-xs px-1.5 py-0.5 rounded-full bg-primary/10 text-primary">
                        {{ Object.keys(stock.selection_reasons || {}).length }}
                      </span>
                    </div>
                    <i :class="[
                      'fas transition-transform duration-300',
                      expandedReasons[stock.code] ? 'fa-chevron-up' : 'fa-chevron-down'
                    ]"></i>
                  </div>

                  <!-- 选择理由详情（可折叠） -->
                  <div :id="`reason-mobile-${stock.code.replace(/\./g, '_').replace(/[^\w-]/g, '')}`" :class="[
                    'overflow-hidden transition-all duration-300 mt-1',
                    expandedReasons[stock.code] === undefined || expandedReasons[stock.code] ? 'h-auto opacity-100' : 'h-0 opacity-0'
                  ]">
                    <div v-if="Object.keys(stock.selection_reasons || {}).length > 0" class="p-2 bg-muted/10 rounded">
                      <div v-for="(reason, window) in stock.selection_reasons" :key="window"
                        class="mb-1 text-xs text-muted-foreground">
                        <span class="font-medium text-primary">{{ window }}天:</span>
                        {{ reason }}
                      </div>

                      <!-- 成交量分析结果 -->
                      <div v-if="stock.volume_analysis && stock.volume_analysis[window]"
                        class="mt-2 border-t border-border pt-1">
                        <div class="text-xs font-medium text-primary">成交量分析:</div>
                        <div v-if="stock.volume_analysis[window].has_consolidation_volume"
                          class="text-xs text-muted-foreground flex items-center">
                          <i class="fas fa-check-circle text-green-500 mr-1"></i>
                          成交量萎缩
                        </div>
                        <div v-if="stock.volume_analysis[window].has_breakthrough"
                          class="text-xs text-muted-foreground flex items-center">
                          <i class="fas fa-arrow-circle-up text-primary mr-1"></i>
                          成交量突破 ({{
                            stock.volume_analysis[window].breakthrough_details.volume_increase_ratio.toFixed(2) }}倍)
                        </div>
                      </div>

                      <!-- 窗口权重得分 -->
                      <div v-if="stock.weighted_score !== undefined" class="mt-2 border-t border-border pt-1">
                        <div class="text-xs font-medium text-primary">加权得分:</div>
                        <div class="text-xs text-muted-foreground flex items-center">
                          <i class="fas fa-star text-yellow-500 mr-1"></i>
                          <div class="flex items-center">
                            <span class="font-medium">{{ stock.weighted_score.toFixed(2) }}</span>
                            <div class="ml-2 w-24 bg-muted rounded-full h-1.5 overflow-hidden">
                              <div class="bg-yellow-500 h-full rounded-full"
                                :style="{ width: `${Math.min(100, stock.weighted_score * 100)}%` }"></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div v-else class="text-xs text-muted-foreground italic p-2">
                      无选择理由
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </transition>

        <!-- 无结果提示 -->
        <transition name="fade">
          <div v-if="!loading && taskStatus === 'completed' && platformStocks.length === 0 && !error && hasSearched"
            class="card p-8 text-center my-5 text-muted-foreground">
            <i class="fas fa-search text-4xl mb-3 opacity-50"></i>
            <p>未找到符合当前条件的股票。</p>
            <p class="text-sm mt-2">尝试调整扫描参数后重新搜索。</p>
          </div>
        </transition>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, inject, provide, nextTick } from 'vue';
import axios from 'axios';
import KlineChart from './components/KlineChart.vue'; // 缩略图K线图组件
import FullKlineChart from './components/FullKlineChart.vue'; // 完整K线图组件
import TaskProgress from './components/TaskProgress.vue'; // 任务进度组件
import { ParameterHelpManager, ParameterLabel } from './components/parameter-help'; // 参数帮助组件
import CaseManager from './components/case-management/CaseManager.vue'; // 案例管理组件
import ThemeToggle from './components/ThemeToggle.vue'; // 主题切换组件
import { gsap } from 'gsap';

const config = ref({
  // 基本参数
  windowsInput: '80,100,120', // 标准窗口期设置
  expected_count: 10, // 期望返回的股票数量，默认为10

  // 价格参数
  box_threshold: 0.3, // 箱体阈值
  ma_diff_threshold: 0.25, // 均线粘合度阈值
  volatility_threshold: 0.4, // 波动率阈值

  // 成交量参数
  use_volume_analysis: true, // 是否启用成交量分析
  volume_change_threshold: 0.5, // 成交量变化阈值
  volume_stability_threshold: 0.5, // 成交量稳定性阈值
  volume_increase_threshold: 1.5, // 成交量突破阈值

  // 技术指标参数
  use_technical_indicators: false, // 是否启用技术指标
  use_breakthrough_prediction: false, // 是否启用突破前兆识别

  // 位置参数
  use_low_position: true, // 是否启用低位判断
  high_point_lookback_days: 365, // 高点查找时间范围
  decline_period_days: 180, // 下跌时间范围
  decline_threshold: 0.3, // 下跌阈值

  // 快速下跌判断参数
  use_rapid_decline_detection: true, // 是否启用快速下跌判断
  rapid_decline_days: 30, // 快速下跌窗口
  rapid_decline_threshold: 0.15, // 快速下跌阈值

  // 突破确认参数
  use_breakthrough_confirmation: false, // 是否启用突破确认
  breakthrough_confirmation_days: 1, // 确认天数，默认为1天，这样启用时不需要手动修改

  // 箱体检测参数
  use_box_detection: true, // 是否启用箱体检测
  box_quality_threshold: 0.3, // 箱体质量阈值

  // 基本面筛选参数
  use_fundamental_filter: false, // 是否启用基本面筛选
  revenue_growth_percentile: 0.3, // 营收增长率行业百分位
  profit_growth_percentile: 0.3, // 净利润增长率行业百分位
  roe_percentile: 0.3, // ROE行业百分位
  liability_percentile: 0.3, // 资产负债率行业百分位
  pe_percentile: 0.7, // PE行业百分位
  pb_percentile: 0.7, // PB行业百分位
  fundamental_years_to_check: 3, // 检查连续增长的年数

  // 窗口权重参数
  use_window_weights: false, // 是否使用窗口权重
  window_weights: {} // 窗口权重
});

const platformStocks = ref([]);
const loading = ref(false);
const error = ref(null);
const hasSearched = ref(false); // Track if a search has been performed
const isDarkMode = ref(false); // 暗色模式状态
const windowWeights = ref({}); // 窗口权重
const expandedReasons = ref({}); // 跟踪每个股票的选择理由是否展开

// 窗口期预设
const windowPresets = [
  { name: '标准', value: '80,100,120' },
  { name: '短期', value: '10,20,30' },
  { name: '中期', value: '30,60,90' },
  { name: '长期', value: '60,120,180' },
  { name: '混合', value: '30,60,120' }
];

// 自定义窗口期相关
const showCustomWindowInput = ref(false);
const isUsingPreset = computed(() => {
  return windowPresets.some(preset => preset.value === config.value.windowsInput);
});

// 任务状态相关
const currentTaskId = ref(null);
const taskStatus = ref(null); // 'pending', 'running', 'completed', 'failed'
const taskProgress = ref(0);
const taskMessage = ref('');
const taskError = ref(null);
const pollingInterval = ref(null);

// K线图弹窗相关状态
const showFullChart = ref(false);
const showCaseManager = ref(false);
const selectedStock = ref(null);

// 参数帮助相关
const parameterHelp = inject('parameterHelp', {
  openTutorial: (id) => {
    console.log('打开教程:', id);
  },
  closeTutorial: () => {
    console.log('关闭教程');
  },
  getTooltip: (id) => {
    console.log('获取提示:', id);
    return null;
  }
});

// 打开参数教程
const showParameterTutorial = (id) => {
  console.log('App: 显示参数教程:', id);
  // 这个方法现在可能不会被调用，因为 ParameterLabel 组件会直接调用 parameterHelp.openTutorial
  if (parameterHelp && parameterHelp.openTutorial) {
    parameterHelp.openTutorial(id);
  } else {
    console.warn('App: parameterHelp.openTutorial 未提供');
  }
};

// 初始化时检查系统偏好
onMounted(() => {
  // 检查本地存储中的主题设置
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    isDarkMode.value = true;
    document.documentElement.classList.add('dark');
  }

  // 添加页面加载动画
  gsap.from('.card', {
    y: 20,
    opacity: 0,
    duration: 0.6,
    stagger: 0.1,
    ease: 'power2.out'
  });

  // 初始化所有选择理由为收起状态
  expandedReasons.value = {};
});

// 切换暗色模式
function toggleDarkMode () {
  isDarkMode.value = !isDarkMode.value;
  if (isDarkMode.value) {
    document.documentElement.classList.add('dark');
    localStorage.setItem('theme', 'dark');
  } else {
    document.documentElement.classList.remove('dark');
    localStorage.setItem('theme', 'light');
  }
}

// 提供 isDarkMode 变量和 toggleDarkMode 函数，以便子组件可以注入它们
provide('isDarkMode', isDarkMode);
provide('toggleDarkMode', toggleDarkMode);

// 生成标记线数据
function generateMarkLines (stock) {
  // 打印完整的股票数据，用于调试
  console.log('生成标记线数据，股票数据:', stock);

  // 首先检查后端是否直接提供了标记线数据（支持两种可能的字段名）
  if (stock.mark_lines && stock.mark_lines.length > 0) {
    console.log('使用后端提供的标记线数据 (mark_lines):', stock.mark_lines);
    return stock.mark_lines;
  }

  if (stock.markLines && stock.markLines.length > 0) {
    console.log('使用后端提供的标记线数据 (markLines):', stock.markLines);
    return stock.markLines;
  }

  const markLines = [];
  console.log('后端未提供标记线数据，尝试从分析数据生成');

  // 如果有低位分析，添加高点标记
  if (stock.position_analysis && stock.position_analysis.details) {
    const details = stock.position_analysis.details;
    console.log('低位分析详情:', details);
    if (details.high_date) {
      markLines.push({
        date: details.high_date,
        text: '高点',
        color: '#ec0000' // 红色
      });
    }
  }

  // 如果有快速下跌分析，添加快速下跌开始和结束标记
  if (stock.decline_details) {
    const details = stock.decline_details;
    console.log('快速下跌详情:', details);
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
  if (stock.breakthrough_analysis && stock.breakthrough_analysis.has_breakthrough_signal) {
    const details = stock.breakthrough_analysis;
    console.log('突破分析详情:', details);
    if (details.breakthrough_date) {
      markLines.push({
        date: details.breakthrough_date,
        text: '突破',
        color: '#10b981' // 绿色
      });
    }
  }

  // 检查是否有decline_analysis字段（可能是后端返回的字段名不同）
  if (stock.decline_analysis) {
    const details = stock.decline_analysis;
    console.log('下跌分析详情 (decline_analysis):', details);
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

  // 检查是否有position_details字段（可能是后端返回的字段名不同）
  if (stock.position_details) {
    const details = stock.position_details;
    console.log('位置分析详情 (position_details):', details);
    if (details.high_date) {
      markLines.push({
        date: details.high_date,
        text: '高点',
        color: '#ec0000' // 红色
      });
    }
  }

  // 检查是否有decline_details字段（可能是后端返回的字段名不同）
  if (stock.decline_details) {
    const details = stock.decline_details;
    console.log('下跌分析详情 (decline_details):', details);
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

  console.log('生成的标记线数据:', markLines);
  return markLines;
}

// 获取支撑位数据
function getSupportLevels (stock) {
  if (!stock) return [];

  // 首先检查后端是否直接提供了箱体分析结果
  if (stock.box_analysis) {
    console.log('使用后端提供的箱体分析结果获取支撑位');
    return stock.box_analysis.support_levels || [];
  }

  if (!stock.details) return [];

  // 遍历所有窗口，查找箱体分析结果
  for (const window in stock.details) {
    if (stock.details[window].box_analysis) {
      console.log(`从窗口 ${window} 获取支撑位`);
      return stock.details[window].box_analysis.support_levels || [];
    }
  }

  return [];
}

// 获取阻力位数据
function getResistanceLevels (stock) {
  if (!stock) return [];

  // 首先检查后端是否直接提供了箱体分析结果
  if (stock.box_analysis) {
    console.log('使用后端提供的箱体分析结果获取阻力位');
    return stock.box_analysis.resistance_levels || [];
  }

  if (!stock.details) return [];

  // 遍历所有窗口，查找箱体分析结果
  for (const window in stock.details) {
    if (stock.details[window].box_analysis) {
      console.log(`从窗口 ${window} 获取阻力位`);
      return stock.details[window].box_analysis.resistance_levels || [];
    }
  }

  return [];
}

// 打开完整K线图
function openFullChart (stock) {
  console.log('打开K线图:', stock.code, stock.name);
  console.log('K线数据长度:', stock.kline_data?.length);
  console.log('当前主题模式:', isDarkMode.value ? '暗色' : '亮色');

  // 生成标记线数据
  const markLines = generateMarkLines(stock);
  console.log('标记线数据:', markLines);

  // 获取支撑位和阻力位
  const supportLevels = getSupportLevels(stock);
  const resistanceLevels = getResistanceLevels(stock);
  console.log('支撑位:', supportLevels);
  console.log('阻力位:', resistanceLevels);

  // 设置选中的股票和标记线数据
  selectedStock.value = {
    ...stock,
    markLines: markLines,
    supportLevels: supportLevels,
    resistanceLevels: resistanceLevels
  };
  showFullChart.value = true;
}

// 切换选择理由的展开/收起状态
function toggleReasonExpand (stockCode) {
  // 如果该股票的展开状态尚未初始化，则初始化为true（展开）
  // 否则切换当前状态
  expandedReasons.value[stockCode] = expandedReasons.value[stockCode] === undefined ?
    true : !expandedReasons.value[stockCode];

  // 创建安全的选择器ID（替换点号和其他特殊字符）
  const safeStockCode = stockCode.replace(/\./g, '_').replace(/[^\w-]/g, '');

  // 使用GSAP添加动画效果
  nextTick(() => {
    // 同时处理桌面端和移动端的元素
    const desktopElement = document.querySelector(`#reason-${safeStockCode}`);
    const mobileElement = document.querySelector(`#reason-mobile-${safeStockCode}`);

    const elements = [desktopElement, mobileElement].filter(el => el); // 过滤掉不存在的元素

    if (elements.length > 0) {
      if (expandedReasons.value[stockCode]) {
        // 展开动画
        elements.forEach(element => {
          gsap.fromTo(element,
            { height: 0, opacity: 0 },
            {
              height: 'auto',
              opacity: 1,
              duration: 0.3,
              ease: 'power2.out',
              onComplete: () => {
                // 确保展开后高度为auto
                element.style.height = 'auto';
              }
            }
          );
        });
      } else {
        // 收起动画
        elements.forEach(element => {
          // 先获取当前高度
          const height = element.offsetHeight;

          // 设置为具体高度，以便动画
          element.style.height = `${height}px`;

          // 强制回流
          element.offsetHeight;

          // 执行收起动画
          gsap.to(element, {
            height: 0,
            opacity: 0,
            duration: 0.3,
            ease: 'power2.in'
          });
        });
      }
    }
  });
}

// 导出股票到案例库
async function exportToCase (stock) {
  try {
    // 显示加载状态
    const loadingToast = showToast('正在导出到案例库...', 'loading');

    // 生成标记线数据
    const markLines = generateMarkLines(stock);
    console.log('导出到案例库: 标记线数据:', markLines);

    // 获取支撑位和阻力位
    const supportLevels = getSupportLevels(stock);
    const resistanceLevels = getResistanceLevels(stock);
    console.log('导出到案例库: 支撑位:', supportLevels);
    console.log('导出到案例库: 阻力位:', resistanceLevels);

    // 准备请求数据
    const exportData = {
      stockData: {
        code: stock.code,
        name: stock.name,
        industry: stock.industry || '未知行业'
      },
      analysisResult: {
        is_platform: true,
        platform_windows: Object.keys(stock.selection_reasons || {}).map(w => parseInt(w)),
        selection_reasons: stock.selection_reasons || {},
        parameters: {
          windows: parsedWindows.value,
          box_threshold: config.value.box_threshold,
          ma_diff_threshold: config.value.ma_diff_threshold,
          volatility_threshold: config.value.volatility_threshold,
          use_volume_analysis: config.value.use_volume_analysis,
          volume_change_threshold: config.value.volume_change_threshold,
          volume_stability_threshold: config.value.volume_stability_threshold,
          volume_increase_threshold: config.value.volume_increase_threshold,
          use_breakthrough_prediction: config.value.use_breakthrough_prediction,
          use_window_weights: config.value.use_window_weights,
          use_low_position: config.value.use_low_position,
          high_point_lookback_days: config.value.high_point_lookback_days,
          decline_period_days: config.value.decline_period_days,
          decline_threshold: config.value.decline_threshold,

          // 快速下跌判断参数
          use_rapid_decline_detection: config.value.use_rapid_decline_detection,
          rapid_decline_days: config.value.rapid_decline_days,
          rapid_decline_threshold: config.value.rapid_decline_threshold,

          // 箱体检测参数
          use_box_detection: config.value.use_box_detection,
          box_quality_threshold: config.value.box_quality_threshold
        },
        // 添加标记线数据
        mark_lines: markLines
      },
      klineData: stock.kline_data || []
    };

    // 如果有成交量分析，添加到结果中
    if (stock.volume_analysis) {
      exportData.analysisResult.volume_analysis = stock.volume_analysis;
    }

    // 如果有突破预测，添加到结果中
    if (stock.breakthrough_prediction) {
      exportData.analysisResult.breakthrough_prediction = stock.breakthrough_prediction;
    }

    // 如果有低位分析，添加到结果中
    if (stock.position_analysis) {
      exportData.analysisResult.position_analysis = stock.position_analysis;
    }

    // 如果有快速下跌分析，添加到结果中
    if (stock.decline_details) {
      exportData.analysisResult.decline_details = stock.decline_details;
      exportData.analysisResult.is_rapid_decline = stock.is_rapid_decline || false;
      exportData.analysisResult.has_decline_pattern = stock.has_decline_pattern || false;
    }

    // 如果有箱体分析，添加到结果中
    if (stock.box_analysis) {
      exportData.analysisResult.box_analysis = stock.box_analysis;

      // 确保箱体分析中包含支撑位和阻力位
      if (supportLevels && supportLevels.length > 0) {
        exportData.analysisResult.box_analysis.support_levels = supportLevels;
      }
      if (resistanceLevels && resistanceLevels.length > 0) {
        exportData.analysisResult.box_analysis.resistance_levels = resistanceLevels;
      }
    } else if (supportLevels.length > 0 || resistanceLevels.length > 0) {
      // 如果没有箱体分析但有支撑位或阻力位，创建箱体分析对象
      exportData.analysisResult.box_analysis = {
        is_box_pattern: true,
        box_quality: 0.8,
        support_levels: supportLevels,
        resistance_levels: resistanceLevels
      };
    }

    // 发送请求到后端
    const response = await axios.post('/api/cases/export', exportData);

    // 关闭加载提示
    closeToast(loadingToast);

    // 显示成功提示
    if (response.data && response.data.success) {
      showToast(`${stock.name} 已成功导出到案例库`, 'success');
    } else {
      throw new Error(response.data.message || '导出失败');
    }
  } catch (error) {
    console.error('导出到案例库失败:', error);
    showToast(`导出失败: ${error.message || '未知错误'}`, 'error');
  }
}

// 显示提示消息
function showToast (message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `fixed top-4 right-4 p-4 rounded-md shadow-md z-50 transition-all duration-300 transform translate-y-0 opacity-100 ${type === 'success' ? 'bg-gundam-blue' :
    type === 'error' ? 'bg-gundam-red' :
      type === 'loading' ? 'bg-gundam-yellow' : 'bg-gray-700'
    } text-white`;

  // 添加图标
  let icon = '';
  if (type === 'success') {
    icon = '<i class="fas fa-check-circle mr-2"></i>';
  } else if (type === 'error') {
    icon = '<i class="fas fa-exclamation-circle mr-2"></i>';
  } else if (type === 'loading') {
    icon = '<i class="fas fa-spinner fa-spin mr-2"></i>';
  } else {
    icon = '<i class="fas fa-info-circle mr-2"></i>';
  }

  toast.innerHTML = `<div class="flex items-center">${icon}${message}</div>`;
  document.body.appendChild(toast);

  // 如果不是加载中状态，3秒后自动关闭
  if (type !== 'loading') {
    setTimeout(() => {
      closeToast(toast);
    }, 3000);
  }

  return toast;
}

// 关闭提示消息
function closeToast (toast) {
  if (!toast) return;

  toast.classList.replace('translate-y-0', '-translate-y-4');
  toast.classList.replace('opacity-100', 'opacity-0');

  setTimeout(() => {
    if (toast.parentNode) {
      toast.parentNode.removeChild(toast);
    }
  }, 300);
}

// Computed property to parse windows input string into an array of numbers
const parsedWindows = computed(() => {
  const windows = config.value.windowsInput
    .split(',')
    .map(w => parseInt(w.trim(), 10))
    .filter(w => !isNaN(w) && w > 0);

  // Initialize window weights if needed
  windows.forEach(window => {
    if (windowWeights.value[window] === undefined) {
      windowWeights.value[window] = 5; // Default weight
    }
  });

  return windows;
});

// 选择窗口期预设
function selectWindowPreset (presetValue) {
  config.value.windowsInput = presetValue;
  showCustomWindowInput.value = false;

  // 重新初始化窗口权重
  parsedWindows.value.forEach(window => {
    if (windowWeights.value[window] === undefined) {
      windowWeights.value[window] = 5; // Default weight
    }
  });

  // 更新窗口权重
  updateWindowWeights();
}

// 验证自定义窗口期
function validateCustomWindows () {
  const windows = config.value.windowsInput
    .split(',')
    .map(w => parseInt(w.trim(), 10))
    .filter(w => !isNaN(w) && w > 0);

  if (windows.length === 0) {
    // 如果没有有效的窗口期，恢复为默认值
    config.value.windowsInput = '30,60,90';
    alert('请输入有效的窗口期，例如: 30,60,90');
  } else {
    // 格式化输入
    config.value.windowsInput = windows.join(',');
    showCustomWindowInput.value = false;

    // 重新初始化窗口权重
    windows.forEach(window => {
      if (windowWeights.value[window] === undefined) {
        windowWeights.value[window] = 5; // Default weight
      }
    });

    // 更新窗口权重
    updateWindowWeights();
  }
}

// Update window weights and config
function updateWindowWeights (window, value) {
  // 如果提供了特定窗口的值，更新它
  if (window !== undefined && value !== undefined) {
    windowWeights.value[window] = parseInt(value, 10);
  }

  // Update config.window_weights with normalized values
  const weights = {};
  let total = 0;

  // Calculate total
  for (const [key, value] of Object.entries(windowWeights.value)) {
    if (parsedWindows.value.includes(parseInt(key, 10))) {
      total += value;
    }
  }

  // Normalize weights
  if (total > 0) {
    for (const [key, value] of Object.entries(windowWeights.value)) {
      if (parsedWindows.value.includes(parseInt(key, 10))) {
        weights[key] = value / total;
      }
    }
  }

  // Update config
  config.value.window_weights = weights;
}

// 清理轮询定时器
onUnmounted(() => {
  if (pollingInterval.value) {
    clearInterval(pollingInterval.value);
  }
});

// 开始轮询任务状态
function startPolling (taskId) {
  // 清除之前的轮询
  if (pollingInterval.value) {
    clearInterval(pollingInterval.value);
  }

  // 设置新的轮询
  pollingInterval.value = setInterval(async () => {
    try {
      const response = await axios.get(`/api/scan/status/${taskId}`);
      const taskData = response.data;

      // 更新任务状态
      taskStatus.value = taskData.status;
      taskProgress.value = taskData.progress;
      taskMessage.value = taskData.message;

      // 如果任务完成或失败，停止轮询
      if (taskData.status === 'completed') {
        clearInterval(pollingInterval.value);
        loading.value = false;

        // 处理结果
        if (taskData.result && Array.isArray(taskData.result)) {
          // 处理每个股票数据，确保标记线数据正确
          const processedResults = taskData.result.map(stock => {
            // 如果后端返回了mark_lines字段，将其重命名为markLines
            if (stock.mark_lines) {
              console.log(`处理股票 ${stock.code} 的标记线数据:`, stock.mark_lines);
              stock.markLines = stock.mark_lines;
            }
            return stock;
          });

          platformStocks.value = processedResults;
          console.log('处理后的平台股票数据:', platformStocks.value);
        } else {
          console.error("Task completed but no valid result:", taskData);
          error.value = "任务完成但未返回有效数据。";
        }
      } else if (taskData.status === 'failed') {
        clearInterval(pollingInterval.value);
        loading.value = false;
        taskError.value = taskData.error;
      }
    } catch (e) {
      console.error("Error polling task status:", e);
      // 如果轮询出错，不要立即停止，可能是临时网络问题
    }
  }, 2000); // 每2秒轮询一次
}

async function fetchPlatformStocks () {
  if (loading.value) return; // Prevent multiple clicks

  loading.value = true;
  error.value = null;
  taskError.value = null;
  platformStocks.value = []; // Clear previous results
  expandedReasons.value = {}; // 重置所有选择理由为收起状态
  hasSearched.value = true; // Mark that a search was initiated

  // Reset task status
  taskStatus.value = 'pending';
  taskProgress.value = 0;
  taskMessage.value = '准备开始扫描...';

  // Basic validation
  if (!parsedWindows.value.length) {
    error.value = "请输入有效的窗口期天数 (正整数，用逗号分隔)";
    loading.value = false;
    taskStatus.value = null;
    return;
  }
  if (config.value.box_threshold <= 0 || config.value.box_threshold >= 1) {
    error.value = "振幅阈值应在 0 和 1 之间 (例如 0.3 代表 30%)";
    loading.value = false;
    taskStatus.value = null;
    return;
  }
  // Add similar validation for other thresholds if needed

  try {
    const payload = {
      // 基本参数
      windows: parsedWindows.value,
      expected_count: config.value.expected_count || 10,

      // 价格参数
      box_threshold: config.value.box_threshold,
      ma_diff_threshold: config.value.ma_diff_threshold,
      volatility_threshold: config.value.volatility_threshold,

      // 成交量参数
      use_volume_analysis: config.value.use_volume_analysis,
      volume_change_threshold: config.value.volume_change_threshold,
      volume_stability_threshold: config.value.volume_stability_threshold,
      volume_increase_threshold: config.value.volume_increase_threshold,

      // 技术指标参数
      use_technical_indicators: config.value.use_technical_indicators,
      use_breakthrough_prediction: config.value.use_breakthrough_prediction,

      // 位置参数
      use_low_position: config.value.use_low_position,
      high_point_lookback_days: config.value.high_point_lookback_days,
      decline_period_days: config.value.decline_period_days,
      decline_threshold: config.value.decline_threshold,

      // 快速下跌判断参数
      use_rapid_decline_detection: config.value.use_rapid_decline_detection,
      rapid_decline_days: config.value.rapid_decline_days,
      rapid_decline_threshold: config.value.rapid_decline_threshold,

      // 突破确认参数
      use_breakthrough_confirmation: config.value.use_breakthrough_confirmation,
      breakthrough_confirmation_days: config.value.breakthrough_confirmation_days,

      // 窗口权重参数
      use_window_weights: config.value.use_window_weights,
      window_weights: config.value.window_weights,

      // 箱体检测参数
      use_box_detection: config.value.use_box_detection,
      box_quality_threshold: config.value.box_quality_threshold,

      // 基本面筛选参数
      use_fundamental_filter: config.value.use_fundamental_filter,
      revenue_growth_percentile: config.value.revenue_growth_percentile,
      profit_growth_percentile: config.value.profit_growth_percentile,
      roe_percentile: config.value.roe_percentile,
      liability_percentile: config.value.liability_percentile,
      pe_percentile: config.value.pe_percentile,
      pb_percentile: config.value.pb_percentile,
      fundamental_years_to_check: config.value.fundamental_years_to_check
    };

    console.log("发送POST请求到 /api/scan/start...");
    const resp = await axios.post('/api/scan/start', payload);
    console.log("POST请求完成，获取任务ID:", resp.data);

    if (resp.data && resp.data.task_id) {
      currentTaskId.value = resp.data.task_id;
      taskMessage.value = resp.data.message || '任务已开始，正在处理...';
      taskStatus.value = 'running';

      // 开始轮询任务状态
      startPolling(currentTaskId.value);
    } else {
      throw new Error("服务器未返回有效的任务ID");
    }
  } catch (e) {
    console.error("Error starting scan:", e);
    error.value = `请求失败: ${e.message || '未知错误'}`;
    if (e.response) {
      console.error("Response error:", e.response.data);
      error.value = `服务器错误: ${e.response.data.detail || e.response.statusText}`;
    }
    loading.value = false;
    taskStatus.value = null;
  }
}

// 兼容旧版API的直接请求方法（用于测试）
async function fetchPlatformStocksLegacy () {
  if (loading.value) return;

  loading.value = true;
  error.value = null;
  platformStocks.value = [];
  hasSearched.value = true;

  // Basic validation
  if (!parsedWindows.value.length) {
    error.value = "请输入有效的窗口期天数";
    loading.value = false;
    return;
  }

  try {
    const payload = {
      // 基本参数
      windows: parsedWindows.value,
      expected_count: config.value.expected_count || 10,

      // 价格参数
      box_threshold: config.value.box_threshold,
      ma_diff_threshold: config.value.ma_diff_threshold,
      volatility_threshold: config.value.volatility_threshold,

      // 成交量参数
      use_volume_analysis: config.value.use_volume_analysis,
      volume_change_threshold: config.value.volume_change_threshold,
      volume_stability_threshold: config.value.volume_stability_threshold,
      volume_increase_threshold: config.value.volume_increase_threshold,

      // 技术指标参数
      use_technical_indicators: config.value.use_technical_indicators,
      use_breakthrough_prediction: config.value.use_breakthrough_prediction,

      // 位置参数
      use_low_position: config.value.use_low_position,
      high_point_lookback_days: config.value.high_point_lookback_days,
      decline_period_days: config.value.decline_period_days,
      decline_threshold: config.value.decline_threshold,

      // 快速下跌判断参数
      use_rapid_decline_detection: config.value.use_rapid_decline_detection,
      rapid_decline_days: config.value.rapid_decline_days,
      rapid_decline_threshold: config.value.rapid_decline_threshold,

      // 突破确认参数
      use_breakthrough_confirmation: config.value.use_breakthrough_confirmation,
      breakthrough_confirmation_days: config.value.breakthrough_confirmation_days,

      // 窗口权重参数
      use_window_weights: config.value.use_window_weights,
      window_weights: config.value.window_weights,

      // 箱体检测参数
      use_box_detection: config.value.use_box_detection,
      box_quality_threshold: config.value.box_quality_threshold,

      // 基本面筛选参数
      use_fundamental_filter: config.value.use_fundamental_filter,
      revenue_growth_percentile: config.value.revenue_growth_percentile,
      profit_growth_percentile: config.value.profit_growth_percentile,
      roe_percentile: config.value.roe_percentile,
      liability_percentile: config.value.liability_percentile,
      pe_percentile: config.value.pe_percentile,
      pb_percentile: config.value.pb_percentile,
      fundamental_years_to_check: config.value.fundamental_years_to_check
    };

    console.log("使用旧版API直接请求...");
    const resp = await axios.post('/api/scan', payload, {
      timeout: 300000
    });

    if (Array.isArray(resp.data)) {
      // 处理每个股票数据，确保标记线数据正确
      const processedResults = resp.data.map(stock => {
        // 如果后端返回了mark_lines字段，将其重命名为markLines
        if (stock.mark_lines) {
          console.log(`处理股票 ${stock.code} 的标记线数据:`, stock.mark_lines);
          stock.markLines = stock.mark_lines;
        }
        return stock;
      });

      platformStocks.value = processedResults;
      console.log('处理后的平台股票数据:', platformStocks.value);
    } else {
      error.value = "API返回的数据格式不正确";
    }
  } catch (e) {
    console.error("Error with legacy API:", e);
    error.value = `请求失败: ${e.message || '未知错误'}`;
  } finally {
    loading.value = false;
  }
}

</script>

<style>
/* Add any global styles if needed, or rely on Tailwind */
body {
  background-color: #f7fafc;
  /* Light gray background */
}

/* 防止图表在鼠标悬停时变模糊 */
.chart-container {
  isolation: isolate;
  /* 创建新的层叠上下文 */
  transform: translateZ(0);
  /* 启用硬件加速 */
  backface-visibility: hidden;
  /* 防止 3D 变换时的模糊 */
  will-change: transform;
  /* 告诉浏览器该元素会变化，优化渲染 */
  position: relative;
  /* 创建新的层叠上下文 */
  z-index: 1;
  /* 确保在正确的层级 */
}

/* 确保表格行不会影响图表渲染 */
tbody tr {
  transition: none;
  /* 移除所有过渡效果 */
}

/* 表格行悬停效果，但不影响子元素 */
tbody tr:hover>td:not(:last-child) {
  background-color: rgba(0, 0, 0, 0.05);
}
</style>