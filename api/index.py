# api/index.py
from typing import Optional
from colorama import Fore, Style
import colorama  # For colored console output
import traceback
import pandas as pd
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, RootModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, BackgroundTasks
import sys
import os
from diskcache import Cache
import hashlib
import json # 确保json已导入，如果未导入则添加
# 🚩【项目根路径自动添加到sys.path，便于模块导入】-------------------
import os
import sys
# 获取当前文件所在的目录 (api/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (Stock screening/)
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 Python 路径
sys.path.insert(0, project_root)
# ------------------------------------------------------------

 


# Import our modular components (using absolute imports)
try:
    from api.config import ScanConfig
    from api.task_manager import task_manager, TaskStatus
    from api.data_fetcher import fetch_stock_basics, fetch_industry_data, BaostockConnectionManager, get_next_1am_timestamp
    from api.platform_scanner import prepare_stock_list, scan_stocks
    from api.case_api import router as case_router
except ImportError:
    # 如果绝对导入失败，尝试相对导入（本地开发环境）
    from .config import ScanConfig
    from .task_manager import task_manager, TaskStatus
    from .data_fetcher import fetch_stock_basics, fetch_industry_data, BaostockConnectionManager, get_next_1am_timestamp
    from .platform_scanner import prepare_stock_list, scan_stocks
    from .case_api import router as case_router


# Define request body model using Pydantic
class ScanConfigRequest(BaseModel):
    """Request model for stock platform scan configuration."""
    # Window settings - 基于平台期分析的最佳参数组合
    windows: List[int] = Field(default_factory=lambda: [20, 30, 60])

    # Price pattern thresholds - 适合识别安记食品类型的平台期
    box_threshold: float = 0.5
    ma_diff_threshold: float = 0.03
    volatility_threshold: float = 0.09  # 从0.04调整到0.09，以便更好地识别平台期

    # Volume analysis settings - 适合平台期
    use_volume_analysis: bool = True
    # Maximum volume change ratio for consolidation
    volume_change_threshold: float = 0.9
    # Maximum volume stability for consolidation
    volume_stability_threshold: float = 0.75  # 从0.7调整到0.75，以便在20天窗口也能识别出平台期
    # Minimum volume increase ratio for breakthrough
    volume_increase_threshold: float = 1.5

    # Technical indicators
    use_technical_indicators: bool = False  # Whether to use technical indicators
    # Whether to use breakthrough prediction
    use_breakthrough_prediction: bool = False

    # Position analysis settings
    use_low_position: bool = True  # Whether to use low position analysis
    # Number of days to look back for finding the high point
    high_point_lookback_days: int = 365
    # Number of days within which the decline should have occurred
    decline_period_days: int = 180
    # Minimum decline percentage from high to be considered at low position
    decline_threshold: float = 0.3  # 从0.5降低到0.3，更符合实际情况

    # Rapid decline detection settings
    # Whether to use rapid decline detection
    use_rapid_decline_detection: bool = True
    rapid_decline_days: int = 30  # Number of days to define a rapid decline period
    # Minimum decline percentage within rapid_decline_days to be considered rapid
    rapid_decline_threshold: float = 0.15

    # Breakthrough confirmation settings
    # Whether to use breakthrough confirmation
    use_breakthrough_confirmation: bool = False
    # Number of days to look for confirmation
    breakthrough_confirmation_days: int = 1

    # Box pattern detection settings
    use_box_detection: bool = True  # Whether to use box pattern detection
    # Minimum quality score for a valid box pattern
    box_quality_threshold: float = 0.6

    # Fundamental analysis settings
    use_fundamental_filter: bool = False  # 是否启用基本面筛选
    # 营收增长率行业百分位要求（值越小要求越严格，如0.3表示要求位于行业前30%）
    revenue_growth_percentile: float = 0.3
    # 净利润增长率行业百分位要求（值越小要求越严格，如0.3表示要求位于行业前30%）
    profit_growth_percentile: float = 0.3
    # ROE行业百分位要求（值越小要求越严格，如0.3表示要求位于行业前30%）
    roe_percentile: float = 0.3
    # 资产负债率行业百分位要求（值越大要求越严格，如0.3表示要求位于行业后30%）
    liability_percentile: float = 0.3
    # PE行业百分位要求（值越大要求越宽松，如0.7表示要求不在行业前30%最高估值）
    pe_percentile: float = 0.7
    # PB行业百分位要求（值越大要求越宽松，如0.7表示要求不在行业前30%最高估值）
    pb_percentile: float = 0.7
    # 检查连续增长的年数
    fundamental_years_to_check: int = 3

    # Window weights
    use_window_weights: bool = False  # Whether to use window weights
    window_weights: Dict[int, float] = Field(
        default_factory=dict)  # Weights for different windows

    # System settings
    max_workers: int = 5  # Keep concurrency reasonable for serverless
    retry_attempts: int = 2
    retry_delay: int = 1
    expected_count: int = 10  # 期望返回的股票数量，默认为10

# --- Define response models ---


class SelectionReasons(RootModel[Dict[int, str]]):
    """Maps window sizes to selection reasons (descriptive text)"""
    pass


class KlineDataPoint(BaseModel):
    """Model for a single K-line data point."""
    date: str
    open: Optional[float] = None  # 允许为 None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    turn: Optional[float] = None
    preclose: Optional[float] = None
    pctChg: Optional[float] = None
    peTTM: Optional[float] = None
    pbMRQ: Optional[float] = None


class MarkLine(BaseModel):
    """Model for a marking line on a chart."""
    date: Optional[str] = None
    text: str
    color: str
    type: Optional[str] = None
    value: Optional[float] = None


class StockScanResult(BaseModel):
    """Model for a stock that meets platform criteria."""
    code: str
    name: str
    industry: Optional[str] = "未知行业"
    selection_reasons: Dict[int, str]
    kline_data: List[KlineDataPoint]
    mark_lines: Optional[List[MarkLine]] = None

# --- Task-related models ---


class TaskCreationResponse(BaseModel):
    """Response model for task creation."""
    task_id: str
    message: str


class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str
    status: str
    progress: int
    message: str
    result: Optional[List[StockScanResult]] = None
    error: Optional[str] = None
    cached: bool = False # 新增字段
    created_at: float
    updated_at: float
    completed_at: Optional[float] = None


# Initialize FastAPI app
scan_results_cache = Cache("cache/scan_results", cull_limit=5)

app = FastAPI(
    title="Stock Platform Scanner API",
    description="API for scanning stocks for platform consolidation patterns",
    version="1.0.0"
)

# 配置 Uvicorn 的日志级别
import logging
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.WARNING) # 或者 logging.ERROR

# 如果需要，也可以配置 uvicorn.access 的日志级别
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.setLevel(logging.WARNING) # 或者 logging.ERROR

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gleaming-tanuki-7aafab.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include case management router
app.include_router(case_router, prefix="/api")

# --- API Endpoints ---
@app.get("/health")
async def health():
    return {"status": "OK"}

@app.get("/")
async def root():
    """
    Root endpoint for health check.
    """
    return {
        "status": "ok",
        "message": "Stock Platform Scanner API is running",
        "version": "1.0.0"
    }


@app.post("/api/scan/start", response_model=TaskCreationResponse)
async def start_scan(config_request: ScanConfigRequest, background_tasks: BackgroundTasks):
    """
    Start a stock platform scan as a background task.
    Returns a task ID that can be used to check the status of the scan.
    """
    # Initialize colorama for colored console output
    colorama.init()

    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Starting stock platform scan task{Style.RESET_ALL}")
    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")

    # Create a new task
    task_id = task_manager.create_task()

    # Convert request to config dictionary
    config_dict = config_request.model_dump()
    print(f"{Fore.YELLOW}Scan configuration:{Style.RESET_ALL}")
    for key, value in config_dict.items():
        print(f"  - {key}: {Fore.GREEN}{value}{Style.RESET_ALL}")

    # 生成缓存键
    config_json = config_request.model_dump_json()
    cache_key = hashlib.md5(config_json.encode('utf-8')).hexdigest()

    print(f"{Fore.BLUE}生成的配置 JSON (用于缓存键):{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{config_json}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}生成的缓存键: {cache_key}{Style.RESET_ALL}")

    # 尝试从缓存加载
    cached_result = scan_results_cache.get(cache_key)

    if cached_result:
        print(f"{Fore.GREEN}从分析结果缓存加载: {cache_key} 😊{Style.RESET_ALL}")
        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message="Scan completed from cache.",
            result=cached_result,
            cached=True # 新增字段
        )
        return TaskCreationResponse(
            task_id=task_id,
            message="Scan started successfully. Result loaded from cache."
        )
    else:
        print(f"{Fore.YELLOW}分析结果缓存未命中，开始执行扫描: {cache_key} 🌐{Style.RESET_ALL}")

    # Start the scan in the background
    def run_scan_task(task_id: str, config_dict: Dict[str, any], cache_key: str):
        try:
            # Fetch stock basics
            with BaostockConnectionManager():
                stock_basics_df = fetch_stock_basics()

                # Update task status
                task_manager.update_task(
                    task_id,
                    progress=10,
                    message="Fetched stock basic information"
                )

                # Fetch industry data
                try:
                    industry_df = fetch_industry_data()
                    task_manager.update_task(
                        task_id,
                        progress=20,
                        message="Fetched industry classification data"
                    )
                except Exception as e:
                    print(
                        f"{Fore.YELLOW}Warning: Failed to fetch industry data: {e}{Style.RESET_ALL}")
                    industry_df = pd.DataFrame()
                    task_manager.update_task(
                        task_id,
                        progress=20,
                        message="Warning: Failed to fetch industry data, continuing without it"
                    )

                # Prepare stock list
                stock_list = prepare_stock_list(stock_basics_df, industry_df)
                task_manager.update_task(
                    task_id,
                    progress=30,
                    message=f"Prepared list of {len(stock_list)} stocks for scanning"
                )

                # Create scan config
                scan_config = ScanConfig(**config_dict)

                # Define progress update callback
                def update_progress(progress=None, message=None):
                    if progress is not None and message is not None:
                        # Scale progress to 30-90 range (30% for preparation, 60% for scanning, 10% for post-processing)
                        scaled_progress = 30 + int(progress * 0.6)
                        task_manager.update_task(
                            task_id, progress=scaled_progress, message=message)

                # Run the scan
                platform_stocks = scan_stocks(
                    stock_list, scan_config, update_progress)

                # Process results for API response
                result_stocks = []
                for stock in platform_stocks:
                    # Convert kline_data to KlineDataPoint objects
                    kline_data = []
                    for point in stock.get('kline_data', []):
                        try:
                            kline_point = {
                                'date': str(point.get('date')),
                                'open': float(point['open']) if point.get('open') is not None else None,
                                'high': float(point['high']) if point.get('high') is not None else None,
                                'low': float(point['low']) if point.get('low') is not None else None,
                                'close': float(point['close']) if point.get('close') is not None else None,
                                'volume': float(point['volume']) if point.get('volume') is not None else None,
                                'turn': float(point['turn']) if point.get('turn') is not None else None,
                                'preclose': float(point['preclose']) if point.get('preclose') is not None else None,
                                'pctChg': float(point['pctChg']) if point.get('pctChg') is not None else None,
                                'peTTM': float(point['peTTM']) if point.get('peTTM') is not None else None,
                                'pbMRQ': float(point['pbMRQ']) if point.get('pbMRQ') is not None else None,
                            }
                            kline_data.append(KlineDataPoint(**kline_point))
                        except Exception as e:
                            print(
                                f"{Fore.YELLOW}Warning: Failed to process K-line data point: {e}{Style.RESET_ALL}")
                            continue

                    # Create StockScanResult object
                    try:
                        # 处理标记线数据
                        mark_lines = []
                        if 'mark_lines' in stock:
                            for mark in stock['mark_lines']:
                                try:
                                    mark_lines.append(MarkLine(**mark))
                                except Exception as e:
                                    print(
                                        f"{Fore.YELLOW}Warning: Failed to process mark line: {e}{Style.RESET_ALL}")
                                    continue

                        result_stock = StockScanResult(
                            code=stock['code'],
                            name=stock['name'],
                            industry=stock.get('industry', '未知行业'),
                            selection_reasons=stock.get(
                                'selection_reasons', {}),
                            kline_data=kline_data,
                            mark_lines=mark_lines
                        )
                        result_stocks.append(result_stock)
                    except Exception as e:
                        print(
                            f"{Fore.RED}Error creating StockScanResult: {e}{Style.RESET_ALL}")
                        continue

                # 将结果存入缓存
                scan_results_cache.set(cache_key, [stock.model_dump() for stock in result_stocks], expire=get_next_1am_timestamp())
                print(f"{Fore.GREEN}分析结果已存入缓存 (有效期至次日凌晨1点): {cache_key} ✅{Style.RESET_ALL}")

                # Update task with final result
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    progress=100,
                    message=f"Scan completed. Found {len(result_stocks)} platform stocks.",
                    result=[stock.model_dump() for stock in result_stocks],
                    cached=False # 新增字段
                )

        except Exception as e:
            print(f"{Fore.RED}Error in scan task: {e}{Style.RESET_ALL}")
            traceback.print_exc()
            task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error=f"Scan failed: {str(e)}\n{traceback.format_exc()}"
            )

    # Start the task in the background
    background_tasks.add_task(run_scan_task, task_id, config_dict, cache_key)

    # Return task ID
    return TaskCreationResponse(
        task_id=task_id,
        message="Scan started successfully. Use the task ID to check status."
    )


@app.get("/api/scan/status/{task_id}", response_model=TaskStatusResponse)
async def get_scan_status(task_id: str):
    """
    Get the status of a scan task.
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=404, detail=f"Task with ID {task_id} not found")

    return task.to_dict()

# Legacy endpoint for backward compatibility


@app.post("/api/scan", response_model=List[StockScanResult])
async def run_scan(config_request: ScanConfigRequest):
    """
    Legacy API endpoint for backward compatibility.
    This endpoint starts a scan and waits for it to complete.
    For long-running scans, use the /api/scan/start endpoint instead.
    """
    # Initialize colorama for colored console output
    colorama.init()

    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Legacy scan endpoint called{Style.RESET_ALL}")
    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")

    # Convert request to config dictionary
    config_dict = config_request.model_dump()

    # Create scan config
    scan_config = ScanConfig(**config_dict)

    # Fetch stock basics
    with BaostockConnectionManager():
        stock_basics_df = fetch_stock_basics()

        # Fetch industry data
        try:
            industry_df = fetch_industry_data()
        except Exception as e:
            print(
                f"{Fore.YELLOW}Warning: Failed to fetch industry data: {e}{Style.RESET_ALL}")
            industry_df = pd.DataFrame()

        # Prepare stock list
        stock_list = prepare_stock_list(stock_basics_df, industry_df)

        # Run the scan
        platform_stocks = scan_stocks(stock_list, scan_config)

    # Process results for API response
    result_stocks = []
    for stock in platform_stocks:
        # Convert kline_data to KlineDataPoint objects
        kline_data = []
        for point in stock.get('kline_data', []):
            try:
                kline_point = {
                    'date': str(point.get('date')),
                    'open': float(point['open']) if point.get('open') is not None else None,
                    'high': float(point['high']) if point.get('high') is not None else None,
                    'low': float(point['low']) if point.get('low') is not None else None,
                    'close': float(point['close']) if point.get('close') is not None else None,
                    'volume': float(point['volume']) if point.get('volume') is not None else None,
                    'turn': float(point['turn']) if point.get('turn') is not None else None,
                    'preclose': float(point['preclose']) if point.get('preclose') is not None else None,
                    'pctChg': float(point['pctChg']) if point.get('pctChg') is not None else None,
                    'peTTM': float(point['peTTM']) if point.get('peTTM') is not None else None,
                    'pbMRQ': float(point['pbMRQ']) if point.get('pbMRQ') is not None else None,
                }
                kline_data.append(KlineDataPoint(**kline_point))
            except Exception as e:
                print(
                    f"{Fore.YELLOW}Warning: Failed to process K-line data point: {e}{Style.RESET_ALL}")
                continue

        # Create StockScanResult object
        try:
            # 处理标记线数据
            mark_lines = []
            if 'mark_lines' in stock:
                for mark in stock['mark_lines']:
                    try:
                        mark_lines.append(MarkLine(**mark))
                    except Exception as e:
                        print(
                            f"{Fore.YELLOW}Warning: Failed to process mark line: {e}{Style.RESET_ALL}")
                        continue

            result_stock = StockScanResult(
                code=stock['code'],
                name=stock['name'],
                industry=stock.get('industry', '未知行业'),
                selection_reasons=stock.get('selection_reasons', {}),
                kline_data=kline_data,
                mark_lines=mark_lines
            )
            result_stocks.append(result_stock)
        except Exception as e:
            print(f"{Fore.RED}Error creating StockScanResult: {e}{Style.RESET_ALL}")
            continue

    return result_stocks

# 添加测试API端点


@app.post("/api/scan/test", response_model=List[StockScanResult])
async def test_scan(config_request: ScanConfigRequest):
    """
    Test API endpoint that returns sample data with marking lines.
    This is useful for testing the frontend without running a full scan.
    """
    # 创建一个模拟的股票数据
    from datetime import datetime, timedelta
    import numpy as np

    # 生成日期序列
    end_date = datetime.now()
    dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d')
             for i in range(200)]
    dates.reverse()  # 按时间顺序排列

    # 生成价格数据
    high_price = 30.0
    prices = []

    # 上涨阶段
    for i in range(50):
        prices.append(20 + i * 0.2)

    # 高点和下跌阶段
    for i in range(30):
        prices.append(high_price - i * 0.3)

    # 平台期
    platform_price = 20.0
    for i in range(100):
        # 在平台价格附近波动
        prices.append(platform_price + np.random.normal(0, 0.5))

    # 突破
    for i in range(20):
        prices.append(platform_price + 2 + i * 0.1)

    # 创建K线数据
    kline_data = []
    for i, date in enumerate(dates):
        if i < len(prices):
            price = prices[i]
            kline_point = {
                'date': date,
                'open': price - 0.2,
                'high': price + 0.5,
                'low': price - 0.5,
                'close': price + 0.2,
                'volume': 10000 + np.random.randint(0, 5000),
                'turn': 1.5,
                'preclose': price if i == 0 else prices[i-1],
                'pctChg': 0.5,
                'peTTM': 15.0,
                'pbMRQ': 2.0
            }
            kline_data.append(KlineDataPoint(**kline_point))

    # 创建标记线数据
    mark_lines = [
        MarkLine(date=dates[49], text="高点", color="#ec0000"),
        MarkLine(date=dates[50], text="开始下跌", color="#ec0000"),
        MarkLine(date=dates[80], text="平台期开始", color="#3b82f6"),
        MarkLine(date=dates[180], text="突破", color="#10b981")
    ]

    # 创建支撑位和阻力位
    support_level = platform_price - 0.5
    resistance_level = platform_price + 0.5

    mark_lines.append(MarkLine(type="horizontal",
                      value=support_level, text="支撑位", color="#10b981"))
    mark_lines.append(MarkLine(type="horizontal",
                      value=resistance_level, text="阻力位", color="#ec0000"))

    # 创建结果对象
    result_stock = StockScanResult(
        code="sh.000001",
        name="测试股票",
        industry="测试行业",
        selection_reasons={60: "60天窗口期内价格波动小于50%，均线高度粘合，波动率低，成交量稳定"},
        kline_data=kline_data,
        mark_lines=mark_lines
    )

    return [result_stock]

# 注意：我们已经有了根端点 (/), 不需要额外的 /api 端点
