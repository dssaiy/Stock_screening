"""
Configuration module for stock platform scanner.
"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class ScanConfig(BaseModel):
    """Configuration model for stock platform scanner."""
    # Window settings - 基于安记食品平台期分析的最佳参数组合
    windows: List[int] = Field(default_factory=lambda: [
                               80, 100, 120])  # 使用安记食品平台期的最佳参数

    # Price pattern thresholds - 适合识别安记食品类型的平台期
    box_threshold: float = 0.3  # 降低箱体阈值
    ma_diff_threshold: float = 0.25  # 增加MA差异阈值
    volatility_threshold: float = 0.4  # 增加波动率阈值

    # Volume analysis settings - 适合识别安记食品类型的平台期
    use_volume_analysis: bool = True
    # Maximum volume change ratio for consolidation
    volume_change_threshold: float = 0.5  # 降低成交量变化阈值
    # Maximum volume stability for consolidation
    volume_stability_threshold: float = 0.5  # 降低成交量稳定性要求
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
    decline_threshold: float = 0.3  # 适中的下跌阈值

    # Rapid decline detection settings
    # Whether to use rapid decline detection
    use_rapid_decline_detection: bool = True
    rapid_decline_days: int = 30  # 适中的快速下跌窗口
    # Minimum decline percentage within rapid_decline_days to be considered rapid
    rapid_decline_threshold: float = 0.15  # 适中的快速下跌阈值

    # Breakthrough confirmation settings
    # Whether to use breakthrough confirmation
    use_breakthrough_confirmation: bool = False
    # Number of days to look for confirmation
    breakthrough_confirmation_days: int = 1

    # Box pattern detection settings
    use_box_detection: bool = True  # Whether to use box pattern detection
    # Minimum quality score for a valid box pattern
    box_quality_threshold: float = 0.3  # 降低箱体质量要求

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
    max_workers: int = 5
    retry_attempts: int = 2
    retry_delay: int = 1
    expected_count: int = 10


# Default configuration
DEFAULT_CONFIG = ScanConfig()


def merge_config(user_config: Dict[str, Any]) -> ScanConfig:
    """
    Merge user configuration with default configuration.

    Args:
        user_config: User-provided configuration

    Returns:
        Merged configuration
    """
    # Start with default config
    config_dict = DEFAULT_CONFIG.model_dump()

    # Update with user config
    for key, value in user_config.items():
        if key in config_dict and value is not None:
            config_dict[key] = value

    # Create new config object
    return ScanConfig(**config_dict)
