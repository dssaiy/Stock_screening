"""
Combined Analyzer module for comprehensive stock analysis.
"""
import pandas as pd
import math
from typing import Dict, Any, List, Optional

from .price_analyzer import analyze_price
from .volume_analyzer import analyze_volume
from .breakthrough_analyzer import analyze_breakthrough, check_breakthrough_confirmation
from .window_weight_analyzer import apply_window_weights
from .enhanced_platform_analyzer import analyze_enhanced_platform, check_enhanced_platform
from .box_detector import analyze_box_pattern, check_box_pattern
from .decline_analyzer import analyze_decline_speed, check_decline_pattern


def analyze_stock(df: pd.DataFrame,
                  windows: List[int] = [80, 100, 120],  # 使用安记食品平台期的最佳参数
                  box_threshold: float = 0.3,  # 降低箱体阈值
                  ma_diff_threshold: float = 0.25,  # 增加MA差异阈值
                  volatility_threshold: float = 0.4,  # 增加波动率阈值
                  volume_change_threshold: float = 0.5,  # 降低成交量变化阈值
                  volume_stability_threshold: float = 0.5,  # 降低成交量稳定性要求
                  volume_increase_threshold: float = 1.5,
                  use_volume_analysis: bool = True,  # 使用成交量分析
                  use_breakthrough_prediction: bool = False,
                  use_window_weights: bool = False,
                  window_weights: Dict[int, float] = None,
                  use_low_position: bool = True,
                  high_point_lookback_days: int = 365,
                  decline_period_days: int = 180,
                  decline_threshold: float = 0.4,  # 适中的下跌阈值
                  use_rapid_decline_detection: bool = True,
                  rapid_decline_days: int = 30,  # 适中的快速下跌窗口
                  rapid_decline_threshold: float = 0.15,  # 适中的快速下跌阈值
                  use_breakthrough_confirmation: bool = False,
                  breakthrough_confirmation_days: int = 1,
                  use_box_detection: bool = True,
                  box_quality_threshold: float = 0.3) -> Dict[str, Any]:
    """
    Analyze a stock for platform periods across multiple time windows,
    including price analysis, volume analysis, breakthrough prediction, position analysis,
    decline speed analysis, box pattern detection, and window weighting.

    Args:
        df: DataFrame containing stock price and volume data
        windows: List of window sizes to check
        box_threshold: Maximum allowed price range
        ma_diff_threshold: Maximum allowed MA convergence
        volatility_threshold: Maximum allowed volatility
        volume_change_threshold: Maximum allowed volume change ratio
        volume_stability_threshold: Maximum allowed volume stability
        volume_increase_threshold: Minimum required volume increase ratio
        use_volume_analysis: Whether to include volume analysis
        use_breakthrough_prediction: Whether to include breakthrough prediction
        use_window_weights: Whether to use window weights for scoring
        window_weights: Dictionary mapping window sizes to weights
        use_low_position: Whether to use low position analysis
        high_point_lookback_days: Number of days to look back for finding the high point
        decline_period_days: Number of days within which the decline should have occurred
        decline_threshold: Minimum decline percentage from high to be considered at low position
        use_rapid_decline_detection: Whether to use rapid decline detection
        rapid_decline_days: Number of days to define a rapid decline period
        rapid_decline_threshold: Minimum decline percentage within rapid_decline_days to be considered rapid
        use_breakthrough_confirmation: Whether to use breakthrough confirmation analysis
        breakthrough_confirmation_days: Number of days to look for confirmation
        use_box_detection: Whether to use box pattern detection
        box_quality_threshold: Minimum quality score for a valid box pattern

    Returns:
        Dict containing comprehensive analysis results
    """
    if df.empty:
        return {
            "is_platform": False,
            "windows_checked": windows,
            "platform_windows": [],
            "details": {w: {"status": "无数据"} for w in windows},
            "selection_reasons": {}
        }

    # Check each window
    platform_windows = []
    details = {}
    selection_reasons = {}
    volume_analysis_results = {}
    breakthrough_results = {}

    # Perform position analysis if requested
    position_result = None
    decline_result = None
    has_decline_pattern = False

    if use_low_position:
        if use_rapid_decline_detection:
            # Use enhanced decline analysis with rapid decline detection
            has_decline_pattern, decline_result = check_decline_pattern(
                df,
                high_point_lookback_days,
                decline_period_days,
                decline_threshold,
                rapid_decline_days,
                rapid_decline_threshold
            )

            # Create position result from decline result for compatibility
            position_result = {
                "is_low_position": decline_result.get("is_low_position", False),
                "details": decline_result.get("details", {})
            }
        else:
            # Use traditional position analysis
            from .position_analyzer import analyze_position
            position_result = analyze_position(
                df,
                high_point_lookback_days,
                decline_period_days,
                decline_threshold
            )

    # Perform breakthrough prediction if requested
    if use_breakthrough_prediction:
        breakthrough_analysis = analyze_breakthrough(df)
        breakthrough_results = breakthrough_analysis

    # Perform breakthrough confirmation if requested
    confirmation_result = None
    if use_breakthrough_confirmation:
        confirmation_result = check_breakthrough_confirmation(
            df,
            breakthrough_confirmation_days
        )

    # Use enhanced platform analysis if box detection is enabled
    if use_box_detection:
        # Perform enhanced platform analysis
        enhanced_result = analyze_enhanced_platform(
            df,
            windows,
            box_threshold,
            ma_diff_threshold,
            volatility_threshold,
            volume_change_threshold,
            volume_stability_threshold,
            box_quality_threshold,
            use_box_detection
        )

        # Extract platform windows and details
        platform_windows = enhanced_result["platform_windows"]
        details = enhanced_result["details"]
        selection_reasons = enhanced_result["selection_reasons"]

        # Store volume analysis results for later use
        for window in windows:
            if window in details and "volume_analysis" in details[window]:
                volume_analysis_results[window] = {
                    "has_consolidation_volume": True,  # Assume true if included in platform windows
                    "has_breakthrough": False,  # Will be updated later if needed
                    "consolidation_details": details[window]["volume_analysis"],
                    "breakthrough_details": {"status": "未分析"}
                }
    else:
        # Use traditional analysis method
        for window in windows:
            # Price analysis
            price_analysis = analyze_price(
                df, window, box_threshold, ma_diff_threshold, volatility_threshold
            )

            # Volume analysis if requested and volume data is available
            if use_volume_analysis and 'volume' in df.columns:
                volume_analysis = analyze_volume(
                    df, window, volume_change_threshold,
                    volume_stability_threshold, volume_increase_threshold
                )
                volume_analysis_results[window] = volume_analysis
            else:
                volume_analysis = {
                    "has_consolidation_volume": True,  # Default to True if not using volume analysis
                    "has_breakthrough": False,
                    "consolidation_details": {"status": "未分析"},
                    "breakthrough_details": {"status": "未分析"}
                }

            # Combine price and volume analysis
            is_price_platform = price_analysis["is_price_platform"]
            has_consolidation_volume = volume_analysis["has_consolidation_volume"]

            # A stock is in platform period if price forms a platform and volume shows consolidation
            is_window_platform = is_price_platform
            if use_volume_analysis:
                is_window_platform = is_window_platform and has_consolidation_volume

            # Store details
            details[window] = {
                "price_analysis": price_analysis["details"],
                "volume_analysis": volume_analysis["consolidation_details"] if use_volume_analysis else {"status": "未分析"}
            }

            # Check for breakthrough
            has_breakthrough = volume_analysis["has_breakthrough"] if use_volume_analysis else False
            details[window]["breakthrough"] = volume_analysis["breakthrough_details"] if use_volume_analysis else {
                "status": "未分析"}

            # Add to platform windows if it meets criteria
            if is_window_platform:
                platform_windows.append(window)

                # Create selection reason
                price_details = price_analysis["details"]
                volume_details = volume_analysis["consolidation_details"] if use_volume_analysis else {
                    "status": "未分析"}

                reason = f"{window}日平台期: 价格区间{price_details.get('box_range', 'N/A'):.2f}, "
                reason += f"均线收敛{price_details.get('ma_diff', 'N/A'):.2f}, "
                reason += f"波动率{price_details.get('volatility', 'N/A'):.2f}"

                if use_volume_analysis:
                    reason += f", 成交量变化{volume_details.get('volume_change_ratio', 'N/A'):.2f}"

                    # Add breakthrough information if available
                    if has_breakthrough:
                        breakthrough_details = volume_analysis["breakthrough_details"]
                        reason += f", 成交量突破{breakthrough_details.get('volume_increase_ratio', 'N/A'):.2f}倍"

                selection_reasons[window] = reason

    # 基本平台期判断 - 至少有一个窗口满足价格模式和成交量条件
    is_basic_platform = len(platform_windows) > 0

    # 初始化平台期判断结果
    is_platform = is_basic_platform

    # 记录判断过程
    platform_judgment_log = []
    platform_judgment_log.append(
        f"基本平台期判断: {is_basic_platform} (窗口: {platform_windows})")

    # 如果启用了低位分析，还需要满足低位条件
    if use_low_position and position_result:
        is_low_position = position_result["is_low_position"]
        is_platform = is_platform and is_low_position
        platform_judgment_log.append(f"低位判断: {is_low_position}")

    # 如果启用了快速下跌检测，还需要满足快速下跌条件
    if use_rapid_decline_detection and decline_result:
        is_rapid_decline = decline_result.get("is_rapid_decline", False)
        is_platform = is_platform and is_rapid_decline
        platform_judgment_log.append(f"快速下跌判断: {is_rapid_decline}")

    # 添加箱体检测结果
    box_analysis_results = {}
    if use_box_detection:
        # 使用最大窗口进行箱体检测，以获取更稳定的支撑位和阻力位
        max_window = max(windows) if windows else 90
        box_analysis = analyze_box_pattern(df, max_window)
        box_analysis_results = box_analysis

        # 如果启用了箱体检测，还需要满足箱体条件
        is_box_pattern = box_analysis.get("is_box_pattern", False)
        is_platform = is_platform and is_box_pattern
        platform_judgment_log.append(f"箱体检测: {is_box_pattern}")

    # 记录最终判断结果
    platform_judgment_log.append(f"最终平台期判断: {is_platform}")
    print(f"平台期判断过程: {' -> '.join(platform_judgment_log)}")

    # 重新计算标准模式（低位+快速下跌后形成平台期）
    has_decline_pattern = False
    if is_basic_platform:  # 首先必须满足基本平台期条件
        # 只有当低位分析和快速下跌分析都启用时，才检查标准模式
        if use_low_position and use_rapid_decline_detection and position_result and decline_result:
            has_decline_pattern = (position_result["is_low_position"] and
                                   decline_result.get("is_rapid_decline", False) and
                                   is_basic_platform)
            platform_judgment_log.append(f"标准模式判断: {has_decline_pattern}")

    # 生成标记线数据
    mark_lines = []

    # 如果启用了低位分析，添加高点标记
    if use_low_position and position_result and "details" in position_result:
        details = position_result["details"]
        if "high_date" in details:
            # 将Timestamp转换为字符串
            high_date = str(details["high_date"]).split()[0]  # 只保留日期部分
            mark_lines.append({
                "date": high_date,
                "text": "高点",
                "color": "#ec0000"  # 红色
            })

    # 如果启用了快速下跌检测，添加快速下跌开始和结束标记
    if use_rapid_decline_detection and decline_result and "details" in decline_result:
        details = decline_result["details"]
        if "rapid_decline_start_date" in details:
            # 将Timestamp转换为字符串
            start_date = str(details["rapid_decline_start_date"]).split()[
                0]  # 只保留日期部分
            mark_lines.append({
                "date": start_date,
                "text": "开始下跌",
                "color": "#ec0000"  # 红色
            })
        if "rapid_decline_end_date" in details:
            # 将Timestamp转换为字符串
            end_date = str(details["rapid_decline_end_date"]).split()[
                0]  # 只保留日期部分
            mark_lines.append({
                "date": end_date,
                "text": "平台期开始",
                "color": "#3b82f6"  # 蓝色
            })

    # 如果启用了箱体检测，添加支撑位和阻力位水平线
    if use_box_detection and box_analysis_results:
        # 检查支撑位
        if "support_levels" in box_analysis_results and box_analysis_results["support_levels"]:
            support_levels = box_analysis_results["support_levels"]
            if isinstance(support_levels, list) and len(support_levels) > 0:
                for i, level in enumerate(support_levels[:2]):  # 最多添加两个支撑位
                    mark_lines.append({
                        "type": "horizontal",
                        "value": float(level),
                        "text": f"支撑位{i+1}" if i > 0 else "支撑位",
                        "color": "#10b981"  # 绿色
                    })

        # 检查阻力位
        if "resistance_levels" in box_analysis_results and box_analysis_results["resistance_levels"]:
            resistance_levels = box_analysis_results["resistance_levels"]
            if isinstance(resistance_levels, list) and len(resistance_levels) > 0:
                for i, level in enumerate(resistance_levels[:2]):  # 最多添加两个阻力位
                    mark_lines.append({
                        "type": "horizontal",
                        "value": float(level),
                        "text": f"阻力位{i+1}" if i > 0 else "阻力位",
                        "color": "#ec0000"  # 红色
                    })

    result = {
        "is_platform": is_platform,  # 使用新的平台期判断结果
        "windows_checked": windows,
        "platform_windows": platform_windows,
        "details": details,
        "selection_reasons": selection_reasons,
        "volume_analysis": volume_analysis_results if use_volume_analysis else {},
        "box_analysis": box_analysis_results if use_box_detection else {},
        "mark_lines": mark_lines,  # 直接添加标记线数据
        "platform_judgment_log": platform_judgment_log  # 添加判断过程日志
    }

    # Add position analysis results if available
    if position_result:
        result["is_low_position"] = position_result["is_low_position"]
        result["position_details"] = position_result["details"]

        # Add low position as a selection reason if applicable
        if position_result["is_low_position"]:
            # Get decline percentage from the appropriate source
            if decline_result and "details" in decline_result:
                decline_pct = decline_result["details"].get(
                    "decline_percentage", 0)
                # 检查是否为有效浮点数
                if isinstance(decline_pct, float) and (math.isnan(decline_pct) or math.isinf(decline_pct)):
                    decline_pct = 0
                else:
                    decline_pct = decline_pct * 100
                high_date = decline_result["details"].get("high_date", "未知")
            else:
                decline_pct = position_result["details"].get("decline_pct", 0)
                # 检查是否为有效浮点数
                if isinstance(decline_pct, float) and (math.isnan(decline_pct) or math.isinf(decline_pct)):
                    decline_pct = 0
                high_date = position_result["details"].get("high_date", "未知")

            position_reason = f"低位: 从高点下跌{decline_pct:.2f}%, 高点日期{high_date}"

            # 不再使用非整数键，而是将信息添加到所有平台期窗口
            # Add to all platform windows
            for window in platform_windows:
                if window in selection_reasons:
                    selection_reasons[window] += f", {position_reason}"

        # Add rapid decline information if available
        if decline_result:
            result["is_rapid_decline"] = decline_result.get(
                "is_rapid_decline", False)
            result["decline_details"] = decline_result.get("details", {})
            # 使用新的标准模式判断结果
            result["has_decline_pattern"] = has_decline_pattern

            # Add rapid decline as a selection reason
            if "details" in decline_result:
                details = decline_result["details"]
                rapid_decline_pct = details.get("max_rapid_decline", 0)
                # 检查是否为有效浮点数
                if isinstance(rapid_decline_pct, float) and (math.isnan(rapid_decline_pct) or math.isinf(rapid_decline_pct)):
                    rapid_decline_pct = 0
                else:
                    rapid_decline_pct = rapid_decline_pct * 100
                rapid_start = details.get("rapid_decline_start_date", "未知")
                rapid_end = details.get("rapid_decline_end_date", "未知")

                rapid_reason = f"快速下跌: {rapid_decline_pct:.2f}% ({rapid_start} 至 {rapid_end})"

                # 不再使用非整数键，而是将信息添加到所有平台期窗口
                # Add to all platform windows
                for window in platform_windows:
                    if window in selection_reasons:
                        selection_reasons[window] += f", {rapid_reason}"

                # Add decline pattern information if it matches the pattern
                if has_decline_pattern:
                    pattern_reason = "标准模式: 低位+快速下跌后形成平台期"

                    # 不再使用非整数键，而是将信息添加到所有平台期窗口
                    # Add to all platform windows
                    for window in platform_windows:
                        if window in selection_reasons:
                            selection_reasons[window] += f", {pattern_reason}"
        else:
            result["is_rapid_decline"] = False
            result["has_decline_pattern"] = False

    # Add breakthrough prediction results if available
    if use_breakthrough_prediction:
        result["breakthrough_prediction"] = breakthrough_results

        # Add breakthrough information to selection reasons
        if breakthrough_results.get("has_breakthrough_signal"):
            signal_count = breakthrough_results.get("signal_count", 0)
            signals = breakthrough_results.get("signals", {})

            # Create a summary of signals
            signal_summary = []
            for indicator, has_signal in signals.items():
                if has_signal:
                    signal_summary.append(indicator)

            # Add to all platform windows
            for window in platform_windows:
                if window in selection_reasons:
                    selection_reasons[window] += f", 突破前兆: {signal_count}个指标 ({', '.join(signal_summary)})"

    # Add breakthrough confirmation results if available
    if confirmation_result:
        result["has_breakthrough"] = confirmation_result["has_breakthrough"]
        result["has_breakthrough_confirmation"] = confirmation_result["has_confirmation"]
        result["breakthrough_confirmation_details"] = confirmation_result["details"]

        # Add breakthrough confirmation as a selection reason if applicable
        if confirmation_result["has_confirmation"]:
            confirmation_reason = f"突破已确认: 突破日期{confirmation_result['details']['breakthrough_date']}, 确认天数{confirmation_result['details']['confirmation_days']}"

            # 不再使用非整数键，而是将信息添加到所有平台期窗口
            # Add to all platform windows
            for window in platform_windows:
                if window in selection_reasons:
                    selection_reasons[window] += f", {confirmation_reason}"

    # Apply window weights if requested
    if use_window_weights and window_weights:
        # Convert window weights to proper format if needed
        if not isinstance(window_weights, dict):
            # Try to convert from string or other formats
            try:
                if isinstance(window_weights, str):
                    # Parse from string format like "30:0.5,60:0.3,90:0.2"
                    weights_dict = {}
                    for item in window_weights.split(','):
                        if ':' in item:
                            window, weight = item.split(':')
                            weights_dict[int(window.strip())] = float(
                                weight.strip())
                    window_weights = weights_dict
                else:
                    # Default to equal weights for all windows
                    window_weights = {window: 1.0 for window in windows}
            except Exception:
                # If parsing fails, use equal weights
                window_weights = {window: 1.0 for window in windows}

        # Apply window weights to get weighted score
        result = apply_window_weights(result, window_weights)

        # Add weighted score to selection reasons if it's a platform stock
        if is_platform and result.get("weighted_score", 0) > 0:
            weighted_score = result.get("weighted_score", 0)
            # 检查是否为有效浮点数
            if isinstance(weighted_score, float) and (math.isnan(weighted_score) or math.isinf(weighted_score)):
                weighted_score = 0

            for window in platform_windows:
                if window in selection_reasons:
                    selection_reasons[window] += f", 加权得分: {weighted_score:.2f}"

    return result
