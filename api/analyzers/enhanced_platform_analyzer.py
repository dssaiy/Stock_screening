"""
Enhanced Platform Analyzer Module

This module combines traditional platform period detection with box pattern detection
to provide more accurate identification of consolidation patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from .price_analyzer import analyze_price, check_price_pattern
from .volume_analyzer import analyze_volume
from .box_detector import check_box_pattern


def check_enhanced_platform(df: pd.DataFrame, window: int,
                            box_threshold: float, ma_diff_threshold: float,
                            volatility_threshold: float,
                            volume_change_threshold: float = 0.9,
                            volume_stability_threshold: float = 0.75,
                            box_quality_threshold: float = 0.6,
                            use_box_detection: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a stock is in a platform consolidation period using enhanced detection.

    Args:
        df: DataFrame containing stock price data
        window: Window size in days
        box_threshold: Maximum allowed price range
        ma_diff_threshold: Maximum allowed MA convergence
        volatility_threshold: Maximum allowed volatility
        volume_change_threshold: Maximum allowed volume change
        volume_stability_threshold: Maximum allowed volume stability
        box_quality_threshold: Minimum quality score for a valid box pattern
        use_box_detection: Whether to use box pattern detection

    Returns:
        Tuple of (is_platform, details)
    """
    if len(df) < window:
        return False, {
            "status": "数据不足",
            "window": window,
            "data_points": len(df)
        }

    # Check traditional platform period
    is_traditional_platform, traditional_details = check_price_pattern(
        df, window, box_threshold, ma_diff_threshold, volatility_threshold
    )

    # Check volume conditions if data available
    volume_details = {}
    is_volume_ok = True

    if 'volume' in df.columns:
        volume_analysis = analyze_volume(
            df, window, volume_change_threshold, volume_stability_threshold
        )
        is_volume_ok = volume_analysis.get('has_consolidation_volume', False)
        volume_details = volume_analysis.get('consolidation_details', {})

    # Check box pattern if enabled
    box_details = {}
    is_box = False

    if use_box_detection:
        is_box, box_details = check_box_pattern(
            df, window, box_quality_threshold, volatility_threshold
        )

    # Combine results
    # A stock is considered in a platform period if:
    # 1. It meets traditional platform criteria, AND
    # 2. It meets volume criteria (if volume data available), AND
    # 3. It meets box pattern criteria (if box detection enabled)

    is_platform = is_traditional_platform and is_volume_ok

    if use_box_detection:
        # If box detection is enabled, the stock must also meet box pattern criteria
        is_platform = is_platform and is_box

    # Prepare combined details
    details = {
        **traditional_details,
        "volume_analysis": volume_details,
        "box_analysis": box_details if use_box_detection else {"status": "未启用箱体检测"}
    }

    # Add overall status
    if not is_platform:
        if not is_traditional_platform:
            details["status"] = traditional_details.get("status", "不符合传统平台期条件")
        elif not is_volume_ok:
            details["status"] = "成交量条件不符"
        elif use_box_detection and not is_box:
            details["status"] = box_details.get("status", "不符合箱体条件")
        else:
            details["status"] = "不符合平台期条件"
    else:
        details["status"] = "符合平台期条件"

    return is_platform, details


def analyze_enhanced_platform(df: pd.DataFrame,
                              windows: List[int],
                              box_threshold: float,
                              ma_diff_threshold: float,
                              volatility_threshold: float,
                              volume_change_threshold: float = 0.9,
                              volume_stability_threshold: float = 0.75,
                              box_quality_threshold: float = 0.6,
                              use_box_detection: bool = True) -> Dict[str, Any]:
    """
    Analyze a stock for platform periods across multiple time windows using enhanced detection.

    Args:
        df: DataFrame containing stock price data
        windows: List of window sizes to check
        box_threshold: Maximum allowed price range
        ma_diff_threshold: Maximum allowed MA convergence
        volatility_threshold: Maximum allowed volatility
        volume_change_threshold: Maximum allowed volume change
        volume_stability_threshold: Maximum allowed volume stability
        box_quality_threshold: Minimum quality score for a valid box pattern
        use_box_detection: Whether to use box pattern detection

    Returns:
        Dict containing analysis results
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

    for window in windows:
        is_platform, window_details = check_enhanced_platform(
            df, window, box_threshold, ma_diff_threshold, volatility_threshold,
            volume_change_threshold, volume_stability_threshold,
            box_quality_threshold, use_box_detection
        )

        details[window] = window_details

        if is_platform:
            platform_windows.append(window)

            # Create selection reason string
            reason_parts = []
            reason_parts.append(f"{window}日平台期")

            if 'box_range' in window_details:
                reason_parts.append(f"价格区间{window_details['box_range']:.2f}")

            if 'ma_diff' in window_details:
                reason_parts.append(f"均线收敛{window_details['ma_diff']:.2f}")

            if 'volatility' in window_details:
                reason_parts.append(f"波动率{window_details['volatility']:.2f}")

            if use_box_detection and 'box_analysis' in window_details:
                box_quality = window_details['box_analysis'].get(
                    'box_quality', 0)
                if box_quality > 0:
                    reason_parts.append(f"箱体质量{box_quality:.2f}")

            selection_reasons[window] = ", ".join(reason_parts)

    # Determine overall platform status
    is_platform = len(platform_windows) > 0

    return {
        "is_platform": is_platform,
        "windows_checked": windows,
        "platform_windows": platform_windows,
        "details": details,
        "selection_reasons": selection_reasons,
        "enhanced_detection": True,
        "box_detection_used": use_box_detection
    }
