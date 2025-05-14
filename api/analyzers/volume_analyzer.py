"""
Volume Analyzer module for analyzing stock volume patterns.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

def calculate_volume_features(df: pd.DataFrame, window: int) -> Dict[str, float]:
    """
    Calculate volume-related features for a given window.
    
    Args:
        df: DataFrame containing stock price and volume data
        window: Window size in days
    
    Returns:
        Dict containing calculated volume features:
            - volume_change_ratio: Ratio of recent volume to historical volume
            - volume_stability: Stability of volume in the window
            - volume_trend: Trend of volume in the window (positive or negative)
    """
    if len(df) < window + 10:  # Need extra data for comparison
        return {
            'volume_change_ratio': float('nan'),
            'volume_stability': float('nan'),
            'volume_trend': float('nan')
        }
    
    # Get the most recent window of data and previous data for comparison
    recent_df = df.iloc[-window:].copy()
    previous_df = df.iloc[-(window*2):-window].copy()
    
    # Calculate average volume
    recent_avg_volume = recent_df['volume'].mean()
    previous_avg_volume = previous_df['volume'].mean()
    
    # Calculate volume change ratio
    if previous_avg_volume > 0:
        volume_change_ratio = recent_avg_volume / previous_avg_volume
    else:
        volume_change_ratio = float('nan')
    
    # Calculate volume stability (coefficient of variation)
    if recent_avg_volume > 0:
        volume_stability = recent_df['volume'].std() / recent_avg_volume
    else:
        volume_stability = float('nan')
    
    # Calculate volume trend (linear regression slope)
    if len(recent_df) >= 5:
        x = np.arange(len(recent_df))
        y = recent_df['volume'].values
        slope, _ = np.polyfit(x, y, 1)
        volume_trend = slope / recent_avg_volume if recent_avg_volume > 0 else float('nan')
    else:
        volume_trend = float('nan')
    
    return {
        'volume_change_ratio': volume_change_ratio,
        'volume_stability': volume_stability,
        'volume_trend': volume_trend
    }

def check_volume_pattern(df: pd.DataFrame, window: int, 
                         volume_change_threshold: float = 0.8,
                         volume_stability_threshold: float = 0.5) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a stock has a consolidation volume pattern (typically decreasing or stable volume).
    
    Args:
        df: DataFrame containing stock price and volume data
        window: Window size in days
        volume_change_threshold: Maximum allowed volume change ratio
        volume_stability_threshold: Maximum allowed volume stability
    
    Returns:
        Tuple of (is_consolidation_volume, details)
    """
    if len(df) < window + 10:
        return False, {
            "status": "数据不足",
            "window": window,
            "data_points": len(df)
        }
    
    # Calculate volume features
    features = calculate_volume_features(df, window)
    
    # Check conditions for consolidation volume pattern
    # Typically, we want decreasing or stable volume during consolidation
    is_decreasing_volume = features['volume_change_ratio'] <= volume_change_threshold
    is_stable_volume = features['volume_stability'] <= volume_stability_threshold
    
    is_consolidation_volume = is_decreasing_volume and is_stable_volume
    
    # Prepare details
    details = {
        "window": window,
        "volume_change_ratio": round(features['volume_change_ratio'], 4) if not pd.isna(features['volume_change_ratio']) else None,
        "volume_change_threshold": volume_change_threshold,
        "volume_stability": round(features['volume_stability'], 4) if not pd.isna(features['volume_stability']) else None,
        "volume_stability_threshold": volume_stability_threshold,
        "volume_trend": round(features['volume_trend'], 4) if not pd.isna(features['volume_trend']) else None
    }
    
    # Add reason if not a consolidation volume pattern
    if not is_consolidation_volume:
        if not is_decreasing_volume:
            details["status"] = "成交量变化过大"
        elif not is_stable_volume:
            details["status"] = "成交量波动过大"
        else:
            details["status"] = "未知原因"
    else:
        details["status"] = "符合条件"
    
    return is_consolidation_volume, details

def check_volume_breakthrough(df: pd.DataFrame, window: int = 5, 
                             volume_increase_threshold: float = 1.5) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a stock has a volume breakthrough pattern (typically increasing volume).
    
    Args:
        df: DataFrame containing stock price and volume data
        window: Window size in days for recent volume
        volume_increase_threshold: Minimum required volume increase ratio
    
    Returns:
        Tuple of (is_breakthrough, details)
    """
    if len(df) < window + 10:
        return False, {
            "status": "数据不足",
            "window": window,
            "data_points": len(df)
        }
    
    # Get the most recent window of data and previous data for comparison
    recent_df = df.iloc[-window:].copy()
    previous_df = df.iloc[-(window*3):-window].copy()
    
    # Calculate average volume
    recent_avg_volume = recent_df['volume'].mean()
    previous_avg_volume = previous_df['volume'].mean()
    
    # Calculate volume increase ratio
    if previous_avg_volume > 0:
        volume_increase_ratio = recent_avg_volume / previous_avg_volume
    else:
        volume_increase_ratio = float('nan')
    
    # Check if there's a volume breakthrough
    is_breakthrough = volume_increase_ratio >= volume_increase_threshold
    
    # Prepare details
    details = {
        "window": window,
        "recent_avg_volume": round(recent_avg_volume, 2) if not pd.isna(recent_avg_volume) else None,
        "previous_avg_volume": round(previous_avg_volume, 2) if not pd.isna(previous_avg_volume) else None,
        "volume_increase_ratio": round(volume_increase_ratio, 4) if not pd.isna(volume_increase_ratio) else None,
        "volume_increase_threshold": volume_increase_threshold
    }
    
    if is_breakthrough:
        details["status"] = "成交量突破"
    else:
        details["status"] = "无成交量突破"
    
    return is_breakthrough, details

def analyze_volume(df: pd.DataFrame, window: int, 
                  volume_change_threshold: float = 0.8,
                  volume_stability_threshold: float = 0.5,
                  volume_increase_threshold: float = 1.5) -> Dict[str, Any]:
    """
    Analyze volume patterns for a stock.
    
    Args:
        df: DataFrame containing stock price and volume data
        window: Window size in days
        volume_change_threshold: Maximum allowed volume change ratio for consolidation
        volume_stability_threshold: Maximum allowed volume stability for consolidation
        volume_increase_threshold: Minimum required volume increase ratio for breakthrough
    
    Returns:
        Dict containing volume analysis results
    """
    if df.empty or 'volume' not in df.columns:
        return {
            "has_consolidation_volume": False,
            "has_breakthrough": False,
            "consolidation_details": {"status": "无数据"},
            "breakthrough_details": {"status": "无数据"}
        }
    
    # Check for consolidation volume pattern
    has_consolidation_volume, consolidation_details = check_volume_pattern(
        df, window, volume_change_threshold, volume_stability_threshold
    )
    
    # Check for volume breakthrough
    has_breakthrough, breakthrough_details = check_volume_breakthrough(
        df, min(5, window), volume_increase_threshold
    )
    
    return {
        "has_consolidation_volume": has_consolidation_volume,
        "has_breakthrough": has_breakthrough,
        "consolidation_details": consolidation_details,
        "breakthrough_details": breakthrough_details
    }
