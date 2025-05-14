"""
Price Analyzer module for identifying price patterns in stock data.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

def calculate_price_features(df: pd.DataFrame, window: int) -> Dict[str, float]:
    """
    Calculate price-related features for platform identification based on a specific window.
    
    Args:
        df: DataFrame containing stock price data
        window: Window size in days
    
    Returns:
        Dict containing calculated features:
            - box_range: Price range relative to minimum price
            - ma_diff: Moving average convergence
            - volatility: Price volatility
    """
    if len(df) < window:
        return {
            'box_range': float('inf'),
            'ma_diff': float('inf'),
            'volatility': float('inf')
        }
    
    # Get the most recent window of data
    recent_df = df.iloc[-window:].copy()
    
    # Calculate price range (box)
    price_high = recent_df['high'].max()
    price_low = recent_df['low'].min()
    price_range = price_high - price_low
    box_range = price_range / price_low if price_low > 0 else float('inf')
    
    # Calculate moving average convergence
    ma_values = []
    for ma_period in [5, 10, 20, 30]:
        if len(recent_df) >= ma_period:
            ma = recent_df['close'].rolling(window=ma_period).mean()
            ma_values.append(ma.iloc[-1])
    
    if len(ma_values) >= 2:
        ma_std = np.std(ma_values)
        ma_mean = np.mean(ma_values)
        ma_diff = ma_std / ma_mean if ma_mean > 0 else float('inf')
    else:
        ma_diff = float('inf')
    
    # Calculate volatility (standard deviation of daily returns)
    if len(recent_df) >= 3:
        daily_returns = recent_df['close'].pct_change().dropna()
        volatility = daily_returns.std()
    else:
        volatility = float('inf')
    
    return {
        'box_range': box_range,
        'ma_diff': ma_diff,
        'volatility': volatility
    }

def check_price_pattern(df: pd.DataFrame, window: int, 
                       box_threshold: float, 
                       ma_diff_threshold: float,
                       volatility_threshold: float) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a stock's price is in a platform consolidation pattern.
    
    Args:
        df: DataFrame containing stock price data
        window: Window size in days
        box_threshold: Maximum allowed price range
        ma_diff_threshold: Maximum allowed MA convergence
        volatility_threshold: Maximum allowed volatility
    
    Returns:
        Tuple of (is_platform, details)
    """
    if len(df) < window:
        return False, {
            "status": "数据不足",
            "window": window,
            "data_points": len(df)
        }
    
    # Calculate features
    features = calculate_price_features(df, window)
    
    # Check conditions
    is_platform = (
        features['box_range'] <= box_threshold and
        features['ma_diff'] <= ma_diff_threshold and
        features['volatility'] <= volatility_threshold
    )
    
    # Prepare details
    details = {
        "window": window,
        "box_range": round(features['box_range'], 4),
        "box_threshold": box_threshold,
        "ma_diff": round(features['ma_diff'], 4),
        "ma_diff_threshold": ma_diff_threshold,
        "volatility": round(features['volatility'], 4),
        "volatility_threshold": volatility_threshold,
    }
    
    # Add reason if not a platform
    if not is_platform:
        if features['box_range'] > box_threshold:
            details["status"] = "价格区间过大"
        elif features['ma_diff'] > ma_diff_threshold:
            details["status"] = "均线发散"
        elif features['volatility'] > volatility_threshold:
            details["status"] = "波动性过高"
        else:
            details["status"] = "未知原因"
    else:
        details["status"] = "符合条件"
    
    return is_platform, details

def analyze_price(df: pd.DataFrame, 
                 window: int,
                 box_threshold: float, 
                 ma_diff_threshold: float,
                 volatility_threshold: float) -> Dict[str, Any]:
    """
    Analyze a stock's price pattern for a specific time window.
    
    Args:
        df: DataFrame containing stock price data
        window: Window size to check
        box_threshold: Maximum allowed price range
        ma_diff_threshold: Maximum allowed MA convergence
        volatility_threshold: Maximum allowed volatility
    
    Returns:
        Dict containing price analysis results
    """
    if df.empty:
        return {
            "is_price_platform": False,
            "window": window,
            "details": {"status": "无数据"}
        }
    
    # Check price pattern
    is_price_platform, details = check_price_pattern(
        df, window, box_threshold, ma_diff_threshold, volatility_threshold
    )
    
    return {
        "is_price_platform": is_price_platform,
        "window": window,
        "details": details
    }
