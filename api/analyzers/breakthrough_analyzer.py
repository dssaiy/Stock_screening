"""
Breakthrough Analyzer module for identifying potential breakthrough patterns.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from .technical_indicators import (
    calculate_all_indicators,
    calculate_macd,
    calculate_rsi,
    calculate_kdj,
    calculate_bollinger_bands
)


def check_macd_signal(df: pd.DataFrame, lookback_period: int = 5) -> Tuple[bool, Dict[str, Any]]:
    """
    Check for MACD breakthrough signals.

    Args:
        df: DataFrame containing price and indicator data
        lookback_period: Period to look back for signal confirmation

    Returns:
        Tuple of (has_signal, details)
    """
    if 'macd' not in df.columns or 'macd_signal' not in df.columns:
        df = calculate_macd(df)

    if len(df) < lookback_period + 2:
        return False, {"status": "数据不足", "indicator": "MACD"}

    # Get recent data
    recent_df = df.iloc[-lookback_period-2:].copy()

    # Check for MACD crossover (MACD line crosses above signal line)
    macd_values = recent_df['macd'].values
    signal_values = recent_df['macd_signal'].values

    # Check if MACD was below signal line and now crossed above
    crossover = False
    for i in range(1, len(macd_values)):
        if (macd_values[i-1] < signal_values[i-1]) and (macd_values[i] > signal_values[i]):
            crossover = True
            break

    # Check if MACD is increasing
    macd_increasing = recent_df['macd'].diff().iloc[-1] > 0

    # Check if histogram is increasing
    hist_increasing = recent_df['macd_hist'].diff().iloc[-1] > 0

    # Determine if there's a breakthrough signal
    has_signal = crossover or (macd_increasing and hist_increasing)

    # Prepare details
    details = {
        "indicator": "MACD",
        "status": "有突破信号" if has_signal else "无突破信号",
        "crossover": crossover,
        "macd_increasing": macd_increasing,
        "histogram_increasing": hist_increasing,
        "current_macd": round(float(recent_df['macd'].iloc[-1]), 4),
        "current_signal": round(float(recent_df['macd_signal'].iloc[-1]), 4),
        "current_histogram": round(float(recent_df['macd_hist'].iloc[-1]), 4)
    }

    return has_signal, details


def check_rsi_signal(df: pd.DataFrame, oversold_threshold: float = 30, overbought_threshold: float = 70) -> Tuple[bool, Dict[str, Any]]:
    """
    Check for RSI breakthrough signals.

    Args:
        df: DataFrame containing price and indicator data
        oversold_threshold: Threshold for oversold condition
        overbought_threshold: Threshold for overbought condition

    Returns:
        Tuple of (has_signal, details)
    """
    if 'rsi' not in df.columns:
        df = calculate_rsi(df)

    if len(df) < 5:
        return False, {"status": "数据不足", "indicator": "RSI"}

    # Get recent data
    recent_df = df.iloc[-5:].copy()

    # Get current RSI and previous RSI
    current_rsi = recent_df['rsi'].iloc[-1]
    prev_rsi = recent_df['rsi'].iloc[-2]

    # Check for oversold to normal transition (potential bullish signal)
    oversold_to_normal = (prev_rsi < oversold_threshold) and (
        current_rsi >= oversold_threshold)

    # Check if RSI is increasing
    rsi_increasing = current_rsi > prev_rsi

    # Determine if there's a breakthrough signal
    has_signal = oversold_to_normal or (rsi_increasing and current_rsi > 50)

    # Prepare details
    details = {
        "indicator": "RSI",
        "status": "有突破信号" if has_signal else "无突破信号",
        "oversold_to_normal": oversold_to_normal,
        "rsi_increasing": rsi_increasing,
        "current_rsi": round(float(current_rsi), 2),
        "previous_rsi": round(float(prev_rsi), 2)
    }

    return has_signal, details


def check_kdj_signal(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Check for KDJ breakthrough signals.

    Args:
        df: DataFrame containing price and indicator data

    Returns:
        Tuple of (has_signal, details)
    """
    if not all(col in df.columns for col in ['k', 'd', 'j']):
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df = calculate_kdj(df)
        else:
            return False, {"status": "数据不足", "indicator": "KDJ"}

    if len(df) < 5:
        return False, {"status": "数据不足", "indicator": "KDJ"}

    # Get recent data
    recent_df = df.iloc[-5:].copy()

    # Check for golden cross (K line crosses above D line)
    k_values = recent_df['k'].values
    d_values = recent_df['d'].values

    # Check if K was below D and now crossed above
    golden_cross = False
    for i in range(1, len(k_values)):
        if (k_values[i-1] < d_values[i-1]) and (k_values[i] > d_values[i]):
            golden_cross = True
            break

    # Check if K and J are both increasing
    k_increasing = recent_df['k'].diff().iloc[-1] > 0
    j_increasing = recent_df['j'].diff().iloc[-1] > 0

    # Determine if there's a breakthrough signal
    has_signal = golden_cross or (k_increasing and j_increasing)

    # Prepare details
    details = {
        "indicator": "KDJ",
        "status": "有突破信号" if has_signal else "无突破信号",
        "golden_cross": golden_cross,
        "k_increasing": k_increasing,
        "j_increasing": j_increasing,
        "current_k": round(float(recent_df['k'].iloc[-1]), 2),
        "current_d": round(float(recent_df['d'].iloc[-1]), 2),
        "current_j": round(float(recent_df['j'].iloc[-1]), 2)
    }

    return has_signal, details


def check_bollinger_bands_signal(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Check for Bollinger Bands breakthrough signals.

    Args:
        df: DataFrame containing price and indicator data

    Returns:
        Tuple of (has_signal, details)
    """
    if not all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower', 'bb_bandwidth']):
        df = calculate_bollinger_bands(df)

    if len(df) < 5:
        return False, {"status": "数据不足", "indicator": "布林带"}

    # Get recent data
    recent_df = df.iloc[-5:].copy()

    # Check if price is near upper band
    close_to_upper = recent_df['close'].iloc[-1] > (recent_df['bb_middle'].iloc[-1] + 0.5 * (
        recent_df['bb_upper'].iloc[-1] - recent_df['bb_middle'].iloc[-1]))

    # Check if bandwidth is expanding (volatility increasing)
    bandwidth_expanding = recent_df['bb_bandwidth'].diff().iloc[-1] > 0

    # Check if price is above middle band and increasing
    above_middle = recent_df['close'].iloc[-1] > recent_df['bb_middle'].iloc[-1]
    price_increasing = recent_df['close'].diff().iloc[-1] > 0

    # Determine if there's a breakthrough signal
    has_signal = (close_to_upper and bandwidth_expanding) or (
        above_middle and price_increasing)

    # Prepare details
    details = {
        "indicator": "布林带",
        "status": "有突破信号" if has_signal else "无突破信号",
        "close_to_upper": close_to_upper,
        "bandwidth_expanding": bandwidth_expanding,
        "above_middle": above_middle,
        "price_increasing": price_increasing,
        "current_bandwidth": round(float(recent_df['bb_bandwidth'].iloc[-1]), 4),
        "bandwidth_change": round(float(recent_df['bb_bandwidth'].diff().iloc[-1]), 4)
    }

    return has_signal, details


def analyze_breakthrough(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze potential breakthrough patterns using multiple technical indicators.

    Args:
        df: DataFrame containing price data

    Returns:
        Dict containing breakthrough analysis results
    """
    if df.empty or 'close' not in df.columns:
        return {
            "has_breakthrough_signal": False,
            "signal_count": 0,
            "signals": {},
            "status": "无数据"
        }

    # Calculate all indicators
    df_with_indicators = calculate_all_indicators(df)

    # Check for signals from different indicators
    macd_signal, macd_details = check_macd_signal(df_with_indicators)
    rsi_signal, rsi_details = check_rsi_signal(df_with_indicators)

    # Check KDJ only if high/low data is available
    if all(col in df.columns for col in ['high', 'low']):
        kdj_signal, kdj_details = check_kdj_signal(df_with_indicators)
    else:
        kdj_signal, kdj_details = False, {
            "status": "无高低价数据", "indicator": "KDJ"}

    bb_signal, bb_details = check_bollinger_bands_signal(df_with_indicators)

    # Count positive signals
    signals = {
        "MACD": macd_signal,
        "RSI": rsi_signal,
        "KDJ": kdj_signal,
        "布林带": bb_signal
    }

    signal_count = sum(1 for signal in signals.values() if signal)
    # Require at least 2 indicators to confirm
    has_breakthrough_signal = signal_count >= 2

    # Prepare details
    details = {
        "MACD": macd_details,
        "RSI": rsi_details,
        "KDJ": kdj_details,
        "布林带": bb_details
    }

    return {
        "has_breakthrough_signal": has_breakthrough_signal,
        "signal_count": signal_count,
        "signals": signals,
        "details": details,
        "status": f"有{signal_count}个指标显示突破信号" if signal_count > 0 else "无突破信号"
    }


def check_breakthrough_confirmation(df: pd.DataFrame,
                                    confirmation_days: int = 1) -> Dict[str, Any]:
    """
    Check if a breakthrough has been confirmed by subsequent days' price action.

    Args:
        df: DataFrame containing price data
        confirmation_days: Number of days to look for confirmation (default: 1 for next day)

    Returns:
        Dict containing confirmation analysis results
    """
    if df.empty or len(df) < confirmation_days + 5:  # Need enough data for analysis
        return {
            "has_confirmation": False,
            "has_breakthrough": False,
            "details": {"status": "数据不足以进行确认分析"}
        }

    # Get the most recent data
    recent_df = df.iloc[-5-confirmation_days:].copy()

    # Check for potential breakthrough day
    # A breakthrough day typically has higher volume and a significant price increase
    volume_avg = recent_df['volume'].iloc[:-confirmation_days-1].mean()
    price_avg = recent_df['close'].iloc[:-confirmation_days-1].mean()

    breakthrough_day_idx = len(recent_df) - confirmation_days - 1
    breakthrough_day = recent_df.iloc[breakthrough_day_idx]

    # Define breakthrough criteria
    volume_increase = breakthrough_day['volume'] / volume_avg > 1.5
    price_increase = breakthrough_day['close'] > breakthrough_day['open'] and \
        (breakthrough_day['close'] - breakthrough_day['open']
         ) / breakthrough_day['open'] > 0.02

    # Check confirmation days
    confirmation_data = recent_df.iloc[-confirmation_days:]

    # For confirmation, we want to see continued upward movement or at least holding the gains
    confirmed = True
    confirmation_details = []

    for i, day in confirmation_data.iterrows():
        # Allow for small pullback
        day_confirmed = day['close'] >= breakthrough_day['close'] * 0.98
        confirmation_details.append({
            "date": day['date'],
            "close": float(day['close']),
            "confirmed": day_confirmed
        })
        confirmed = confirmed and day_confirmed

    # Determine if there's a confirmed breakthrough
    has_breakthrough = volume_increase and price_increase
    has_confirmation = has_breakthrough and confirmed

    # Prepare details
    details = {
        "breakthrough_date": breakthrough_day['date'],
        "breakthrough_close": round(float(breakthrough_day['close']), 2),
        "breakthrough_volume_increase": round(float(breakthrough_day['volume'] / volume_avg), 2),
        "breakthrough_price_change_pct": round(float((breakthrough_day['close'] - breakthrough_day['open']) / breakthrough_day['open'] * 100), 2),
        "has_breakthrough": has_breakthrough,
        "confirmation_days": confirmation_days,
        "confirmation_details": confirmation_details,
        "has_confirmation": has_confirmation,
        "status": "突破已确认" if has_confirmation else "突破未确认" if has_breakthrough else "无突破信号"
    }

    return {
        "has_confirmation": has_confirmation,
        "has_breakthrough": has_breakthrough,
        "details": details
    }
