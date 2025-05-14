"""
Decline Analyzer Module

This module provides functions to analyze the decline characteristics of a stock,
including decline speed, concentration, and volatility.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def analyze_decline_speed(df: pd.DataFrame,
                          lookback_days: int = 365,
                          decline_period_days: int = 180,
                          decline_threshold: float = 0.3,
                          rapid_decline_days: int = 30,
                          rapid_decline_threshold: float = 0.15) -> Dict[str, Any]:
    """
    Analyze the decline speed and characteristics of a stock.

    Args:
        df: DataFrame containing stock price data
        lookback_days: Number of days to look back for finding the high point
        decline_period_days: Number of days within which the decline should have occurred
        decline_threshold: Minimum decline percentage from high to be considered at low position
        rapid_decline_days: Number of days to define a rapid decline
        rapid_decline_threshold: Minimum decline percentage within rapid_decline_days to be considered rapid

    Returns:
        Dict containing decline analysis results
    """
    try:
        if len(df) < 60:  # Require at least 60 days of data
            return {
                "status": "insufficient_data",
                "is_low_position": False,
                "is_rapid_decline": False,
                "details": {
                    "data_points": len(df),
                    "required_points": 60
                }
            }

        # Sort DataFrame by date
        df = df.sort_values('date')

        # Calculate lookback period
        if len(df) <= lookback_days:
            lookback_df = df.copy()
        else:
            lookback_df = df.iloc[-lookback_days:].copy()

        # Find the highest price in the lookback period
        max_price = lookback_df['high'].max()
        max_idx = lookback_df[lookback_df['high'] == max_price].index[0]
        max_date = lookback_df.loc[max_idx, 'date']

        # Get data after the high point
        after_high_df = lookback_df.loc[max_idx:].copy()

        # If there's not enough data after high point, return early
        if len(after_high_df) < 20:
            return {
                "status": "insufficient_after_high_data",
                "is_low_position": False,
                "is_rapid_decline": False,
                "details": {
                    "high_price": max_price,
                    "high_date": max_date,
                    "data_points_after_high": len(after_high_df)
                }
            }

        # Find the lowest price after the high point
        min_price = after_high_df['low'].min()
        min_idx = after_high_df[after_high_df['low'] == min_price].index[0]
        min_date = after_high_df.loc[min_idx, 'date']

        # Calculate decline percentage
        decline_percentage = (max_price - min_price) / max_price

        # Check if the decline occurred within the specified period
        decline_period_satisfied = True
        if decline_period_days > 0:
            high_date = pd.to_datetime(max_date)
            low_date = pd.to_datetime(min_date)
            days_between = (low_date - high_date).days
            decline_period_satisfied = days_between <= decline_period_days

        # Determine if it's a low position
        is_low_position = decline_percentage >= decline_threshold and decline_period_satisfied

        # Calculate decline speed metrics
        decline_days = len(after_high_df.loc[max_idx:min_idx])
        daily_decline_rate = decline_percentage / \
            decline_days if decline_days > 0 else 0

        # Check for rapid decline periods
        rapid_decline_detected = False
        max_rapid_decline = 0
        rapid_decline_start_date = None
        rapid_decline_end_date = None

        # Scan for the most rapid decline period
        for i in range(len(after_high_df) - rapid_decline_days + 1):
            window = after_high_df.iloc[i:i+rapid_decline_days]
            window_high = window['high'].max()
            window_low = window['low'].min()
            window_decline = (window_high - window_low) / window_high

            if window_decline > max_rapid_decline:
                max_rapid_decline = window_decline
                rapid_decline_start_date = window.iloc[0]['date']
                rapid_decline_end_date = window.iloc[-1]['date']

        # Determine if there was a rapid decline
        is_rapid_decline = max_rapid_decline >= rapid_decline_threshold

        # Calculate decline concentration (percentage of total decline that occurred in the rapid period)
        decline_concentration = max_rapid_decline / \
            decline_percentage if decline_percentage > 0 else 0

        # Calculate decline volatility (standard deviation of daily returns during decline)
        decline_volatility = after_high_df.loc[max_idx:min_idx]['close'].pct_change(
        ).std()

        # Prepare result
        result = {
            "status": "analyzed",
            "is_low_position": is_low_position,
            "is_rapid_decline": is_rapid_decline,
            "details": {
                "high_price": float(max_price),
                "high_date": max_date,
                "low_price": float(min_price),
                "low_date": min_date,
                "decline_percentage": float(decline_percentage),
                "decline_days": int(decline_days),
                "daily_decline_rate": float(daily_decline_rate),
                "max_rapid_decline": float(max_rapid_decline),
                "rapid_decline_start_date": rapid_decline_start_date,
                "rapid_decline_end_date": rapid_decline_end_date,
                "decline_concentration": float(decline_concentration),
                "decline_volatility": float(decline_volatility),
                "decline_period_satisfied": decline_period_satisfied
            }
        }

        return result

    except Exception as e:
        logger.error(f"Error in analyze_decline_speed: {str(e)}")
        return {
            "status": "error",
            "is_low_position": False,
            "is_rapid_decline": False,
            "details": {
                "error": str(e)
            }
        }


def check_decline_pattern(df: pd.DataFrame,
                          lookback_days: int = 365,
                          decline_period_days: int = 180,
                          decline_threshold: float = 0.3,
                          rapid_decline_days: int = 30,
                          rapid_decline_threshold: float = 0.15) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a stock has a significant decline pattern with rapid decline followed by consolidation.

    Args:
        df: DataFrame containing stock price data
        lookback_days: Number of days to look back for finding the high point
        decline_period_days: Number of days within which the decline should have occurred
        decline_threshold: Minimum decline percentage from high to be considered at low position
        rapid_decline_days: Number of days to define a rapid decline
        rapid_decline_threshold: Minimum decline percentage within rapid_decline_days to be considered rapid

    Returns:
        Tuple of (has_decline_pattern, details)
    """
    # Analyze decline speed
    analysis = analyze_decline_speed(
        df, lookback_days, decline_period_days,
        decline_threshold, rapid_decline_days, rapid_decline_threshold
    )

    # Check if it has the desired decline pattern
    has_decline_pattern = (
        analysis.get("status") == "analyzed" and
        analysis.get("is_low_position", False) and
        analysis.get("is_rapid_decline", True)  # 修改为True，表示需要满足快速下跌条件
    )

    return has_decline_pattern, analysis
