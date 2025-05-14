"""
Box (Consolidation) Pattern Detector Module

This module provides functions to detect box/consolidation patterns in stock price data
using more advanced techniques than simple range checks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from scipy.signal import argrelextrema
from scipy.stats import linregress


def detect_local_extrema(prices: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect local maxima and minima in price data.

    Args:
        prices: Array of price data
        order: How many points on each side to use for the comparison

    Returns:
        Tuple of (maxima_indices, minima_indices)
    """
    # Find local maxima
    maxima_indices = argrelextrema(prices, np.greater, order=order)[0]

    # Find local minima
    minima_indices = argrelextrema(prices, np.less, order=order)[0]

    return maxima_indices, minima_indices


def identify_support_resistance(df: pd.DataFrame, window: int,
                                extrema_order: int = 5) -> Dict[str, Any]:
    """
    Identify support and resistance levels in the given window.

    Args:
        df: DataFrame with price data
        window: Window size to analyze
        extrema_order: Order parameter for extrema detection

    Returns:
        Dict with support and resistance information
    """
    if len(df) < window:
        return {
            "status": "insufficient_data",
            "support_levels": [],
            "resistance_levels": [],
            "support_strength": 0,
            "resistance_strength": 0,
            "is_box_pattern": False
        }

    # Get recent data for the window
    recent_df = df.iloc[-window:].copy()

    # Get high and low prices
    highs = recent_df['high'].values
    lows = recent_df['low'].values

    # Find local maxima and minima
    high_maxima, _ = detect_local_extrema(highs, order=extrema_order)
    _, low_minima = detect_local_extrema(lows, order=extrema_order)

    # Extract the price values at these points
    resistance_points = highs[high_maxima] if len(high_maxima) > 0 else []
    support_points = lows[low_minima] if len(low_minima) > 0 else []

    # Cluster close resistance and support levels
    resistance_levels = cluster_price_levels(resistance_points)
    support_levels = cluster_price_levels(support_points)

    # Calculate strength of support and resistance (number of touches)
    support_strength = calculate_level_strength(lows, support_levels)
    resistance_strength = calculate_level_strength(highs, resistance_levels)

    # Determine if this forms a box pattern
    is_box_pattern = False
    box_quality = np.nan

    if len(support_levels) > 0 and len(resistance_levels) > 0:
        # Calculate the average of the strongest support and resistance levels
        main_support = support_levels[0] if len(support_levels) > 0 else np.nan
        main_resistance = resistance_levels[0] if len(
            resistance_levels) > 0 else np.nan

        if not np.isnan(main_support) and not np.isnan(main_resistance):
            # Calculate box height as percentage of support level
            box_height_pct = (main_resistance - main_support) / main_support

            # Check if price is contained within the box for most of the window
            prices_in_box = ((recent_df['close'] >= main_support * 0.98) &
                             (recent_df['close'] <= main_resistance * 1.02)).mean()

            # Calculate linear regression of closing prices to check for horizontal trend
            x = np.arange(len(recent_df))
            y = recent_df['close'].values
            slope, _, r_value, _, _ = linregress(x, y)

            # Normalize slope by average price
            avg_price = recent_df['close'].mean()
            norm_slope = abs(slope) / avg_price

            # Calculate box quality score (higher is better)
            # A good box has:
            # 1. Strong support and resistance (multiple touches)
            # 2. Most prices contained within the box
            # 3. A relatively flat trend (low absolute slope)
            # 4. Reasonable box height (not too narrow, not too wide)

            # Box quality factors
            support_factor = min(support_strength / 2, 1.0)  # Cap at 1.0
            resistance_factor = min(resistance_strength / 2, 1.0)  # Cap at 1.0
            containment_factor = prices_in_box
            # Lower slope is better
            trend_factor = max(0, 1 - norm_slope * 100)

            # Box height factor - penalize if too narrow or too wide
            # Ideal range: 3% to 20%
            height_factor = 0.0
            if 0.03 <= box_height_pct <= 0.2:
                height_factor = 1.0
            elif box_height_pct < 0.03:
                height_factor = box_height_pct / 0.03  # Gradually reduce score for narrow boxes
            else:  # box_height_pct > 0.2
                # Gradually reduce score for wide boxes
                height_factor = max(0, 1 - (box_height_pct - 0.2) / 0.3)

            # Calculate overall box quality
            box_quality = (support_factor * 0.25 +
                           resistance_factor * 0.25 +
                           containment_factor * 0.2 +
                           trend_factor * 0.15 +
                           height_factor * 0.15)

            # Determine if this is a box pattern based on quality score
            is_box_pattern = box_quality >= 0.6  # Threshold for box pattern

    return {
        "status": "analyzed",
        "support_levels": support_levels.tolist() if isinstance(support_levels, np.ndarray) else support_levels,
        "resistance_levels": resistance_levels.tolist() if isinstance(resistance_levels, np.ndarray) else resistance_levels,
        "support_strength": support_strength,
        "resistance_strength": resistance_strength,
        "is_box_pattern": is_box_pattern,
        "box_quality": round(box_quality, 2) if not np.isnan(box_quality) else 0.0
    }


def cluster_price_levels(price_points: np.ndarray, tolerance: float = 0.02) -> np.ndarray:
    """
    Cluster close price levels together.

    Args:
        price_points: Array of price points
        tolerance: Relative tolerance for clustering (as percentage)

    Returns:
        Array of clustered price levels, sorted by strength (number of points in cluster)
    """
    if len(price_points) == 0:
        return np.array([])

    # Sort price points
    sorted_points = np.sort(price_points)

    # Initialize clusters
    clusters = []
    current_cluster = [sorted_points[0]]

    # Cluster points
    for i in range(1, len(sorted_points)):
        # Calculate relative difference
        relative_diff = abs(
            sorted_points[i] - current_cluster[0]) / current_cluster[0]

        if relative_diff <= tolerance:
            # Add to current cluster
            current_cluster.append(sorted_points[i])
        else:
            # Start a new cluster
            clusters.append(current_cluster)
            current_cluster = [sorted_points[i]]

    # Add the last cluster
    clusters.append(current_cluster)

    # Calculate average price for each cluster and count points
    cluster_info = [(np.mean(cluster), len(cluster)) for cluster in clusters]

    # Sort clusters by strength (number of points) in descending order
    sorted_clusters = sorted(cluster_info, key=lambda x: x[1], reverse=True)

    # Return just the price levels
    return np.array([price for price, _ in sorted_clusters])


def calculate_level_strength(prices: np.ndarray, levels: np.ndarray,
                             tolerance: float = 0.01) -> int:
    """
    Calculate the strength of price levels by counting how many times prices approach them.

    Args:
        prices: Array of price data
        levels: Array of price levels
        tolerance: Relative tolerance for considering a price to be at a level

    Returns:
        Strength score (higher means stronger levels)
    """
    if len(levels) == 0:
        return 0

    # Focus on the strongest level (first one)
    main_level = levels[0]

    # Count how many times prices come close to the level
    touches = 0
    for price in prices:
        relative_diff = abs(price - main_level) / main_level
        if relative_diff <= tolerance:
            touches += 1

    return touches


def analyze_box_pattern(df: pd.DataFrame, window: int) -> Dict[str, Any]:
    """
    Analyze if the stock is forming a box/consolidation pattern.

    Args:
        df: DataFrame with price data
        window: Window size to analyze

    Returns:
        Dict with box pattern analysis results
    """
    # Identify support and resistance
    sr_analysis = identify_support_resistance(df, window)

    # Calculate additional box pattern metrics
    if len(df) >= window:
        recent_df = df.iloc[-window:].copy()

        # Calculate price volatility within the box
        volatility = recent_df['close'].pct_change().std()

        # Calculate volume trend
        if 'volume' in recent_df.columns:
            # Check if volume is decreasing or stable
            volume_trend = recent_df['volume'].pct_change().mean()
            is_volume_decreasing = volume_trend < 0

            # Calculate volume volatility
            volume_volatility = recent_df['volume'].pct_change().std()
        else:
            is_volume_decreasing = False
            volume_volatility = np.nan

        # Update the analysis with additional metrics
        sr_analysis.update({
            "volatility": round(volatility, 4) if not np.isnan(volatility) else None,
            "is_volume_decreasing": is_volume_decreasing,
            "volume_volatility": round(volume_volatility, 4) if not np.isnan(volume_volatility) else None
        })

    return sr_analysis


def check_box_pattern(df: pd.DataFrame, window: int,
                      box_quality_threshold: float = 0.6,
                      volatility_threshold: float = 0.09) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a stock is forming a box/consolidation pattern.

    Args:
        df: DataFrame with price data
        window: Window size to analyze
        box_quality_threshold: Minimum quality score for a valid box pattern
        volatility_threshold: Maximum allowed volatility

    Returns:
        Tuple of (is_box_pattern, details)
    """
    # Analyze box pattern
    analysis = analyze_box_pattern(df, window)

    # Check if it's a valid box pattern
    is_box = (
        analysis.get("is_box_pattern", False) and
        analysis.get("box_quality", 0) >= box_quality_threshold and
        (analysis.get("volatility", float('inf')) <=
         volatility_threshold if analysis.get("volatility") is not None else False)
    )

    # Add status message
    if not is_box:
        if not analysis.get("is_box_pattern", False):
            analysis["status"] = "不是箱体形态"
        elif analysis.get("box_quality", 0) < box_quality_threshold:
            analysis["status"] = "箱体质量不足"
        elif analysis.get("volatility", float('inf')) > volatility_threshold:
            analysis["status"] = "波动性过高"
        else:
            analysis["status"] = "不符合箱体条件"
    else:
        analysis["status"] = "符合箱体条件"

    return is_box, analysis
