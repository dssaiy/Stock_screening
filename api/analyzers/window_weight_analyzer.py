"""
Window Weight Analyzer module for handling window weights and combined scoring.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

def normalize_weights(weights: Dict[int, float]) -> Dict[int, float]:
    """
    Normalize weights so they sum to 1.0.
    
    Args:
        weights: Dictionary mapping window sizes to weights
    
    Returns:
        Dictionary with normalized weights
    """
    if not weights:
        return {}
    
    total = sum(weights.values())
    if total == 0:
        # If all weights are 0, assign equal weights
        equal_weight = 1.0 / len(weights)
        return {window: equal_weight for window in weights}
    
    return {window: weight / total for window, weight in weights.items()}

def calculate_weighted_score(
    window_results: Dict[int, Dict[str, Any]],
    weights: Dict[int, float],
    platform_windows: List[int]
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate a weighted score based on window results and weights.
    
    Args:
        window_results: Dictionary mapping window sizes to analysis results
        weights: Dictionary mapping window sizes to weights
        platform_windows: List of windows that are identified as platform periods
    
    Returns:
        Tuple of (weighted_score, details)
    """
    # Normalize weights
    normalized_weights = normalize_weights(weights)
    
    # If no weights provided, return 0 score
    if not normalized_weights:
        return 0.0, {
            "status": "无权重设置",
            "normalized_weights": {},
            "window_scores": {},
            "weighted_score": 0.0
        }
    
    # Calculate score for each window
    window_scores = {}
    for window, weight in normalized_weights.items():
        # Skip windows that don't have results
        if window not in window_results:
            window_scores[window] = 0.0
            continue
        
        # Base score: 1.0 if it's a platform window, 0.0 otherwise
        base_score = 1.0 if window in platform_windows else 0.0
        
        # Get result details for this window
        result = window_results[window]
        
        # Adjust score based on price analysis
        price_analysis = result.get("price_analysis", {})
        price_score = 0.0
        
        if price_analysis.get("status") == "符合条件":
            # Better scores for tighter price ranges and lower volatility
            box_range = price_analysis.get("box_range", 1.0)
            volatility = price_analysis.get("volatility", 0.1)
            ma_diff = price_analysis.get("ma_diff", 0.1)
            
            # Normalize values (lower is better)
            box_range_score = max(0, 1.0 - box_range)
            volatility_score = max(0, 1.0 - volatility * 10)  # Scale volatility
            ma_diff_score = max(0, 1.0 - ma_diff * 10)  # Scale MA diff
            
            # Combine price scores
            price_score = (box_range_score + volatility_score + ma_diff_score) / 3
        
        # Adjust score based on volume analysis
        volume_analysis = result.get("volume_analysis", {})
        volume_score = 0.0
        
        if volume_analysis.get("status") == "符合条件":
            # Better scores for stable and decreasing volume
            volume_change = volume_analysis.get("volume_change_ratio", 1.0)
            volume_stability = volume_analysis.get("volume_stability", 1.0)
            
            # Normalize values (lower is better for change, higher for stability)
            volume_change_score = max(0, 1.0 - volume_change)
            volume_stability_score = max(0, 1.0 - volume_stability)
            
            # Combine volume scores
            volume_score = (volume_change_score + volume_stability_score) / 2
        
        # Adjust score based on breakthrough
        breakthrough = result.get("breakthrough", {})
        breakthrough_score = 0.0
        
        if breakthrough.get("status") == "有成交量突破":
            # Better scores for stronger volume increases
            volume_increase = breakthrough.get("volume_increase_ratio", 1.0)
            breakthrough_score = min(1.0, volume_increase / 3.0)  # Cap at 1.0
        
        # Combine all scores
        combined_score = base_score
        if base_score > 0:
            # Only adjust score if it's a platform window
            combined_score = (base_score * 0.5) + (price_score * 0.25) + (volume_score * 0.15) + (breakthrough_score * 0.1)
        
        window_scores[window] = combined_score
    
    # Calculate weighted score
    weighted_score = sum(window_scores.get(window, 0.0) * normalized_weights.get(window, 0.0) 
                         for window in normalized_weights)
    
    # Prepare details
    details = {
        "status": "已计算加权得分",
        "normalized_weights": normalized_weights,
        "window_scores": window_scores,
        "weighted_score": weighted_score
    }
    
    return weighted_score, details

def apply_window_weights(
    analysis_result: Dict[str, Any],
    window_weights: Dict[int, float]
) -> Dict[str, Any]:
    """
    Apply window weights to analysis results.
    
    Args:
        analysis_result: Result from analyze_stock function
        window_weights: Dictionary mapping window sizes to weights
    
    Returns:
        Updated analysis result with weighted scores
    """
    # If window weights are not provided or empty, return original result
    if not window_weights:
        analysis_result["weighted_score"] = 0.0
        analysis_result["weight_details"] = {
            "status": "未使用窗口权重",
            "normalized_weights": {},
            "window_scores": {},
            "weighted_score": 0.0
        }
        return analysis_result
    
    # Get window results and platform windows
    window_results = analysis_result.get("details", {})
    platform_windows = analysis_result.get("platform_windows", [])
    
    # Calculate weighted score
    weighted_score, weight_details = calculate_weighted_score(
        window_results, window_weights, platform_windows
    )
    
    # Update analysis result
    analysis_result["weighted_score"] = weighted_score
    analysis_result["weight_details"] = weight_details
    
    return analysis_result
