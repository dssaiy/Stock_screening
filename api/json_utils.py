"""
JSON utilities for handling special values like NaN and Infinity.
"""
import json
import math
from typing import Any, Dict, List, Union


def sanitize_float_for_json(value: Any) -> Any:
    """
    Sanitize float values for JSON serialization.
    Converts NaN, Infinity, and -Infinity to None.

    Args:
        value: The value to sanitize

    Returns:
        The sanitized value
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    elif isinstance(value, dict):
        return {k: sanitize_float_for_json(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [sanitize_float_for_json(item) for item in value]
    return value


def sanitize_kline_data(kline_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sanitize K-line data for JSON serialization.
    Specifically handles numeric fields that might contain NaN or Infinity.

    Args:
        kline_data: List of K-line data points

    Returns:
        Sanitized K-line data
    """
    if not kline_data:
        return []

    numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'turn',
                      'preclose', 'pctChg', 'peTTM', 'pbMRQ']

    sanitized_data = []
    for point in kline_data:
        sanitized_point = {}
        for key, value in point.items():
            if key in numeric_fields and isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    sanitized_point[key] = None
                else:
                    sanitized_point[key] = value
            else:
                sanitized_point[key] = value
        sanitized_data.append(sanitized_point)

    return sanitized_data


def sanitize_task_result(task_result: Union[List[Dict[str, Any]], None]) -> Union[List[Dict[str, Any]], None]:
    """
    Sanitize task result for JSON serialization.

    Args:
        task_result: The task result to sanitize

    Returns:
        The sanitized task result
    """
    if task_result is None:
        return None

    sanitized_result = []
    for stock in task_result:
        sanitized_stock = sanitize_float_for_json(stock)

        # 特别处理K线数据
        if 'kline_data' in sanitized_stock and sanitized_stock['kline_data']:
            sanitized_stock['kline_data'] = sanitize_kline_data(
                sanitized_stock['kline_data'])

        sanitized_result.append(sanitized_stock)

    return sanitized_result
