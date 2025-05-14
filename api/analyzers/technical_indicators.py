"""
Technical Indicators module for calculating various technical indicators.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

def calculate_ma(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 30, 60]) -> pd.DataFrame:
    """
    Calculate Moving Averages for the given periods.
    
    Args:
        df: DataFrame containing price data with 'close' column
        periods: List of periods to calculate MA for
    
    Returns:
        DataFrame with additional MA columns
    """
    result_df = df.copy()
    
    for period in periods:
        result_df[f'ma{period}'] = result_df['close'].rolling(window=period).mean()
    
    return result_df

def calculate_ema(df: pd.DataFrame, periods: List[int] = [12, 26]) -> pd.DataFrame:
    """
    Calculate Exponential Moving Averages for the given periods.
    
    Args:
        df: DataFrame containing price data with 'close' column
        periods: List of periods to calculate EMA for
    
    Returns:
        DataFrame with additional EMA columns
    """
    result_df = df.copy()
    
    for period in periods:
        result_df[f'ema{period}'] = result_df['close'].ewm(span=period, adjust=False).mean()
    
    return result_df

def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df: DataFrame containing price data with 'close' column
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
    
    Returns:
        DataFrame with additional MACD columns
    """
    result_df = df.copy()
    
    # Calculate fast and slow EMAs
    fast_ema = result_df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = result_df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    result_df['macd'] = fast_ema - slow_ema
    
    # Calculate signal line
    result_df['macd_signal'] = result_df['macd'].ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    result_df['macd_hist'] = result_df['macd'] - result_df['macd_signal']
    
    return result_df

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate RSI (Relative Strength Index).
    
    Args:
        df: DataFrame containing price data with 'close' column
        period: Period for RSI calculation
    
    Returns:
        DataFrame with additional RSI column
    """
    result_df = df.copy()
    
    # Calculate price changes
    delta = result_df['close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    result_df['rsi'] = 100 - (100 / (1 + rs))
    
    return result_df

def calculate_kdj(df: pd.DataFrame, k_period: int = 9, d_period: int = 3, j_period: int = 3) -> pd.DataFrame:
    """
    Calculate KDJ indicator.
    
    Args:
        df: DataFrame containing price data with 'high', 'low', 'close' columns
        k_period: Period for K line
        d_period: Period for D line
        j_period: Period for J line
    
    Returns:
        DataFrame with additional KDJ columns
    """
    result_df = df.copy()
    
    # Calculate lowest low and highest high
    low_min = result_df['low'].rolling(window=k_period).min()
    high_max = result_df['high'].rolling(window=k_period).max()
    
    # Calculate RSV (Raw Stochastic Value)
    rsv = 100 * ((result_df['close'] - low_min) / (high_max - low_min))
    
    # Calculate K
    result_df['k'] = rsv.rolling(window=d_period).mean()
    
    # Calculate D
    result_df['d'] = result_df['k'].rolling(window=j_period).mean()
    
    # Calculate J
    result_df['j'] = 3 * result_df['k'] - 2 * result_df['d']
    
    return result_df

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame containing price data with 'close' column
        period: Period for moving average
        std_dev: Number of standard deviations for bands
    
    Returns:
        DataFrame with additional Bollinger Bands columns
    """
    result_df = df.copy()
    
    # Calculate middle band (SMA)
    result_df['bb_middle'] = result_df['close'].rolling(window=period).mean()
    
    # Calculate standard deviation
    rolling_std = result_df['close'].rolling(window=period).std()
    
    # Calculate upper and lower bands
    result_df['bb_upper'] = result_df['bb_middle'] + (rolling_std * std_dev)
    result_df['bb_lower'] = result_df['bb_middle'] - (rolling_std * std_dev)
    
    # Calculate bandwidth
    result_df['bb_bandwidth'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
    
    return result_df

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators.
    
    Args:
        df: DataFrame containing price data
    
    Returns:
        DataFrame with all technical indicators
    """
    if df.empty or 'close' not in df.columns:
        return df
    
    result_df = df.copy()
    
    # Calculate Moving Averages
    result_df = calculate_ma(result_df)
    
    # Calculate MACD
    result_df = calculate_macd(result_df)
    
    # Calculate RSI
    result_df = calculate_rsi(result_df)
    
    # Calculate KDJ
    if all(col in result_df.columns for col in ['high', 'low']):
        result_df = calculate_kdj(result_df)
    
    # Calculate Bollinger Bands
    result_df = calculate_bollinger_bands(result_df)
    
    return result_df
