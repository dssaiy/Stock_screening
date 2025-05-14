# -*- coding: utf-8 -*-
"""
Stock Platform Consolidation Period Selector Module

This module identifies stocks potentially in a platform consolidation phase
based on price range, moving average convergence, and volatility criteria.
It uses the baostock library to fetch historical K-line data for Chinese A-shares.
"""

import baostock as bs
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time  # Import time for retry delay
import os
import sys
import platform
import multiprocessing
from contextlib import redirect_stdout
from tqdm import tqdm  # 进度条
import colorama  # 命令行颜色
from colorama import Fore, Style  # 颜色和样式

# 初始化 colorama
colorama.init()

# Windows 多进程支持
if sys.platform == 'win32':
    # 设置多进程启动方法为 'spawn'
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过，忽略错误
        pass

# Baostock login/logout context manager


def baostock_login():
    """Login to Baostock silently (without printing to stdout)"""
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            bs.login()


def baostock_logout():
    """Logout from Baostock silently (without printing to stdout)"""
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            bs.logout()


def baostock_relogin():
    """Logout and login again to Baostock silently"""
    baostock_logout()
    baostock_login()


class baostock_login_context:
    """Context manager for Baostock login/logout"""

    def __enter__(self):
        baostock_login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        baostock_logout()

# =================================
# Configuration
# =================================


# Default configuration, can be overridden by API request
# 基于安记食品平台期分析的最佳参数组合
DEFAULT_CONFIG = {
    "windows": [20, 30, 60],            # Platform period time windows (days)
    # Max amplitude threshold (high-low range / low)
    "box_threshold": 0.5,
    # Moving Average convergence threshold (std / avg)
    "ma_diff_threshold": 0.03,
    # Daily return volatility threshold (std dev)
    "volatility_threshold": 0.04,
    # 不设置默认值，使用 Pydantic 模型中的默认值（5）
    "retry_attempts": 2,                # Number of retries for failed data requests
    "retry_delay": 1                    # Delay between retries in seconds
}

# =================================
# Feature Calculation Functions
# =================================


def compute_dynamic_ma_features(df: pd.DataFrame, window: int):
    """
    Calculates moving averages (MA) with periods dynamically chosen based on the window length.
    Returns the standard deviation and average of these MAs.

    Args:
        df (pd.DataFrame): DataFrame containing at least the 'close' price column.
        window (int): The time window length in days.

    Returns:
        tuple: (standard deviation of MAs, average of MAs) or (NaN, NaN) if calculation fails.
    """
    if len(df) < window:
        return np.nan, np.nan  # Not enough data for the window

    # Dynamically select MA periods based on the analysis window
    if window <= 30:
        periods = [5, 10, 20]
    elif window <= 60:
        periods = [10, 20, 30]
    else:  # window > 60
        periods = [20, 30, 60]

    # Ensure the DataFrame is long enough for the longest MA period required
    min_required_length = max(periods)
    if len(df) < min_required_length:
        # If not enough data even for the base MAs, return NaN
        # This check is important if df passed is already sliced
        return np.nan, np.nan

    mas = []
    valid_mas = True
    for p in periods:
        # Calculate rolling mean, ensure enough data points exist for the period 'p'
        if len(df) >= p:
            ma = df['close'].rolling(p).mean().iloc[-1]
            if pd.isna(ma):  # Check if MA calculation resulted in NaN
                valid_mas = False
                break
            mas.append(ma)
        else:
            valid_mas = False  # Not enough data points for this MA period
            break

    if not valid_mas or not mas:  # If any MA failed or list is empty
        return np.nan, np.nan

    # Calculate standard deviation and average of the computed MAs
    ma_std = np.std(mas)
    ma_avg = np.mean(mas)

    # Avoid division by zero or near-zero average
    if ma_avg is None or np.isclose(ma_avg, 0):
        return np.nan, np.nan

    return ma_std, ma_avg

# =================================
# Platform Period Identification
# =================================


def is_platform_window(df: pd.DataFrame,
                       window: int,
                       box_threshold: float,
                       ma_diff_threshold: float,
                       volatility_threshold: float) -> tuple:
    """
    Determines if the stock is in a platform consolidation period within the specified window.
    Checks for: 1. Narrow price range (low amplitude)
                  2. Converging moving averages
                  3. Low daily return volatility

    Args:
        df (pd.DataFrame): DataFrame with historical K-line data (needs 'high', 'low', 'close').
                           Should contain data *up to* the evaluation date.
        window (int): The time window (number of trading days) to analyze.
        box_threshold (float): Maximum allowed amplitude ((max_high - min_low) / min_low).
        ma_diff_threshold (float): Maximum allowed MA dispersion (std_dev / average).
        volatility_threshold (float): Maximum allowed standard deviation of daily returns.

    Returns:
        tuple: (is_platform, details_dict) where:
            - is_platform (bool): True if the stock meets platform criteria, False otherwise.
            - details_dict (dict): Dictionary with detailed information about the check.
    """
    # 初始化详细信息字典
    details = {
        "window": window,
        "data_points": 0,
        "price_high": np.nan,
        "price_low": np.nan,
        "price_range": np.nan,
        "box_condition": False,
        "ma_std": np.nan,
        "ma_avg": np.nan,
        "ma_convergence_ratio": np.nan,
        "ma_condition": False,
        "volatility": np.nan,
        "vol_condition": False,
        "status": "未检查"
    }

    # Ensure enough data exists for the specified window
    if len(df) < window:
        details["status"] = "数据不足"
        details["data_points"] = len(df)
        return False, details

    # Slice the DataFrame for the recent 'window' days
    recent_df = df.iloc[-window:]
    details["data_points"] = len(recent_df)

    # --- 1. Amplitude Check (Box Range) ---
    min_low = recent_df['low'].min()
    max_high = recent_df['high'].max()

    details["price_low"] = min_low
    details["price_high"] = max_high

    # Avoid division by zero if min_low is 0 or NaN
    if min_low is None or np.isclose(min_low, 0) or pd.isna(min_low) or pd.isna(max_high):
        details["status"] = "无效价格数据"
        return False, details

    box_range = (max_high - min_low) / min_low
    details["price_range"] = box_range
    details["box_condition"] = box_range <= box_threshold

    if not details["box_condition"]:
        details["status"] = "箱体范围过大"
        return False, details

    # --- 2. Moving Average Convergence Check ---
    # Pass the *full available history* up to the end date for MA calculation continuity
    ma_std, ma_avg = compute_dynamic_ma_features(df, window)

    details["ma_std"] = ma_std
    details["ma_avg"] = ma_avg

    # Check if MA calculation was successful and avg is valid
    if pd.isna(ma_std) or pd.isna(ma_avg) or np.isclose(ma_avg, 0):
        details["status"] = "MA计算失败"
        return False, details

    ma_convergence_ratio = ma_std / ma_avg
    details["ma_convergence_ratio"] = ma_convergence_ratio
    details["ma_condition"] = ma_convergence_ratio <= ma_diff_threshold

    if not details["ma_condition"]:
        details["status"] = "MA收敛度不足"
        return False, details

    # --- 3. Volatility Check (Standard Deviation of Daily Returns) ---
    # Calculate percentage change on the 'close' price for the window
    daily_returns = recent_df['close'].pct_change().dropna()

    # Need at least 2 returns to calculate standard deviation
    if len(daily_returns) < 2:
        details["status"] = "收益率数据不足"
        return False, details  # Cannot calculate volatility with less than 2 returns

    volatility = daily_returns.std()
    details["volatility"] = volatility

    if pd.isna(volatility):  # Check if std calculation resulted in NaN
        details["status"] = "波动率计算失败"
        return False, details

    details["vol_condition"] = volatility <= volatility_threshold
    if not details["vol_condition"]:
        details["status"] = "波动率过高"
        return False, details

    # If all checks passed
    details["status"] = "符合条件"
    return True, details

# =================================
# Data Fetching
# =================================


def fetch_kline_data(code: str, start_date: str, end_date: str,
                     retry_attempts: int,
                     retry_delay: int) -> pd.DataFrame:
    """
    Fetches daily K-line data for a specific stock code using the baostock API.
    Includes robust retry logic based on the DataManager example.

    Args:
        code (str): The stock code (e.g., 'sh.600000').
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        retry_attempts (int): Maximum number of times to retry on failure.
        retry_delay (int): Delay in seconds between retries.

    Returns:
        pd.DataFrame: A DataFrame containing the K-line data, or an empty DataFrame on failure.
                      Columns include: date, open, high, low, close, volume, turn,
                                       preclose, pctChg, peTTM, pbMRQ.
    """
    # 使用无限循环和计数器来控制重试，更接近示例代码中的实现
    retries = 0

    while True:  # 无限循环，直到成功或达到最大重试次数
        # 使用彩色输出，但只在调试时打印
        # print(f"{Fore.CYAN}尝试 {retries+1}/{retry_attempts}: 获取 {code} 从 {start_date} 到 {end_date} 的数据{Style.RESET_ALL}")
        try:
            # Query historical K-line data
            # adjustflag='3' for no adjustment (raw data)
            # adjustflag='2' for forward adjustment
            # adjustflag='1' for backward adjustment
            # Using '2' (forward) is common for analysis comparing across time
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume,turn,preclose,pctChg,peTTM,pbMRQ",
                start_date=start_date,
                end_date=end_date,  # Use the provided end_date
                frequency="d",     # Daily frequency
                adjustflag="2"     # Forward adjusted prices
            )

            # Check for API errors
            if rs.error_code != '0':
                retries += 1
                print(
                    f"{Fore.YELLOW}尝试 {retries}/{retry_attempts}: Baostock 查询失败 {code}. 错误: {rs.error_msg}{Style.RESET_ALL}")

                # 检查是否达到最大重试次数
                if retries >= retry_attempts:
                    print(
                        f"{Fore.RED}在 {retry_attempts} 次尝试后获取 {code} 数据失败{Style.RESET_ALL}")
                    return pd.DataFrame()  # 达到最大重试次数，返回空数据框

                # 重试前重新登录
                time.sleep(retry_delay * (1 + retries * 0.5))  # 增加延迟时间，实现指数退避
                print(f"{Fore.YELLOW}重新登录 Baostock...{Style.RESET_ALL}")
                baostock_relogin()
                continue  # 继续下一次重试

            # Process the data if query was successful
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)

                # Convert data types - crucial for calculations
                numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                                'turn', 'preclose', 'pctChg', 'peTTM', 'pbMRQ']
                for col in numeric_cols:
                    # Use errors='coerce' to turn unparseable values into NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Convert date to datetime objects (optional, but good practice)
                df['date'] = pd.to_datetime(df['date'])

                # Drop rows where essential price data is missing (NaN) after conversion
                df.dropna(subset=['open', 'high',
                          'low', 'close'], inplace=True)

                # Sort by date just in case API doesn't guarantee order
                df.sort_values(by='date', inplace=True)
                # Reset index after sorting/dropping
                df.reset_index(drop=True, inplace=True)

                # 成功获取数据
                print(f"{Fore.GREEN}成功获取 {code} 的 {len(df)} 行数据{Style.RESET_ALL}")
                return df  # Return the DataFrame on success
            else:
                # 查询成功但没有数据
                print(
                    f"{Fore.YELLOW}在 {start_date} 和 {end_date} 之间没有 {code} 的数据{Style.RESET_ALL}")
                return pd.DataFrame()  # Return empty DF if no data rows

        except Exception as e:
            retries += 1
            print(
                f"{Fore.RED}尝试 {retries}/{retry_attempts}: 获取 {code} 数据时发生异常: {e}{Style.RESET_ALL}")

            # 检查是否达到最大重试次数
            if retries >= retry_attempts:
                print(
                    f"{Fore.RED}在 {retry_attempts} 次尝试后获取 {code} 数据失败{Style.RESET_ALL}")
                return pd.DataFrame()  # 达到最大重试次数，返回空数据框

            # 重试前重新登录
            time.sleep(retry_delay * (1 + retries * 0.5))  # 增加延迟时间，实现指数退避
            print(f"{Fore.YELLOW}重新登录 Baostock...{Style.RESET_ALL}")
            baostock_relogin()

# =================================
# Core Scanning Function
# =================================


def scan_platform_stocks(stock_list: list[dict], config: dict = None) -> list[dict]:
    """
    Scans a list of stocks concurrently to find those meeting platform criteria
    in any of the specified window lengths.

    Args:
        stock_list (list[dict]): A list of dictionaries, each containing
                                 {'code': str, 'name': str}.
        config (dict, optional): A dictionary to override DEFAULT_CONFIG parameters.

    Returns:
        list[dict]: A list of dictionaries for stocks that match the criteria.
                    Each dictionary contains:
                    'code': Stock code.
                    'name': Stock name.
                    'platform_flags': Dict indicating which windows met criteria (e.g., {30: True, 60: False}).
                    'kline_data': DataFrame containing the K-line data for the max_window period.
                                  (Note: Returning DataFrame directly, conversion happens in API layer)
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    # 使用用户配置的 max_workers 值，如果没有则使用默认值 5
    max_workers = cfg.get('max_workers', 5)  # 默认值与 Pydantic 模型保持一致
    cfg['max_workers'] = max_workers  # 确保 cfg 中有 max_workers 键
    print(f"Using max_workers={max_workers} for Baostock connections")

    # Determine date range needed based on the longest window
    max_window = max(cfg['windows'])
    # Add buffer days (~50%) for MA calculations needing prior data
    # Ensure enough buffer for longest MA period
    buffer_days = int(max_window * 0.5) + 20
    end_date = datetime.today().strftime('%Y-%m-%d')
    # Calculate start date considering the max window and buffer
    start_date_obj = datetime.today() - timedelta(days=max_window + buffer_days)
    start_date = start_date_obj.strftime('%Y-%m-%d')

    results = []

    # 打印扫描信息
    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
    print(f"{Fore.CYAN}开始平台期选股扫描{Style.RESET_ALL}")
    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}扫描参数:{Style.RESET_ALL}")
    print(f"  - 时间窗口: {Fore.GREEN}{cfg['windows']}{Style.RESET_ALL}")
    print(f"  - 箱体阈值: {Fore.GREEN}{cfg['box_threshold']}{Style.RESET_ALL}")
    print(
        f"  - MA收敛阈值: {Fore.GREEN}{cfg['ma_diff_threshold']}{Style.RESET_ALL}")
    print(
        f"  - 波动率阈值: {Fore.GREEN}{cfg['volatility_threshold']}{Style.RESET_ALL}")
    print(f"  - 并发数: {Fore.GREEN}{max_workers}{Style.RESET_ALL}")
    print(
        f"  - 日期范围: {Fore.GREEN}{start_date}{Style.RESET_ALL} 至 {Fore.GREEN}{end_date}{Style.RESET_ALL}")
    print(f"  - 股票数量: {Fore.GREEN}{len(stock_list)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")

    # 尝试使用 ProcessPoolExecutor，如果失败则回退到 ThreadPoolExecutor
    try:
        # Use ProcessPoolExecutor for concurrent data fetching
        # Each process will have its own Baostock connection
        executor_class = ProcessPoolExecutor
        print(f"{Fore.GREEN}使用 ProcessPoolExecutor 进行并发数据获取{Style.RESET_ALL}")
    except Exception as e:
        # 如果 ProcessPoolExecutor 初始化失败，回退到 ThreadPoolExecutor
        print(f"{Fore.RED}ProcessPoolExecutor 初始化失败: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}回退到 ThreadPoolExecutor{Style.RESET_ALL}")
        executor_class = ThreadPoolExecutor

    # 确保 Baostock 已登录
    print(f"{Fore.CYAN}确保主进程已登录 Baostock...{Style.RESET_ALL}")
    try:
        baostock_logout()  # 先登出，确保清理之前的连接
        baostock_login()   # 重新登录
    except Exception as e:
        print(f"{Fore.RED}主进程登录 Baostock 失败: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}尝试继续执行，子进程将自行登录{Style.RESET_ALL}")

    # 创建进度条
    print(f"{Fore.CYAN}开始获取股票数据...{Style.RESET_ALL}")

    with executor_class(max_workers=cfg['max_workers'], initializer=baostock_login) as executor:
        # Submit tasks: fetch data for each stock
        future_to_stock = {
            executor.submit(fetch_kline_data, s['code'], start_date, end_date,
                            cfg['retry_attempts'], cfg['retry_delay']): s
            for s in stock_list
        }

        # 创建进度条
        pbar = tqdm(total=len(future_to_stock), desc="获取股票数据",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        # 计数器
        success_count = 0
        empty_count = 0
        error_count = 0
        platform_count = 0

        # Process completed futures as they finish
        for future in as_completed(future_to_stock):
            stock_info = future_to_stock[future]
            stock_code = stock_info['code']
            try:
                df = future.result()  # Get the DataFrame result from the future
                success_count += 1

                # Skip if data fetching failed or returned empty DataFrame
                if df is None or df.empty:
                    empty_count += 1
                    pbar.set_postfix(
                        成功=success_count, 空数据=empty_count, 错误=error_count, 平台期=platform_count)
                    pbar.update(1)
                    continue

                # Check platform conditions for each specified window
                flags = {}
                is_platform_in_any_window = False
                window_details = {}  # 存储每个窗口的详细信息

                for w in cfg['windows']:
                    # Ensure enough data exists *before* calling is_platform_window
                    # Need at least 'w' days for the check itself
                    if len(df) >= w:
                        # 直接调用平台期检测函数，不需要预先切片数据

                        # 调用平台期检测函数，获取详细信息
                        platform_status, details = is_platform_window(df, w,
                                                                      cfg['box_threshold'],
                                                                      cfg['ma_diff_threshold'],
                                                                      cfg['volatility_threshold'])
                        flags[w] = platform_status

                        # 格式化详细信息，用于显示
                        formatted_details = {
                            "status": details["status"],
                            "days": details["data_points"],
                            "price_high": details["price_high"],
                            "price_low": details["price_low"],
                            "price_range": f"{details['price_range']:.2%}" if not pd.isna(details['price_range']) else "N/A",
                            "box_condition": details["box_condition"],
                            "ma_convergence_ratio": f"{details['ma_convergence_ratio']:.4f}" if not pd.isna(details['ma_convergence_ratio']) else "N/A",
                            "ma_condition": details["ma_condition"],
                            "volatility": f"{details['volatility']:.4f}" if not pd.isna(details['volatility']) else "N/A",
                            "vol_condition": details["vol_condition"]
                        }

                        # 存储窗口详细信息
                        window_details[w] = formatted_details

                        if platform_status:
                            is_platform_in_any_window = True
                    else:
                        # Not enough data for this window check
                        flags[w] = False
                        window_details[w] = {"status": "数据不足", "days": len(df)}

                # If the stock meets criteria in at least one window, add to results
                if is_platform_in_any_window:
                    platform_count += 1
                    # Return only the data for the max_window period for the frontend chart
                    kline_subset = df.iloc[-max_window:].copy()

                    results.append({
                        'code': stock_info['code'],
                        'name': stock_info['name'],
                        'platform_flags': flags,
                        'details': window_details,
                        'kline_data': kline_subset  # Return the DataFrame subset
                    })

                    # 打印符合条件的股票信息
                    print(
                        f"\n{Fore.GREEN}发现平台期股票: {stock_info['name']}({stock_code}){Style.RESET_ALL}")
                    for w, details in window_details.items():
                        if flags[w]:
                            print(f"  {Fore.CYAN}窗口 {w} 天:{Style.RESET_ALL}")
                            if "price_high" in details:
                                print(
                                    f"    - 价格区间: {Fore.YELLOW}{details['price_low']:.2f} - {details['price_high']:.2f}{Style.RESET_ALL}")
                                print(
                                    f"    - 价格波动: {Fore.YELLOW}{details['price_range']}{Style.RESET_ALL}")
                                if "ma_convergence_ratio" in details:
                                    print(
                                        f"    - MA收敛度: {Fore.YELLOW}{details['ma_convergence_ratio']}{Style.RESET_ALL}")
                                if "volatility" in details:
                                    print(
                                        f"    - 波动率: {Fore.YELLOW}{details['volatility']}{Style.RESET_ALL}")

                # 更新进度条
                pbar.set_postfix(成功=success_count, 空数据=empty_count,
                                 错误=error_count, 平台期=platform_count)
                pbar.update(1)

            except Exception as e:
                # Catch exceptions during result processing or platform checking
                error_count += 1
                print(f"{Fore.RED}处理股票 {stock_code} 时出错: {e}{Style.RESET_ALL}")
                import traceback
                traceback.print_exc()  # Print stack trace for debugging
                # 更新进度条
                pbar.set_postfix(成功=success_count, 空数据=empty_count,
                                 错误=error_count, 平台期=platform_count)
                pbar.update(1)

        # 关闭进度条
        pbar.close()

        # 打印扫描结果摘要
        print(
            f"\n{Fore.CYAN}======================================{Style.RESET_ALL}")
        print(f"{Fore.CYAN}扫描完成{Style.RESET_ALL}")
        print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
        print(f"  - 总股票数: {Fore.GREEN}{len(stock_list)}{Style.RESET_ALL}")
        print(f"  - 成功获取: {Fore.GREEN}{success_count}{Style.RESET_ALL}")
        print(f"  - 空数据: {Fore.YELLOW}{empty_count}{Style.RESET_ALL}")
        print(f"  - 处理错误: {Fore.RED}{error_count}{Style.RESET_ALL}")
        print(f"  - 符合平台期条件: {Fore.GREEN}{platform_count}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")

    # 确保所有进程都已经正确登出
    try:
        baostock_logout()
    except Exception as e:
        print(f"{Fore.YELLOW}登出 Baostock 时出错: {e}{Style.RESET_ALL}")  # 记录错误但继续执行

    return results

# =================================
# Example Usage (Commented Out for Module Import)
# =================================
# if __name__ == '__main__':
#     print("Running example usage...")
#     # --- Get Stock List ---
#     print("Logging in to Baostock to get stock list...")
#     lg = bs.login()
#     if lg.error_code != '0':
#         print(f"Baostock login failed: {lg.error_msg}")
#     else:
#         print("Querying all stock basics...")
#         rs = bs.query_stock_basic() # Can add stock_status='1' for listed stocks
#         stocks = []
#         if rs.error_code == '0':
#             while rs.next():
#                 row = rs.get_row_data()
#                 # Filter for Shanghai (sh) and Shenzhen (sz) stocks
#                 if row and len(row) >= 2 and row[0] and row[1] and (row[0].startswith('sh.') or row[0].startswith('sz.')):
#                      stocks.append({'code': row[0], 'name': row[1]})
#             print(f"Fetched {len(stocks)} stock codes.")
#         else:
#             print(f"Failed to query stock basics: {rs.error_msg}")
#         bs.logout()
#         print("Logged out from Baostock.")

#         # --- Define Custom Config (Optional) ---
#         # Example: Use different windows and thresholds
#         user_config = {
#             'windows': [45, 75],
#             'box_threshold': 0.4,
#             'ma_diff_threshold': 0.025,
#             'volatility_threshold': 0.02,
#             'max_workers': 10 # Increase workers for example
#         }
#         print(f"Using custom config: {user_config}")

#         # --- Run Scan ---
#         if stocks:
#             print("Starting scan with custom config...")
#             # Limit stock list for faster example run:
#             # platform_stocks = scan_platform_stocks(stocks[:100], user_config) # Scan first 100 stocks
#             platform_stocks = scan_platform_stocks(stocks, user_config) # Scan all stocks

#             print("\n--- Scan Results ---")
#             if platform_stocks:
#                 print(f"Found {len(platform_stocks)} platform stocks:")
#                 for stock in platform_stocks:
#                     # Print summary, kline_data is a DataFrame here
#                     print(f"  Code: {stock['code']}, Name: {stock['name']}, Flags: {stock['platform_flags']}")
#                     # print(stock['kline_data'].tail()) # Print last few rows of kline data
#             else:
#                 print("No platform stocks found with the given criteria.")
#         else:
#             print("No stocks to scan.")

#     print("Example usage finished.")
