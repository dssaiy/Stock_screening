"""
Platform Scanner module for scanning stocks for platform consolidation patterns.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from colorama import Fore, Style

from .data_fetcher import fetch_kline_data, baostock_login
from .industry_filter import apply_industry_diversity_filter
from .config import ScanConfig

# Import analyzers
from .analyzers.price_analyzer import analyze_price
from .analyzers.volume_analyzer import analyze_volume
from .analyzers.combined_analyzer import analyze_stock
from .analyzers.fundamental_analyzer import analyze_fundamentals


def prepare_stock_list(stock_basics_df: pd.DataFrame,
                       industry_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Prepare a list of stocks for scanning, excluding indices and merging industry data.

    Args:
        stock_basics_df: DataFrame containing stock basic information
        industry_df: DataFrame containing industry classification

    Returns:
        List of dictionaries with stock information
    """
    # Filter out indices (type=2) and non-active stocks (status=0)
    stock_list = []

    for _, row in stock_basics_df.iterrows():
        # Skip indices and inactive stocks
        if row['type'] == '2' or row['status'] == '0':
            continue

        stock_info = {
            'code': row['code'],
            'name': row['code_name'],
            'type': row['type'],
            'status': row['status'],
            'industry': 'Unknown'  # Default value
        }

        # Add to list
        stock_list.append(stock_info)

    # Add industry information if available
    if not industry_df.empty:
        industry_dict = dict(zip(industry_df['code'], industry_df['industry']))
        for stock in stock_list:
            if stock['code'] in industry_dict:
                stock['industry'] = industry_dict[stock['code']]

    return stock_list


def scan_stocks(stock_list: List[Dict[str, Any]],
                config: ScanConfig,
                update_progress: Optional[callable] = None) -> List[Dict[str, Any]]:
    """
    Scan stocks for platform consolidation patterns.

    Args:
        stock_list: List of stocks to scan
        config: Scan configuration
        update_progress: Optional callback for updating progress

    Returns:
        List of stocks that meet platform criteria
    """
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    # Use the maximum window size plus some buffer for the start date
    max_window = max(config.windows) if config.windows else 90
    start_date = (datetime.now() - timedelta(days=max_window * 2)
                  ).strftime('%Y-%m-%d')

    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Starting stock platform scan{Style.RESET_ALL}")
    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Scan parameters:{Style.RESET_ALL}")
    print(
        f"  - Date range: {Fore.GREEN}{start_date} to {end_date}{Style.RESET_ALL}")
    print(f"  - Windows: {Fore.GREEN}{config.windows}{Style.RESET_ALL}")
    print(
        f"  - Box threshold: {Fore.GREEN}{config.box_threshold}{Style.RESET_ALL}")
    print(
        f"  - MA diff threshold: {Fore.GREEN}{config.ma_diff_threshold}{Style.RESET_ALL}")
    print(
        f"  - Volatility threshold: {Fore.GREEN}{config.volatility_threshold}{Style.RESET_ALL}")

    # Print volume analysis parameters if enabled
    if config.use_volume_analysis:
        print(f"  - Volume analysis: {Fore.GREEN}Enabled{Style.RESET_ALL}")
        print(
            f"  - Volume change threshold: {Fore.GREEN}{config.volume_change_threshold}{Style.RESET_ALL}")
        print(
            f"  - Volume stability threshold: {Fore.GREEN}{config.volume_stability_threshold}{Style.RESET_ALL}")
        print(
            f"  - Volume increase threshold: {Fore.GREEN}{config.volume_increase_threshold}{Style.RESET_ALL}")
    else:
        print(f"  - Volume analysis: {Fore.YELLOW}Disabled{Style.RESET_ALL}")

    # Print window weights if enabled
    if config.use_window_weights:
        print(f"  - Window weights: {Fore.GREEN}Enabled{Style.RESET_ALL}")
        for window, weight in config.window_weights.items():
            print(
                f"    - {window} days: {Fore.GREEN}{weight}{Style.RESET_ALL}")
    else:
        print(f"  - Window weights: {Fore.YELLOW}Disabled{Style.RESET_ALL}")

    print(
        f"  - Max workers: {Fore.GREEN}{config.max_workers}{Style.RESET_ALL}")
    print(f"  - Stock count: {Fore.GREEN}{len(stock_list)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")

    # Try to use ProcessPoolExecutor, fall back to ThreadPoolExecutor if needed
    try:
        executor_class = ProcessPoolExecutor
        print(
            f"{Fore.GREEN}Using ProcessPoolExecutor for concurrent data fetching{Style.RESET_ALL}")
    except Exception as e:
        print(
            f"{Fore.RED}ProcessPoolExecutor initialization failed: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Falling back to ThreadPoolExecutor{Style.RESET_ALL}")
        executor_class = ThreadPoolExecutor

    # Initialize counters
    success_count = 0
    empty_count = 0
    error_count = 0
    platform_count = 0

    # List to store platform stocks
    platform_stocks = []

    # Use executor for concurrent processing
    with executor_class(max_workers=config.max_workers, initializer=baostock_login) as executor:
        # Submit tasks
        future_to_stock = {
            executor.submit(fetch_kline_data, s['code'], start_date, end_date,
                            config.retry_attempts, config.retry_delay): s
            for s in stock_list
        }

        # Create progress bar
        total_stocks = len(future_to_stock)
        pbar = tqdm(total=total_stocks, desc="Fetching stock data",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        # Process results as they complete
        for i, future in enumerate(future_to_stock):
            stock = future_to_stock[future]
            stock_code = stock['code']
            stock_name = stock['name']

            try:
                # Get K-line data
                df = future.result()

                if df.empty:
                    empty_count += 1
                    pbar.set_postfix(success=success_count, empty=empty_count,
                                     error=error_count, platform=platform_count)
                    pbar.update(1)
                    continue

                # Analyze for platform periods
                analysis_result = analyze_stock(
                    df,
                    config.windows,
                    config.box_threshold,
                    config.ma_diff_threshold,
                    config.volatility_threshold,
                    config.volume_change_threshold,
                    config.volume_stability_threshold,
                    config.volume_increase_threshold,
                    config.use_volume_analysis,
                    config.use_breakthrough_prediction,
                    config.use_window_weights,
                    config.window_weights,
                    config.use_low_position,
                    config.high_point_lookback_days,
                    config.decline_period_days,
                    config.decline_threshold,
                    config.use_rapid_decline_detection,
                    config.rapid_decline_days,
                    config.rapid_decline_threshold,
                    config.use_breakthrough_confirmation,
                    config.breakthrough_confirmation_days,
                    config.use_box_detection,
                    config.box_quality_threshold
                )

                success_count += 1

                # If it's a platform stock, add to results
                if analysis_result["is_platform"]:
                    platform_count += 1

                    # Create result object
                    platform_stock = {
                        'code': stock_code,
                        'name': stock_name,
                        'industry': stock.get('industry', 'Unknown'),
                        'platform_windows': analysis_result["platform_windows"],
                        'details': analysis_result["details"],
                        'selection_reasons': analysis_result["selection_reasons"],
                        'kline_data': df.to_dict(orient='records')
                    }

                    # Add mark lines if available
                    if "mark_lines" in analysis_result:
                        platform_stock['mark_lines'] = analysis_result["mark_lines"]
                        print(
                            f"{Fore.GREEN}添加标记线数据到股票 {stock_code}: {analysis_result['mark_lines']}{Style.RESET_ALL}")

                    # Add volume analysis results if available
                    if config.use_volume_analysis and "volume_analysis" in analysis_result:
                        platform_stock['volume_analysis'] = analysis_result["volume_analysis"]

                    # Add breakthrough prediction results if available
                    if config.use_breakthrough_prediction and "breakthrough_prediction" in analysis_result:
                        platform_stock['breakthrough_prediction'] = analysis_result["breakthrough_prediction"]

                    # Add window weight results if available
                    if config.use_window_weights and "weighted_score" in analysis_result:
                        platform_stock['weighted_score'] = analysis_result["weighted_score"]
                        platform_stock['weight_details'] = analysis_result.get(
                            "weight_details", {})

                    platform_stocks.append(platform_stock)

                # Update progress
                if update_progress and i % 10 == 0:  # Update every 10 stocks
                    progress_pct = (i + 1) / total_stocks * 100
                    update_progress(
                        progress=int(progress_pct),
                        message=f"Processed {i+1}/{total_stocks} stocks. Found {platform_count} platform stocks."
                    )

            except Exception as e:
                error_count += 1
                print(
                    f"{Fore.RED}Error processing stock {stock_code}: {e}{Style.RESET_ALL}")
                import traceback
                traceback.print_exc()

            # Update progress bar
            pbar.set_postfix(success=success_count, empty=empty_count,
                             error=error_count, platform=platform_count)
            pbar.update(1)

        # Close progress bar
        pbar.close()

    # Apply fundamental analysis filter if enabled
    if config.use_fundamental_filter:
        print(f"{Fore.CYAN}Applying fundamental analysis filter...{Style.RESET_ALL}")
        fundamental_filtered_stocks = analyze_fundamentals(
            platform_stocks,
            use_fundamental_filter=config.use_fundamental_filter,
            revenue_growth_percentile=config.revenue_growth_percentile,
            profit_growth_percentile=config.profit_growth_percentile,
            roe_percentile=config.roe_percentile,
            liability_percentile=config.liability_percentile,
            pe_percentile=config.pe_percentile,
            pb_percentile=config.pb_percentile,
            years_to_check=config.fundamental_years_to_check
        )
        fundamental_count = len(fundamental_filtered_stocks)
        print(f"{Fore.GREEN}Fundamental analysis complete. {fundamental_count} stocks passed out of {platform_count}.{Style.RESET_ALL}")
    else:
        fundamental_filtered_stocks = platform_stocks
        fundamental_count = platform_count
        print(f"{Fore.YELLOW}Fundamental analysis filter disabled.{Style.RESET_ALL}")

    # Apply industry diversity filter
    filtered_stocks = apply_industry_diversity_filter(
        fundamental_filtered_stocks,
        expected_count=config.expected_count
    )

    # Print summary
    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Scan completed{Style.RESET_ALL}")
    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
    print(
        f"Total stocks processed: {Fore.GREEN}{success_count + empty_count + error_count}{Style.RESET_ALL}")
    print(f"  - Success: {Fore.GREEN}{success_count}{Style.RESET_ALL}")
    print(f"  - Empty data: {Fore.YELLOW}{empty_count}{Style.RESET_ALL}")
    print(f"  - Errors: {Fore.RED}{error_count}{Style.RESET_ALL}")
    print(
        f"Platform stocks found: {Fore.GREEN}{platform_count}{Style.RESET_ALL}")
    if config.use_fundamental_filter:
        print(
            f"Fundamental filtered stocks: {Fore.GREEN}{fundamental_count}{Style.RESET_ALL}")
    print(
        f"Filtered stocks (industry diversity): {Fore.GREEN}{len(filtered_stocks)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")

    # Final progress update
    if update_progress:
        update_progress(
            progress=100,
            message=f"Scan completed. Found {platform_count} platform stocks, filtered to {len(filtered_stocks)}."
        )

    return filtered_stocks
