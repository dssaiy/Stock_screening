"""
Data Fetcher module for retrieving stock data from Baostock.
Implements robust connection handling and retry logic.
"""
import baostock as bs
import pandas as pd
import time
import threading
from typing import List, Dict, Any, Optional
from colorama import Fore, Style
import traceback

# Thread-local storage for Baostock connections
_thread_local = threading.local()

def baostock_login() -> None:
    """
    Login to Baostock API with thread-local connection.
    Each thread/process will have its own connection.
    """
    # Check if already logged in
    if hasattr(_thread_local, 'logged_in') and _thread_local.logged_in:
        return
    
    # Login
    lg = bs.login()
    if lg.error_code != '0':
        print(f"{Fore.RED}Baostock login failed: {lg.error_msg}{Style.RESET_ALL}")
        raise ConnectionError(f"Baostock login failed: {lg.error_msg}")
    
    _thread_local.logged_in = True
    print(f"{Fore.GREEN}Baostock login successful in thread {threading.current_thread().name}{Style.RESET_ALL}")

def baostock_logout() -> None:
    """
    Logout from Baostock API and clean up thread-local connection.
    """
    if hasattr(_thread_local, 'logged_in') and _thread_local.logged_in:
        bs.logout()
        _thread_local.logged_in = False
        print(f"{Fore.GREEN}Baostock logout successful in thread {threading.current_thread().name}{Style.RESET_ALL}")

def baostock_relogin() -> None:
    """
    Re-login to Baostock API (logout first, then login again).
    """
    baostock_logout()
    baostock_login()

class BaostockConnectionManager:
    """
    Context manager for Baostock connections.
    Ensures proper login/logout handling.
    """
    def __enter__(self):
        baostock_login()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        baostock_logout()
        return False  # Don't suppress exceptions

def fetch_stock_basics() -> pd.DataFrame:
    """
    Fetch basic information for all stocks.
    
    Returns:
        pd.DataFrame: DataFrame containing stock basic information
    
    Raises:
        ConnectionError: If Baostock connection fails
        ValueError: If no data is returned
    """
    with BaostockConnectionManager():
        print(f"{Fore.CYAN}Fetching stock basic information...{Style.RESET_ALL}")
        rs = bs.query_stock_basic()
        
        if rs.error_code != '0':
            raise ConnectionError(f"Failed to query stock basics: {rs.error_msg}")
        
        stock_basics_list = []
        while rs.next():
            stock_basics_list.append(rs.get_row_data())
        
        if not stock_basics_list:
            raise ValueError("No stock basic information retrieved")
        
        return pd.DataFrame(stock_basics_list, columns=rs.fields)

def fetch_industry_data() -> pd.DataFrame:
    """
    Fetch industry classification data for all stocks.
    
    Returns:
        pd.DataFrame: DataFrame containing industry classification
    
    Raises:
        ConnectionError: If Baostock connection fails
        ValueError: If no data is returned
    """
    with BaostockConnectionManager():
        print(f"{Fore.CYAN}Fetching industry classification data...{Style.RESET_ALL}")
        rs = bs.query_stock_industry()
        
        if rs.error_code != '0':
            raise ConnectionError(f"Failed to query industry data: {rs.error_msg}")
        
        industry_list = []
        while rs.next():
            industry_list.append(rs.get_row_data())
        
        if not industry_list:
            raise ValueError("No industry classification data retrieved")
        
        return pd.DataFrame(industry_list, columns=rs.fields)

def fetch_kline_data(code: str, start_date: str, end_date: str,
                     retry_attempts: int = 3,
                     retry_delay: int = 1) -> pd.DataFrame:
    """
    Fetch K-line data for a specific stock with retry logic.
    
    Args:
        code: Stock code (e.g., 'sh.600000')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        retry_attempts: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        pd.DataFrame: DataFrame containing K-line data
    """
    retries = 0
    
    while True:
        try:
            # Ensure we're logged in
            baostock_login()
            
            # Query historical K-line data
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume,turn,preclose,pctChg,peTTM,pbMRQ",
                start_date=start_date,
                end_date=end_date,
                frequency="d",     # Daily frequency
                adjustflag="2"     # Forward adjusted prices
            )
            
            # Check for API errors
            if rs.error_code != '0':
                retries += 1
                print(f"{Fore.YELLOW}Attempt {retries}/{retry_attempts}: Baostock query failed for {code}. Error: {rs.error_msg}{Style.RESET_ALL}")
                
                if retries >= retry_attempts:
                    print(f"{Fore.RED}Failed to fetch data for {code} after {retry_attempts} attempts{Style.RESET_ALL}")
                    return pd.DataFrame()
                
                # Retry with re-login
                time.sleep(retry_delay * (1 + retries * 0.5))
                baostock_relogin()
                continue
            
            # Process the data if query was successful
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            # Convert to DataFrame
            if not data_list:
                print(f"{Fore.YELLOW}No data returned for {code} from {start_date} to {end_date}{Style.RESET_ALL}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turn', 'preclose', 'pctChg', 'peTTM', 'pbMRQ']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            retries += 1
            print(f"{Fore.RED}Attempt {retries}/{retry_attempts}: Exception while fetching data for {code}: {e}{Style.RESET_ALL}")
            
            if retries >= retry_attempts:
                print(f"{Fore.RED}Failed to fetch data for {code} after {retry_attempts} attempts{Style.RESET_ALL}")
                return pd.DataFrame()
            
            # Retry with re-login
            time.sleep(retry_delay * (1 + retries * 0.5))
            baostock_relogin()
