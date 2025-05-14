"""
Fundamental Analyzer module for stock fundamental analysis.
"""
import pandas as pd
import numpy as np
import baostock as bs
from typing import Dict, Any, List, Optional, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_fundamentals(stock_list: List[Dict[str, Any]], 
                         use_fundamental_filter: bool = False,
                         revenue_growth_percentile: float = 0.3,
                         profit_growth_percentile: float = 0.3,
                         roe_percentile: float = 0.3,
                         liability_percentile: float = 0.3,
                         pe_percentile: float = 0.7,
                         pb_percentile: float = 0.7,
                         years_to_check: int = 3) -> List[Dict[str, Any]]:
    """
    Analyze stocks based on fundamental metrics and filter them according to industry percentiles.
    
    Args:
        stock_list: List of stock dictionaries with at least 'code' and 'industry' fields
        use_fundamental_filter: Whether to apply fundamental filtering
        revenue_growth_percentile: Required percentile for revenue growth (lower = stricter)
        profit_growth_percentile: Required percentile for profit growth (lower = stricter)
        roe_percentile: Required percentile for ROE (lower = stricter)
        liability_percentile: Required percentile for liability ratio (higher = stricter)
        pe_percentile: Required percentile for PE ratio (higher = stricter)
        pb_percentile: Required percentile for PB ratio (higher = stricter)
        years_to_check: Number of consecutive years to check for growth metrics
    
    Returns:
        List of stocks that pass the fundamental filters
    """
    if not use_fundamental_filter or not stock_list:
        return stock_list
    
    logger.info(f"Starting fundamental analysis for {len(stock_list)} stocks")
    
    # 按行业分组
    industry_groups = {}
    for stock in stock_list:
        industry = stock.get('industry', 'Unknown')
        if industry not in industry_groups:
            industry_groups[industry] = []
        industry_groups[industry].append(stock)
    
    # 获取并计算每个行业内的基本面指标
    filtered_stocks = []
    
    for industry, stocks in industry_groups.items():
        if len(stocks) <= 1:  # 如果行业内只有一只股票，直接保留
            filtered_stocks.extend(stocks)
            continue
            
        logger.info(f"Analyzing {len(stocks)} stocks in industry: {industry}")
        
        # 获取该行业所有股票的基本面数据
        industry_fundamentals = []
        
        for stock in stocks:
            code = stock['code']
            try:
                # 获取基本面数据
                stock_fundamentals = get_stock_fundamentals(code, years_to_check)
                if stock_fundamentals:
                    stock_fundamentals['code'] = code
                    industry_fundamentals.append(stock_fundamentals)
            except Exception as e:
                logger.error(f"Error getting fundamentals for {code}: {str(e)}")
        
        if not industry_fundamentals:
            continue
            
        # 转换为DataFrame便于计算行业百分位
        industry_df = pd.DataFrame(industry_fundamentals)
        
        # 计算行业内各指标的百分位
        for stock in industry_fundamentals:
            code = stock['code']
            
            # 检查是否满足所有基本面条件
            meets_criteria = True
            reasons = []
            
            # 1. 营收增长率 - 要求位于行业前revenue_growth_percentile
            if 'revenue_growth_consistent' in stock and not stock['revenue_growth_consistent']:
                meets_criteria = False
                reasons.append("营收增长不连续")
            elif 'avg_revenue_growth' in stock:
                percentile = calculate_percentile(industry_df, 'avg_revenue_growth', stock['avg_revenue_growth'], ascending=False)
                if percentile > revenue_growth_percentile:
                    meets_criteria = False
                    reasons.append(f"营收增长率行业百分位({percentile:.2f})不达标")
            else:
                meets_criteria = False
                reasons.append("缺少营收增长数据")
                
            # 2. 净利润增长率 - 要求位于行业前profit_growth_percentile
            if 'profit_growth_consistent' in stock and not stock['profit_growth_consistent']:
                meets_criteria = False
                reasons.append("净利润增长不连续")
            elif 'avg_profit_growth' in stock:
                percentile = calculate_percentile(industry_df, 'avg_profit_growth', stock['avg_profit_growth'], ascending=False)
                if percentile > profit_growth_percentile:
                    meets_criteria = False
                    reasons.append(f"净利润增长率行业百分位({percentile:.2f})不达标")
            else:
                meets_criteria = False
                reasons.append("缺少净利润增长数据")
                
            # 3. ROE - 要求位于行业前roe_percentile
            if 'roe_consistent' in stock and not stock['roe_consistent']:
                meets_criteria = False
                reasons.append("ROE不连续")
            elif 'avg_roe' in stock:
                percentile = calculate_percentile(industry_df, 'avg_roe', stock['avg_roe'], ascending=False)
                if percentile > roe_percentile:
                    meets_criteria = False
                    reasons.append(f"ROE行业百分位({percentile:.2f})不达标")
            else:
                meets_criteria = False
                reasons.append("缺少ROE数据")
                
            # 4. 资产负债率 - 要求位于行业后liability_percentile (较低)
            if 'liability_ratio_consistent' in stock and not stock['liability_ratio_consistent']:
                meets_criteria = False
                reasons.append("资产负债率不稳定")
            elif 'avg_liability_ratio' in stock:
                percentile = calculate_percentile(industry_df, 'avg_liability_ratio', stock['avg_liability_ratio'], ascending=True)
                if percentile > liability_percentile:
                    meets_criteria = False
                    reasons.append(f"资产负债率行业百分位({percentile:.2f})不达标")
            else:
                meets_criteria = False
                reasons.append("缺少资产负债率数据")
                
            # 5. PE - 要求不在行业前pe_percentile (较高)
            if 'pe_ttm' in stock:
                percentile = calculate_percentile(industry_df, 'pe_ttm', stock['pe_ttm'], ascending=True)
                if percentile < pe_percentile:
                    meets_criteria = False
                    reasons.append(f"PE行业百分位({percentile:.2f})过高")
            else:
                meets_criteria = False
                reasons.append("缺少PE数据")
                
            # 6. PB - 要求不在行业前pb_percentile (较高)
            if 'pb_mrq' in stock:
                percentile = calculate_percentile(industry_df, 'pb_mrq', stock['pb_mrq'], ascending=True)
                if percentile < pb_percentile:
                    meets_criteria = False
                    reasons.append(f"PB行业百分位({percentile:.2f})过高")
            else:
                meets_criteria = False
                reasons.append("缺少PB数据")
            
            # 如果满足所有条件，添加到过滤后的列表
            if meets_criteria:
                # 找到原始stock对象
                for original_stock in stocks:
                    if original_stock['code'] == code:
                        # 添加基本面分析结果
                        original_stock['fundamental_analysis'] = {
                            'avg_revenue_growth': stock.get('avg_revenue_growth'),
                            'avg_profit_growth': stock.get('avg_profit_growth'),
                            'avg_roe': stock.get('avg_roe'),
                            'avg_liability_ratio': stock.get('avg_liability_ratio'),
                            'pe_ttm': stock.get('pe_ttm'),
                            'pb_mrq': stock.get('pb_mrq')
                        }
                        filtered_stocks.append(original_stock)
                        break
            else:
                # 记录未通过的原因
                logger.info(f"Stock {code} failed fundamental filter: {', '.join(reasons)}")
    
    logger.info(f"Fundamental analysis complete. {len(filtered_stocks)} stocks passed out of {len(stock_list)}")
    return filtered_stocks

def get_stock_fundamentals(code: str, years_to_check: int = 3) -> Dict[str, Any]:
    """
    Get fundamental data for a stock over multiple years.
    
    Args:
        code: Stock code
        years_to_check: Number of years to check
    
    Returns:
        Dictionary with fundamental metrics
    """
    # 获取当前年份和季度
    import datetime
    current_year = datetime.datetime.now().year
    
    # 获取最近几年的财务数据
    growth_data = []
    profit_data = []
    balance_data = []
    
    # 获取最近years_to_check年的年报数据
    for year in range(current_year-years_to_check, current_year):
        # 获取成长能力数据
        rs_growth = bs.query_growth_data(code=code, year=year, quarter=4)
        if rs_growth.error_code != '0':
            logger.warning(f"Failed to get growth data for {code}, year {year}: {rs_growth.error_msg}")
            continue
            
        growth_list = []
        while (rs_growth.error_code == '0') and rs_growth.next():
            growth_list.append(rs_growth.get_row_data())
        
        if growth_list:
            growth_data.append(pd.DataFrame(growth_list, columns=rs_growth.fields))
        
        # 获取盈利能力数据
        rs_profit = bs.query_profit_data(code=code, year=year, quarter=4)
        if rs_profit.error_code != '0':
            logger.warning(f"Failed to get profit data for {code}, year {year}: {rs_profit.error_msg}")
            continue
            
        profit_list = []
        while (rs_profit.error_code == '0') and rs_profit.next():
            profit_list.append(rs_profit.get_row_data())
        
        if profit_list:
            profit_data.append(pd.DataFrame(profit_list, columns=rs_profit.fields))
        
        # 获取偿债能力数据
        rs_balance = bs.query_balance_data(code=code, year=year, quarter=4)
        if rs_balance.error_code != '0':
            logger.warning(f"Failed to get balance data for {code}, year {year}: {rs_balance.error_msg}")
            continue
            
        balance_list = []
        while (rs_balance.error_code == '0') and rs_balance.next():
            balance_list.append(rs_balance.get_row_data())
        
        if balance_list:
            balance_data.append(pd.DataFrame(balance_list, columns=rs_balance.fields))
    
    # 如果没有足够的数据，返回空字典
    if len(growth_data) < years_to_check or len(profit_data) < years_to_check or len(balance_data) < years_to_check:
        logger.warning(f"Insufficient data for {code}, need {years_to_check} years but got growth:{len(growth_data)}, profit:{len(profit_data)}, balance:{len(balance_data)}")
        return {}
    
    # 合并数据
    growth_df = pd.concat(growth_data)
    profit_df = pd.concat(profit_data)
    balance_df = pd.concat(balance_data)
    
    # 获取当前的市盈率和市净率
    # 获取最近的K线数据
    rs_k = bs.query_history_k_data_plus(
        code,
        "date,code,close,peTTM,pbMRQ",
        start_date=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
        end_date=datetime.datetime.now().strftime('%Y-%m-%d'),
        frequency="d"
    )
    
    pe_ttm = None
    pb_mrq = None
    
    if rs_k.error_code == '0':
        k_list = []
        while rs_k.next():
            k_list.append(rs_k.get_row_data())
        
        if k_list:
            k_df = pd.DataFrame(k_list, columns=rs_k.fields)
            # 转换为数值类型
            k_df['peTTM'] = pd.to_numeric(k_df['peTTM'], errors='coerce')
            k_df['pbMRQ'] = pd.to_numeric(k_df['pbMRQ'], errors='coerce')
            
            # 获取最新的非NaN值
            pe_ttm = k_df.loc[k_df['peTTM'].notna(), 'peTTM'].iloc[-1] if not k_df.loc[k_df['peTTM'].notna()].empty else None
            pb_mrq = k_df.loc[k_df['pbMRQ'].notna(), 'pbMRQ'].iloc[-1] if not k_df.loc[k_df['pbMRQ'].notna()].empty else None
    
    # 提取并计算关键指标
    result = {}
    
    # 1. 营收增长率 - 检查连续性和计算平均值
    try:
        revenue_growth_values = []
        for i in range(years_to_check):
            yoy_value = growth_df.iloc[i]['YOYAsset']
            if yoy_value and yoy_value != '' and float(yoy_value) > 0:
                revenue_growth_values.append(float(yoy_value))
            else:
                break
        
        result['revenue_growth_consistent'] = len(revenue_growth_values) == years_to_check
        if result['revenue_growth_consistent']:
            result['avg_revenue_growth'] = sum(revenue_growth_values) / len(revenue_growth_values)
    except Exception as e:
        logger.error(f"Error calculating revenue growth for {code}: {str(e)}")
    
    # 2. 净利润增长率 - 检查连续性和计算平均值
    try:
        profit_growth_values = []
        for i in range(years_to_check):
            yoy_value = growth_df.iloc[i]['YOYPNI']
            if yoy_value and yoy_value != '' and float(yoy_value) > 0:
                profit_growth_values.append(float(yoy_value))
            else:
                break
        
        result['profit_growth_consistent'] = len(profit_growth_values) == years_to_check
        if result['profit_growth_consistent']:
            result['avg_profit_growth'] = sum(profit_growth_values) / len(profit_growth_values)
    except Exception as e:
        logger.error(f"Error calculating profit growth for {code}: {str(e)}")
    
    # 3. ROE - 检查连续性和计算平均值
    try:
        roe_values = []
        for i in range(years_to_check):
            roe_value = profit_df.iloc[i]['roeAvg']
            if roe_value and roe_value != '' and float(roe_value) > 0:
                roe_values.append(float(roe_value))
            else:
                break
        
        result['roe_consistent'] = len(roe_values) == years_to_check
        if result['roe_consistent']:
            result['avg_roe'] = sum(roe_values) / len(roe_values)
    except Exception as e:
        logger.error(f"Error calculating ROE for {code}: {str(e)}")
    
    # 4. 资产负债率 - 检查连续性和计算平均值
    try:
        liability_ratio_values = []
        for i in range(years_to_check):
            ratio_value = balance_df.iloc[i]['liabilityToAsset']
            if ratio_value and ratio_value != '':
                liability_ratio_values.append(float(ratio_value))
            else:
                break
        
        result['liability_ratio_consistent'] = len(liability_ratio_values) == years_to_check
        if result['liability_ratio_consistent']:
            result['avg_liability_ratio'] = sum(liability_ratio_values) / len(liability_ratio_values)
    except Exception as e:
        logger.error(f"Error calculating liability ratio for {code}: {str(e)}")
    
    # 5. 市盈率
    if pe_ttm is not None:
        result['pe_ttm'] = float(pe_ttm)
    
    # 6. 市净率
    if pb_mrq is not None:
        result['pb_mrq'] = float(pb_mrq)
    
    return result

def calculate_percentile(df: pd.DataFrame, column: str, value: float, ascending: bool = True) -> float:
    """
    Calculate the percentile of a value within a DataFrame column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to calculate percentile for
        value: Value to find percentile for
        ascending: Whether to sort in ascending order
    
    Returns:
        Percentile as a float between 0 and 1
    """
    if column not in df.columns or df[column].isna().all():
        return 1.0  # 如果列不存在或全为NaN，返回最大百分位
    
    # 过滤有效值
    valid_values = df[column].dropna()
    if len(valid_values) == 0:
        return 1.0
    
    # 排序并计算百分位
    sorted_values = sorted(valid_values, reverse=not ascending)
    rank = sorted_values.index(value) if value in sorted_values else len(sorted_values)
    
    return rank / len(sorted_values)
