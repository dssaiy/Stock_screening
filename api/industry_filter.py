"""
Industry Filter module for ensuring industry diversity in stock selection.
"""
from typing import List, Dict, Any
import pandas as pd
from collections import Counter

def apply_industry_diversity_filter(platform_stocks: List[Dict[str, Any]], 
                                   expected_count: int = 10) -> List[Dict[str, Any]]:
    """
    Filter platform stocks to ensure industry diversity.
    
    Args:
        platform_stocks: List of stocks that meet platform criteria
        expected_count: Expected number of stocks to return
    
    Returns:
        Filtered list of stocks with industry diversity
    """
    if not platform_stocks:
        return []
    
    if len(platform_stocks) <= expected_count:
        return platform_stocks
    
    # Extract industries
    industries = [stock.get('industry', 'Unknown') for stock in platform_stocks]
    industry_counts = Counter(industries)
    
    # Calculate target count per industry
    total_industries = len(industry_counts)
    base_per_industry = expected_count // total_industries
    remainder = expected_count % total_industries
    
    # Allocate slots per industry
    industry_slots = {}
    for i, (industry, _) in enumerate(industry_counts.most_common()):
        industry_slots[industry] = base_per_industry + (1 if i < remainder else 0)
    
    # Select stocks based on allocated slots
    selected_stocks = []
    current_counts = Counter()
    
    # First pass: select stocks until we reach the target for each industry
    for stock in platform_stocks:
        industry = stock.get('industry', 'Unknown')
        if current_counts[industry] < industry_slots[industry]:
            selected_stocks.append(stock)
            current_counts[industry] += 1
    
    # If we still need more stocks, add the remaining ones
    if len(selected_stocks) < expected_count:
        remaining_slots = expected_count - len(selected_stocks)
        remaining_stocks = [s for s in platform_stocks if s not in selected_stocks]
        selected_stocks.extend(remaining_stocks[:remaining_slots])
    
    return selected_stocks

def get_industry_distribution(stocks: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Get the distribution of industries in a list of stocks.
    
    Args:
        stocks: List of stocks
    
    Returns:
        Dictionary mapping industry names to counts
    """
    industries = [stock.get('industry', 'Unknown') for stock in stocks]
    return dict(Counter(industries))
