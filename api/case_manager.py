"""
Case management module for the platform consolidation scanner.
"""
import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from .json_utils import sanitize_float_for_json, sanitize_kline_data

# Define the case directory
CASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cases')

# Ensure the case directory exists
os.makedirs(CASE_DIR, exist_ok=True)

# Define the index file path
INDEX_FILE = os.path.join(CASE_DIR, 'index.json')

# Initialize the index file if it doesn't exist
if not os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump({'cases': [], 'lastUpdated': datetime.now(
        ).isoformat()}, f, ensure_ascii=False, indent=2)


def get_cases() -> List[Dict[str, Any]]:
    """
    Get all cases from the index file.

    Returns:
        List of case metadata.
    """
    try:
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('cases', [])
    except Exception as e:
        print(f"Error loading cases: {e}")
        return []


def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific case by ID.

    Args:
        case_id: The ID of the case to get.

    Returns:
        Case data if found, None otherwise.
    """
    # Get case metadata from index
    cases = get_cases()
    case_meta = next((case for case in cases if case['id'] == case_id), None)

    if not case_meta:
        return None

    # Case directory path
    case_dir = os.path.join(CASE_DIR, case_id)

    # Check if case directory exists
    if not os.path.exists(case_dir):
        return case_meta

    # Load additional case data
    result = {**case_meta}

    # Load description if exists
    description_file = os.path.join(case_dir, 'description.md')
    if os.path.exists(description_file):
        with open(description_file, 'r', encoding='utf-8') as f:
            result['description'] = f.read()

    # Load analysis if exists
    analysis_file = os.path.join(case_dir, 'analysis.json')
    if os.path.exists(analysis_file):
        with open(analysis_file, 'r', encoding='utf-8') as f:
            result['analysis'] = json.load(f)

    # Load K-line data if exists
    kline_file = os.path.join(case_dir, 'kline_data.json')
    if os.path.exists(kline_file):
        with open(kline_file, 'r', encoding='utf-8') as f:
            result['kline_data'] = json.load(f)

    return result


def create_case(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new case.

    Args:
        case_data: The case data to create.

    Returns:
        The created case data.
    """
    # Validate required fields
    required_fields = ['title', 'stockCode', 'stockName']
    for field in required_fields:
        if field not in case_data:
            raise ValueError(f"Missing required field: {field}")

    # Generate case ID if not provided
    if 'id' not in case_data:
        case_data['id'] = f"case_{int(datetime.now().timestamp())}"

    # Set timestamps
    now = datetime.now().isoformat()
    case_data['createdAt'] = now
    case_data['updatedAt'] = now

    # Ensure tags is a list
    if 'tags' not in case_data:
        case_data['tags'] = []

    # Create case directory
    case_dir = os.path.join(CASE_DIR, case_data['id'])
    os.makedirs(case_dir, exist_ok=True)

    # Save metadata to index
    cases = get_cases()
    cases.append({
        'id': case_data['id'],
        'title': case_data['title'],
        'stockCode': case_data['stockCode'],
        'stockName': case_data['stockName'],
        'createdAt': case_data['createdAt'],
        'updatedAt': case_data['updatedAt'],
        'tags': case_data['tags']
    })

    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'cases': cases,
            'lastUpdated': now
        }, f, ensure_ascii=False, indent=2)

    # Save description if provided
    if 'description' in case_data:
        with open(os.path.join(case_dir, 'description.md'), 'w', encoding='utf-8') as f:
            f.write(case_data['description'])

    # Save analysis if provided
    if 'analysis' in case_data:
        sanitized_analysis = sanitize_float_for_json(case_data['analysis'])
        with open(os.path.join(case_dir, 'analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(sanitized_analysis, f, ensure_ascii=False, indent=2)

    # Save K-line data if provided
    if 'kline_data' in case_data:
        sanitized_kline_data = case_data['kline_data'].copy()
        if 'data' in sanitized_kline_data:
            sanitized_kline_data['data'] = sanitize_kline_data(
                sanitized_kline_data['data'])
        with open(os.path.join(case_dir, 'kline_data.json'), 'w', encoding='utf-8') as f:
            json.dump(sanitized_kline_data, f, ensure_ascii=False, indent=2)

    return case_data


def update_case(case_id: str, case_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Update an existing case.

    Args:
        case_id: The ID of the case to update.
        case_data: The new case data.

    Returns:
        The updated case data if successful, None otherwise.
    """
    # Get cases from index
    cases = get_cases()
    case_index = next((i for i, case in enumerate(cases)
                      if case['id'] == case_id), None)

    if case_index is None:
        return None

    # Update timestamp
    now = datetime.now().isoformat()
    case_data['updatedAt'] = now

    # Update metadata in index
    for field in ['title', 'stockCode', 'stockName', 'tags']:
        if field in case_data:
            cases[case_index][field] = case_data[field]

    cases[case_index]['updatedAt'] = now

    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'cases': cases,
            'lastUpdated': now
        }, f, ensure_ascii=False, indent=2)

    # Case directory path
    case_dir = os.path.join(CASE_DIR, case_id)

    # Create directory if it doesn't exist
    os.makedirs(case_dir, exist_ok=True)

    # Update description if provided
    if 'description' in case_data:
        with open(os.path.join(case_dir, 'description.md'), 'w', encoding='utf-8') as f:
            f.write(case_data['description'])

    # Update analysis if provided
    if 'analysis' in case_data:
        sanitized_analysis = sanitize_float_for_json(case_data['analysis'])
        with open(os.path.join(case_dir, 'analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(sanitized_analysis, f, ensure_ascii=False, indent=2)

    # Update K-line data if provided
    if 'kline_data' in case_data:
        sanitized_kline_data = case_data['kline_data'].copy()
        if 'data' in sanitized_kline_data:
            sanitized_kline_data['data'] = sanitize_kline_data(
                sanitized_kline_data['data'])
        with open(os.path.join(case_dir, 'kline_data.json'), 'w', encoding='utf-8') as f:
            json.dump(sanitized_kline_data, f, ensure_ascii=False, indent=2)

    return get_case(case_id)


def delete_case(case_id: str) -> bool:
    """
    Delete a case.

    Args:
        case_id: The ID of the case to delete.

    Returns:
        True if the case was deleted, False otherwise.
    """
    # Get cases from index
    cases = get_cases()
    case_index = next((i for i, case in enumerate(cases)
                      if case['id'] == case_id), None)

    if case_index is None:
        return False

    # Remove from index
    cases.pop(case_index)

    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'cases': cases,
            'lastUpdated': datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)

    # Remove case directory
    case_dir = os.path.join(CASE_DIR, case_id)
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)

    return True


def create_case_from_analysis(stock_data: Dict[str, Any], analysis_result: Dict[str, Any],
                              kline_data: pd.DataFrame, title: Optional[str] = None,
                              description: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create a case from analysis results.

    Args:
        stock_data: Stock basic information (code, name).
        analysis_result: Analysis results.
        kline_data: K-line data as DataFrame.
        title: Optional case title.
        description: Optional case description.
        tags: Optional tags.

    Returns:
        The created case data.
    """
    # Generate title if not provided
    if not title:
        title = f"{stock_data['name']}({stock_data['code']})平台期分析"

    # Generate default tags if not provided
    if not tags:
        tags = []
        if analysis_result.get('is_platform'):
            tags.append('平台期')
        if analysis_result.get('position_analysis', {}).get('is_low_position'):
            tags.append('低位')
        if analysis_result.get('breakthrough_analysis', {}).get('has_breakthrough_signal'):
            tags.append('突破信号')
        if analysis_result.get('breakthrough_confirmation', {}).get('has_confirmation'):
            tags.append('突破确认')
        if analysis_result.get('is_rapid_decline'):
            tags.append('快速下跌')
        if analysis_result.get('has_decline_pattern'):
            tags.append('安记食品模式')
        if analysis_result.get('box_analysis', {}).get('is_box_pattern'):
            tags.append('箱体形态')

    # Generate default description if not provided
    if not description:
        description = f"# {stock_data['name']}({stock_data['code']})平台期分析\n\n"

        # Add platform period information
        if analysis_result.get('is_platform'):
            platform_windows = analysis_result.get('platform_windows', [])
            description += "## 平台期特征\n\n"
            description += f"该股票在以下窗口期内表现出平台期特征：{', '.join(map(str, platform_windows))}天\n\n"

            # Add selection reasons
            if 'selection_reasons' in analysis_result:
                description += "### 选择原因\n\n"
                for window, reason in analysis_result['selection_reasons'].items():
                    description += f"- {window}天窗口：{reason}\n"
                description += "\n"

        # Add position analysis
        if 'position_analysis' in analysis_result:
            position = analysis_result['position_analysis']
            description += "## 位置分析\n\n"
            if position.get('is_low_position'):
                description += "该股票处于低位，具体表现为：\n\n"
                if 'details' in position:
                    details = position['details']
                    if 'historical_high' in details and 'current_price' in details:
                        description += f"- 历史高点：{details['historical_high']}\n"
                        description += f"- 当前价格：{details['current_price']}\n"
                    if 'decline_percentage' in details:
                        description += f"- 下跌幅度：{details['decline_percentage']:.2f}%\n"
                    if 'days_since_high' in details:
                        description += f"- 高点以来天数：{details['days_since_high']}天\n"
            else:
                description += "该股票不处于低位。\n"
            description += "\n"

        # Add rapid decline analysis
        if analysis_result.get('is_rapid_decline') or analysis_result.get('has_decline_pattern'):
            description += "## 下跌速度分析\n\n"
            if analysis_result.get('is_rapid_decline'):
                description += "该股票经历了快速下跌，具体表现为：\n\n"
                if 'decline_details' in analysis_result:
                    details = analysis_result['decline_details']
                    if 'high_price' in details and 'low_price' in details:
                        description += f"- 高点价格：{details.get('high_price')}\n"
                        description += f"- 低点价格：{details.get('low_price')}\n"
                    if 'decline_percentage' in details:
                        description += f"- 总下跌幅度：{details.get('decline_percentage', 0) * 100:.2f}%\n"
                    if 'max_rapid_decline' in details:
                        description += f"- 快速下跌幅度：{details.get('max_rapid_decline', 0) * 100:.2f}%\n"
                    if 'rapid_decline_start_date' in details and 'rapid_decline_end_date' in details:
                        description += f"- 快速下跌时间段：{details.get('rapid_decline_start_date')} 至 {details.get('rapid_decline_end_date')}\n"
                    if 'decline_concentration' in details:
                        description += f"- 下跌集中度：{details.get('decline_concentration', 0):.2f}\n"
                    if 'decline_volatility' in details:
                        description += f"- 下跌波动性：{details.get('decline_volatility', 0):.4f}\n"

            if analysis_result.get('has_decline_pattern'):
                description += "\n该股票符合安记食品模式：低位+快速下跌后形成平台期。这种模式通常预示着较大的上涨空间。\n"
            description += "\n"

        # Add box pattern analysis
        if 'box_analysis' in analysis_result:
            box_analysis = analysis_result['box_analysis']
            description += "## 箱体形态分析\n\n"
            if box_analysis.get('is_box_pattern'):
                description += "该股票形成了箱体形态，具体表现为：\n\n"
                if 'support_levels' in box_analysis:
                    support_levels = box_analysis.get('support_levels', [])
                    if support_levels:
                        description += f"- 支撑位：{', '.join([f'{level:.2f}' for level in support_levels])}\n"
                if 'resistance_levels' in box_analysis:
                    resistance_levels = box_analysis.get(
                        'resistance_levels', [])
                    if resistance_levels:
                        description += f"- 阻力位：{', '.join([f'{level:.2f}' for level in resistance_levels])}\n"
                if 'support_strength' in box_analysis:
                    description += f"- 支撑强度：{box_analysis.get('support_strength')}\n"
                if 'resistance_strength' in box_analysis:
                    description += f"- 阻力强度：{box_analysis.get('resistance_strength')}\n"
                if 'volatility' in box_analysis:
                    description += f"- 箱体内波动率：{box_analysis.get('volatility', 0):.4f}\n"
                if 'box_quality' in box_analysis:
                    description += f"- 箱体质量：{box_analysis.get('box_quality', 0):.2f}\n"
            else:
                description += "该股票未形成明显的箱体形态。\n"
            description += "\n"

        # Add breakthrough analysis
        if 'breakthrough_analysis' in analysis_result:
            breakthrough = analysis_result['breakthrough_analysis']
            description += "## 突破分析\n\n"
            if breakthrough.get('has_breakthrough_signal'):
                description += f"该股票显示出突破信号，共有{breakthrough.get('signal_count', 0)}个指标显示突破。\n\n"
                if 'details' in breakthrough:
                    description += "### 突破指标详情\n\n"
                    for indicator, details in breakthrough['details'].items():
                        description += f"- {indicator}：{details.get('status', '')}\n"
            else:
                description += "该股票暂未显示出突破信号。\n"
            description += "\n"

        # Add parameters
        description += "## 分析参数\n\n"
        description += "本次分析使用的参数如下：\n\n"
        description += "```json\n"
        description += json.dumps(analysis_result.get('parameters',
                                  {}), indent=2, ensure_ascii=False)
        description += "\n```\n"

    # Convert DataFrame to list for JSON serialization
    kline_data_list = []
    if not kline_data.empty:
        kline_data_list = kline_data.to_dict('records')
        # Sanitize kline data to handle NaN values
        kline_data_list = sanitize_kline_data(kline_data_list)

    # Create case data
    case_data = {
        'title': title,
        'stockCode': stock_data['code'],
        'stockName': stock_data['name'],
        'tags': tags,
        'description': description,
        'analysis': sanitize_float_for_json(analysis_result),
        'kline_data': {
            'code': stock_data['code'],
            'name': stock_data['name'],
            'data': kline_data_list
        }
    }

    # Create the case
    return create_case(case_data)


def create_anjishi_case():
    """
    Create a case for Anjishi (安记食品) based on our analysis.
    """
    # Stock data
    stock_data = {
        'code': 'sh.603696',
        'name': '安记食品'
    }

    # Analysis result
    analysis_result = {
        'parameters': {
            'windows': [30, 60, 90],
            'box_threshold': 0.5,
            'ma_diff_threshold': 0.03,
            'volatility_threshold': 0.09,
            'use_volume_analysis': True,
            'volume_change_threshold': 0.8,
            'volume_stability_threshold': 0.5,
            'volume_increase_threshold': 1.5,
            'use_breakthrough_prediction': False,
            'use_window_weights': False,
            'use_low_position': True,
            'high_point_lookback_days': 365,
            'decline_period_days': 180,
            'decline_threshold': 0.3,
            'use_rapid_decline_detection': True,
            'rapid_decline_days': 30,
            'rapid_decline_threshold': 0.15,
            'use_box_detection': True,
            'box_quality_threshold': 0.6
        },
        'is_platform': True,
        'platform_windows': [60],
        'selection_reasons': {
            '60': '60天窗口期内价格波动小于50%，均线高度粘合，波动率低，成交量稳定'
        },
        'position_analysis': {
            'is_low_position': True,
            'details': {
                'historical_high': 16.25,
                'high_date': '2024-11-25',
                'current_price': 6.38,
                'decline_percentage': 60.74,
                'days_since_high': 97
            }
        },
        'is_rapid_decline': True,
        'has_decline_pattern': True,
        'decline_details': {
            'high_price': 16.25,
            'high_date': '2024-11-25',
            'low_price': 6.38,
            'low_date': '2025-04-08',
            'decline_percentage': 0.6074,
            'decline_days': 97,
            'daily_decline_rate': 0.0063,
            'max_rapid_decline': 0.5725,
            'rapid_decline_start_date': '2024-11-25',
            'rapid_decline_end_date': '2025-01-03',
            'decline_concentration': 0.94,
            'decline_volatility': 0.0739
        },
        'box_analysis': {
            'is_box_pattern': True,
            'support_levels': [6.38, 6.63, 7.72],
            'resistance_levels': [7.58, 8.00, 8.52],
            'support_strength': 3,
            'resistance_strength': 5,
            'volatility': 0.0815,
            'box_quality': 0.72,
            'status': '符合箱体条件'
        }
    }

    # Create empty DataFrame for K-line data
    # In a real scenario, this would be actual K-line data
    kline_data = pd.DataFrame()

    # Description
    description = """# 安记食品(603696)平台期分析案例

## 背景介绍

安记食品(603696)是一家专注于调味品研发、生产和销售的企业。本案例分析了该股票在2024年9月至2025年3月期间形成的平台期特征，以及随后的突破走势。

## 平台期特征

安记食品在2024年9月至2025年3月期间表现出明显的平台期特征，主要体现在：

1. **价格区间有限**：60天窗口期内，股价波动幅度小于50%
2. **均线高度粘合**：MA5、MA10、MA20等均线相互靠近，几乎平行运行
3. **波动率较低**：日收益率标准差维持在较低水平
4. **成交量萎缩**：平台期内成交量明显低于前期，且保持稳定

## 位置分析

该股票处于明显低位，从2024年11月25日的历史高点16.25元下跌至6.38元，跌幅达60.74%。这种大幅下跌远超过30%的低位标准，表明股价已经处于相对底部区域。

## 下跌速度分析

安记食品在2024年11月底至2025年1月初经历了快速下跌，具体表现为：

- 高点价格：16.25元（2024-11-25）
- 低点价格：6.38元（2025-04-08）
- 总下跌幅度：60.74%
- 快速下跌幅度：57.25%
- 快速下跌时间段：2024-11-25 至 2025-01-03
- 下跌集中度：0.94（表明94%的下跌发生在短短40天内）
- 下跌波动性：0.0739

这种快速下跌后形成平台期的模式，我们称之为"安记食品模式"，通常预示着较大的上涨空间。

## 箱体形态分析

安记食品在快速下跌后形成了明显的箱体形态，具体表现为：

- 支撑位：6.38, 6.63, 7.72
- 阻力位：7.58, 8.00, 8.52
- 支撑强度：3
- 阻力强度：5
- 箱体内波动率：0.0815
- 箱体质量：0.72

这种高质量的箱体形态表明股价在低位得到了有效支撑，并形成了稳定的整理区间。

## 突破分析

在2025年4月初，安记食品出现了放量突破箱体上沿的迹象，随后股价持续上涨。这种底部箱体突破通常预示着较大的上涨空间。

## 操作建议

1. **买入时机**：箱体突破并得到成交量确认后
2. **止损设置**：箱体下沿或突破点下方5-8%
3. **目标价位**：前期高点或1.5-2倍箱体高度

## 经验总结

安记食品的案例展示了典型的"快速下跌后形成底部箱体整理再突破上涨"的形态，这种形态具有较高的成功率和较好的风险收益比。识别此类形态的关键在于：

1. 确认股价经历了快速下跌（通常在1个月内下跌超过15%）
2. 下跌后形成低位箱体整理（支撑位和阻力位明确）
3. 箱体整理时间足够长（通常2-3个月以上）
4. 成交量在箱体期间保持低位稳定
5. 突破时伴随明显的成交量放大

本案例验证了我们的平台期识别系统和箱体检测功能在实际应用中的有效性，特别是对于"安记食品模式"的识别准确度较高。"""

    # Create the case
    return create_case_from_analysis(
        stock_data=stock_data,
        analysis_result=analysis_result,
        kline_data=kline_data,
        title="安记食品底部横盘案例分析",
        description=description,
        tags=["底部横盘", "低位", "平台期", "突破"]
    )
