import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_valuation(inputs):
    """
    Calculate intrinsic value based on the selected model.
    Returns a dictionary with relevant metrics.
    """
    model = inputs['model']
    current_price = inputs['current_price']
    mos = inputs['margin_of_safety']
    
    if model == "Stock-to-Flow (S2F)":
        results = s2f_model(inputs)
    elif model == "Metcalfe's Law":
        results = metcalfe_law(inputs)
    elif model == "Network Value to Transactions (NVT)":
        results = nvt_model(inputs)
    elif model == "Pi Cycle Top Indicator":
        results = pi_cycle(inputs)
    elif model == "Reverse S2F":
        results = reverse_s2f(inputs)
    else:
        results = {'intrinsic_value': 0, 'error': 'Unknown model'}
    
    # Common metrics
    intrinsic_value = results.get('intrinsic_value', 0)
    results['current_price'] = current_price
    results['intrinsic_value'] = max(intrinsic_value, 0) * (1 - mos / 100)
    results['safe_buy_price'] = results['intrinsic_value']
    results['undervaluation'] = ((intrinsic_value - current_price) / current_price * 100) if current_price > 0 else 0
    results['verdict'] = get_verdict(results['undervaluation'])
    results['score'] = calculate_score(inputs, results)
    
    # Model-specific metrics
    results['nvt_ratio'] = inputs['market_cap'] / inputs['transaction_volume'] if inputs['transaction_volume'] > 0 else 50
    results['mvrv_z_score'] = (inputs['mvrv'] - 1) / 0.5 if inputs['mvrv'] > 0 else 0
    results['sopr_signal'] = 'Buy' if inputs['sopr'] < 1 else 'Sell' if inputs['sopr'] > 1.2 else 'Hold'
    results['puell_signal'] = 'Buy' if inputs['puell_multiple'] < 0.5 else 'Sell' if inputs['puell_multiple'] > 4 else 'Hold'
    results['mining_cost_vs_price'] = (inputs['mining_cost'] - current_price) / current_price * 100 if current_price > 0 else 0
    
    # All model values
    all_models = {
        's2f_value': s2f_model(inputs).get('intrinsic_value', 0),
        'metcalfe_value': metcalfe_law(inputs).get('intrinsic_value', 0),
        'nvt_value': nvt_model(inputs).get('intrinsic_value', 0),
        'pi_cycle_value': pi_cycle(inputs).get('intrinsic_value', 0),
        'reverse_s2f_value': reverse_s2f(inputs).get('intrinsic_value', 0)
    }
    results.update(all_models)
    
    return results

def get_verdict(undervaluation):
    """Determine buy/sell/hold verdict."""
    if undervaluation > 20:
        return "Strong Buy"
    elif undervaluation > 0:
        return "Buy"
    elif undervaluation > -20:
        return "Hold"
    else:
        return "Sell"

def calculate_score(inputs, results):
    """Calculate an overall score based on metrics."""
    score = 50
    if results['undervaluation'] > 0:
        score += min(results['undervaluation'], 30)
    if inputs['fear_greed'] < 25:
        score += 10
    elif inputs['fear_greed'] > 75:
        score -= 10
    if inputs['sopr'] < 1:
        score += 10
    if inputs['puell_multiple'] < 0.5:
        score += 10
    return max(min(score, 100), 0)

def s2f_model(inputs):
    """Stock-to-Flow model: price based on scarcity."""
    stock = inputs['circulating_supply']
    flow = 6.25 * 144 * 365  # Approx block reward post-halving
    sf = stock / flow if flow > 0 else 100
    intrinsic_value = np.exp(14.6 + 0.05 * sf)  # Simplified regression
    return {'intrinsic_value': intrinsic_value}

def metcalfe_law(inputs):
    """Metcalfe's Law: value proportional to square of active addresses."""
    n = inputs['active_addresses']
    intrinsic_value = (n ** 2) * 0.0001  # Scaling factor
    return {'intrinsic_value': intrinsic_value}

def nvt_model(inputs):
    """NVT: market cap to transaction volume, benchmarked to historical avg."""
    nvt = inputs['market_cap'] / inputs['transaction_volume'] if inputs['transaction_volume'] > 0 else 50
    intrinsic_value = inputs['current_price'] * (50 / nvt)  # Historical avg NVT ~50
    return {'intrinsic_value': intrinsic_value}

def pi_cycle(inputs):
    """Pi Cycle Top: uses 111-day and 350-day MAs."""
    ma_111 = inputs['50_day_ma'] * 1.1  # Proxy adjustment
    ma_350 = inputs['200_day_ma'] * 1.2
    intrinsic_value = ma_111 if inputs['current_price'] > ma_350 else inputs['current_price']
    return {'intrinsic_value': intrinsic_value}

def reverse_s2f(inputs):
    """Reverse S2F: implied growth to reach current price."""
    stock = inputs['circulating_supply']
    flow = 6.25 * 144 * 365
    sf = stock / flow if flow > 0 else 100
    target_price = inputs['current_price']
    implied_sf = (np.log(target_price) - 14.6) / 0.05
    intrinsic_value = target_price  # Use current price as base
    return {'intrinsic_value': intrinsic_value, 'implied_sf': implied_sf}
