import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from data_fetch import fetch_history

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def calibrate_s2f(history, s2f_intercept=14.6, s2f_slope=0.05):
    """
    Calibrate S2F model using historical data or user-provided coefficients.
    """
    try:
        if history.empty:
            return [s2f_intercept, s2f_slope]
        stock = history['circulating_supply'] if 'circulating_supply' in history else 19700000
        flow = 6.25 * 144 * 365
        sf = stock / flow if flow > 0 else 100
        log_price = np.log(history['Close'])
        coeffs = np.polyfit(sf, log_price, 1)
        return coeffs
    except Exception as e:
        logging.error(f"Error calibrating S2F: {str(e)}")
        return [s2f_intercept, s2f_slope]

def calculate_valuation(inputs):
    """
    Calculate intrinsic value based on the selected model.
    Returns a dictionary with relevant metrics.
    """
    model = inputs['model']
    current_price = inputs['current_price']
    mos = inputs['margin_of_safety']
    volatility_adj = inputs['volatility_adj']
    history = fetch_history(period='5y')
    
    try:
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
        elif model == "Market Sentiment Composite (MSC)":
            results = msc_model(inputs)
        elif model == "Bitcoin Energy Value Model":
            results = energy_value_model(inputs)
        elif model == "RVMR":
            results = rvmr_model(inputs)
        elif model == "Mayer Multiple":
            results = mayer_multiple(inputs)
        elif model == "Hash Ribbons":
            results = hash_ribbons(inputs, history)
        else:
            results = {'intrinsic_value': 0, 'error': 'Unknown model'}
        
        # Common metrics
        intrinsic_value = results.get('intrinsic_value', 0) * (1 - volatility_adj / 100)
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
            'reverse_s2f_value': reverse_s2f(inputs).get('intrinsic_value', 0),
            'msc_value': msc_model(inputs).get('intrinsic_value', 0),
            'energy_value': energy_value_model(inputs).get('intrinsic_value', 0),
            'rvmr_value': rvmr_model(inputs).get('intrinsic_value', 0),
            'mayer_multiple_value': mayer_multiple(inputs).get('intrinsic_value', 0),
            'hash_ribbons_value': hash_ribbons(inputs, history).get('intrinsic_value', 0)
        }
        results.update(all_models)
        
        return results
    
    except Exception as e:
        logging.error(f"Error in valuation: {str(e)}")
        return {'intrinsic_value': 0, 'error': f'Calculation failed: {str(e)}'}

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
    history = fetch_history(period='5y')
    coeffs = calibrate_s2f(history, inputs['s2f_intercept'], inputs['s2f_slope'])
    stock = inputs['circulating_supply']
    flow = inputs['block_reward'] * inputs['blocks_per_day'] * 365
    sf = stock / flow if flow > 0 else 100
    intrinsic_value = np.exp(coeffs[0] + coeffs[1] * sf) * (1 - inputs['volatility_adj'] / 100)
    return {'intrinsic_value': intrinsic_value}

def metcalfe_law(inputs):
    """Metcalfe's Law: value proportional to square of active addresses."""
    n = inputs['active_addresses']
    intrinsic_value = (n ** 2) * inputs['metcalfe_coeff'] * (1 - inputs['volatility_adj'] / 100)
    return {'intrinsic_value': intrinsic_value}

def nvt_model(inputs):
    """NVT: market cap to transaction volume, benchmarked to historical avg."""
    nvt = inputs['market_cap'] / inputs['transaction_volume'] if inputs['transaction_volume'] > 0 else 50
    intrinsic_value = inputs['current_price'] * (50 / nvt) * (1 - inputs['volatility_adj'] / 100)
    return {'intrinsic_value': intrinsic_value}

def pi_cycle(inputs):
    """Pi Cycle Top: uses 111-day and 350-day MAs."""
    ma_111 = inputs['50_day_ma'] * 1.1
    ma_350 = inputs['200_day_ma'] * 1.2
    intrinsic_value = ma_111 if inputs['current_price'] > ma_350 else inputs['current_price']
    intrinsic_value *= (1 - inputs['volatility_adj'] / 100)
    return {'intrinsic_value': intrinsic_value}

def reverse_s2f(inputs):
    """Reverse S2F: implied growth to reach current price."""
    history = fetch_history(period='5y')
    coeffs = calibrate_s2f(history, inputs['s2f_intercept'], inputs['s2f_slope'])
    stock = inputs['circulating_supply']
    flow = inputs['block_reward'] * inputs['blocks_per_day'] * 365
    sf = stock / flow if flow > 0 else 100
    target_price = inputs['current_price']
    implied_sf = (np.log(target_price) - coeffs[0]) / coeffs[1]
    intrinsic_value = target_price * (1 - inputs['volatility_adj'] / 100)
    return {'intrinsic_value': intrinsic_value, 'implied_sf': implied_sf}

def msc_model(inputs):
    """Market Sentiment Composite: Weighted sentiment index."""
    sentiment_index = (inputs['fear_greed'] / 100 * 0.4 + inputs['sentiment_score'] * 0.4 + (inputs['social_volume'] / 10000) * 0.2)
    intrinsic_value = inputs['current_price'] * (1 + sentiment_index) * (1 - inputs['volatility_adj'] / 100)
    return {'intrinsic_value': intrinsic_value, 'sentiment_index': sentiment_index}

def energy_value_model(inputs):
    """Bitcoin Energy Value: Floor price from mining costs."""
    intrinsic_value = inputs['mining_cost'] * (1 - inputs['volatility_adj'] / 100)
    return {'intrinsic_value': intrinsic_value}

def rvmr_model(inputs):
    """RVMR: Realized cap vs miner revenue."""
    miner_revenue = inputs['block_reward'] * inputs['blocks_per_day'] * 365 * inputs['current_price']
    rvmr = inputs['realized_cap'] / miner_revenue if miner_revenue > 0 else 50
    intrinsic_value = inputs['current_price'] * (50 / rvmr) * (1 - inputs['volatility_adj'] / 100)
    return {'intrinsic_value': intrinsic_value, 'rvmr': rvmr}

def mayer_multiple(inputs):
    """Mayer Multiple: Price to 200-day MA."""
    mayer = inputs['current_price'] / inputs['200_day_ma'] if inputs['200_day_ma'] > 0 else 1.0
    intrinsic_value = inputs['200_day_ma'] * 1.5 * (1 - inputs['volatility_adj'] / 100)  # Target fair multiple
    return {'intrinsic_value': intrinsic_value, 'mayer_multiple': mayer}

def hash_ribbons(inputs, history):
    """Hash Ribbons: Miner capitulation via hash rate MAs."""
    if history.empty or 'hash-rate' not in history.columns:
        return {'intrinsic_value': inputs['current_price'], 'hash_ribbon_signal': 'Hold'}
    hash_rate_30d = history['hash-rate'].rolling(30).mean().iloc[-1]
    hash_rate_60d = history['hash-rate'].rolling(60).mean().iloc[-1]
    signal = 'Buy' if hash_rate_30d > hash_rate_60d else 'Sell' if hash_rate_30d < hash_rate_60d * 0.9 else 'Hold'
    adjustment = 1.2 if signal == 'Buy' else 0.8 if signal == 'Sell' else 1.0
    intrinsic_value = inputs['current_price'] * adjustment * (1 - inputs['volatility_adj'] / 100)
    return {'intrinsic_value': intrinsic_value, 'hash_ribbon_signal': signal}
