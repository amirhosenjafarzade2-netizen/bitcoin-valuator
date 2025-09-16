import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import streamlit as st
from data_fetch import fetch_history

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calibrate_s2f(history, s2f_intercept=-1.84, s2f_slope=3.96):
    """
    Calibrate S2F model using historical data or user-provided coefficients.
    Uses ln(sf) for PlanB's model, updated for 2025.
    """
    try:
        if history.empty:
            logging.warning("Empty history for S2F calibration, using defaults")
            st.warning("S2F calibration failed: empty historical data")
            return [s2f_intercept, s2f_slope]
        stock = history['circulating_supply']
        flow = history['block_reward'] * history['blocks_per_day'] * 365
        sf = stock / flow
        sf = np.clip(sf, 1, 1000)  # Avoid extreme values
        log_sf = np.log(sf)
        log_price = np.log(history['Close'])
        valid = (~np.isnan(log_sf)) & (~np.isnan(log_price))
        if valid.sum() < 2:
            logging.warning("Insufficient valid data for S2F calibration")
            return [s2f_intercept, s2f_slope]
        coeffs = np.polyfit(log_sf[valid], log_price[valid], 1)
        logging.info(f"S2F calibrated: intercept={coeffs[1]:.2f}, slope={coeffs[0]:.2f}")
        st.success(f"S2F model calibrated successfully")
        return [coeffs[1], coeffs[0]]
    except Exception as e:
        logging.error(f"Error calibrating S2F: {str(e)}")
        st.error(f"S2F calibration failed: {str(e)}")
        return [s2f_intercept, s2f_slope]

def calibrate_metcalfe(history, metcalfe_coeff=15.0):
    """
    Calibrate Metcalfe coefficient using historical active addresses and price.
    """
    try:
        if history.empty:
            logging.warning("Empty history for Metcalfe calibration, using default")
            st.warning("Metcalfe calibration failed: empty historical data")
            return metcalfe_coeff
        n = history['n-unique-addresses']
        valid = (~np.isnan(n)) & (~np.isnan(history['Close']))
        if valid.sum() < 2:
            logging.warning("Insufficient valid data for Metcalfe calibration")
            return metcalfe_coeff
        market_value = history['Close'] * history['circulating_supply']
        coeff = np.mean(market_value[valid] / (n[valid] ** 2)) if (n[valid] ** 2).mean() > 0 else metcalfe_coeff
        logging.info(f"Metcalfe calibrated: coefficient={coeff:.2f}")
        st.success(f"Metcalfe model calibrated successfully")
        return coeff
    except Exception as e:
        logging.error(f"Error calibrating Metcalfe: {str(e)}")
        st.error(f"Metcalfe calibration failed: {str(e)}")
        return metcalfe_coeff

def calibrate_nvt(history, target_nvt=50):
    """
    Calibrate NVT target ratio using historical data.
    """
    try:
        if history.empty:
            logging.warning("Empty history for NVT calibration, using default")
            st.warning("NVT calibration failed: empty historical data")
            return target_nvt
        market_cap = history['Close'] * history['circulating_supply']
        tx_volume = history['estimated-transaction-volume-usd']
        valid = (~np.isnan(market_cap)) & (~np.isnan(tx_volume)) & (tx_volume > 0)
        if valid.sum() < 2:
            logging.warning("Insufficient valid data for NVT calibration")
            return target_nvt
        nvt = market_cap[valid] / tx_volume[valid]
        target_nvt = np.median(nvt) if nvt.size > 0 else target_nvt
        logging.info(f"NVT calibrated: target_nvt={target_nvt:.2f}")
        st.success(f"NVT model calibrated successfully")
        return target_nvt
    except Exception as e:
        logging.error(f"Error calibrating NVT: {str(e)}")
        st.error(f"NVT calibration failed: {str(e)}")
        return target_nvt

def calculate_valuation(inputs):
    """
    Calculate intrinsic value based on the selected model or ensemble.
    Returns a dictionary with per-BTC metrics and confidence scores.
    """
    model = inputs['model']
    current_price = inputs['current_price']
    mos = inputs['margin_of_safety']
    volatility_adj = inputs['volatility_adj']
    circulating_supply = inputs['circulating_supply']
    history = fetch_history(period='5y')
    
    try:
        # Model weights for ensemble (inspired by second program)
        model_weights = {
            'Stock-to-Flow (S2F)': 0.5,
            'Metcalfe\'s Law': 0.3,
            'Network Value to Transactions (NVT)': 0.2
        }
        
        # Calculate all models
        model_results = {
            's2f': s2f_model(inputs, history),
            'metcalfe': metcalfe_law(inputs, history),
            'nvt': nvt_model(inputs, history),
            'pi_cycle': pi_cycle(inputs),
            'reverse_s2f': reverse_s2f(inputs, history),
            'msc': msc_model(inputs),
            'energy_value': energy_value_model(inputs),
            'rvmr': rvmr_model(inputs),
            'mayer_multiple': mayer_multiple(inputs),
            'hash_ribbons': hash_ribbons(inputs, history),
            'macro_monetary': macro_monetary_model(inputs)
        }
        
        # Select primary model or ensemble
        if model == "Ensemble":
            intrinsic_value = sum(model_results[m]['intrinsic_value'] * w for m, w in model_weights.items())
            results = {'intrinsic_value': intrinsic_value, 'model': 'Ensemble'}
        else:
            results = model_results.get(model.lower().replace(' ', '_'), {'intrinsic_value': 0, 'error': 'Unknown model'})
        
        # Apply volatility and margin of safety adjustments
        intrinsic_value = results.get('intrinsic_value', 65000.0) * (1 - volatility_adj / 100)
        results['current_price'] = current_price
        results['intrinsic_value'] = max(intrinsic_value, 0) * (1 - mos / 100)
        results['safe_buy_price'] = results['intrinsic_value']
        results['undervaluation'] = ((intrinsic_value - current_price) / current_price * 100) if current_price > 0 else 0
        results['verdict'] = get_verdict(results['undervaluation'])
        results['score'] = calculate_score(inputs, results)
        
        # Model-specific metrics
        results['nvt_ratio'] = inputs['market_cap'] / inputs['transaction_volume'] if inputs['transaction_volume'] > 0 else 50
        results['mvrv_z_score'] = (inputs['mvrv'] - 2.0) / 0.5 if inputs['mvrv'] > 0 else 0
        results['sopr_signal'] = 'Buy' if inputs['sopr'] < 1 else 'Sell' if inputs['sopr'] > 1.2 else 'Hold'
        results['puell_signal'] = 'Buy' if inputs['puell_multiple'] < 0.5 else 'Sell' if inputs['puell_multiple'] > 4 else 'Hold'
        results['mining_cost_vs_price'] = (inputs['mining_cost'] - current_price) / current_price * 100 if current_price > 0 else 0
        
        # All model values and confidence scores
        all_models = {
            's2f_value': model_results['s2f'].get('intrinsic_value', 65000.0),
            'metcalfe_value': model_results['metcalfe'].get('intrinsic_value', 65000.0),
            'nvt_value': model_results['nvt'].get('intrinsic_value', 65000.0),
            'pi_cycle_value': model_results['pi_cycle'].get('intrinsic_value', 65000.0),
            'reverse_s2f_value': model_results['reverse_s2f'].get('intrinsic_value', 65000.0),
            'msc_value': model_results['msc'].get('intrinsic_value', 65000.0),
            'energy_value': model_results['energy_value'].get('intrinsic_value', 65000.0),
            'rvmr_value': model_results['rvmr'].get('intrinsic_value', 65000.0),
            'mayer_multiple_value': model_results['mayer_multiple'].get('intrinsic_value', 65000.0),
            'hash_ribbons_value': model_results['hash_ribbons'].get('intrinsic_value', 65000.0),
            'macro_monetary_value': model_results['macro_monetary'].get('intrinsic_value', 65000.0)
        }
        # Assign confidence scores (simplified, based on historical fit)
        confidence_scores = {
            's2f_value': 0.9 if not history.empty else 0.7,
            'metcalfe_value': 0.8 if not history.empty else 0.6,
            'nvt_value': 0.7 if not history.empty else 0.5,
            'pi_cycle_value': 0.6,
            'reverse_s2f_value': 0.6,
            'msc_value': 0.5,
            'energy_value': 0.7,
            'rvmr_value': 0.6,
            'mayer_multiple_value': 0.6,
            'hash_ribbons_value': 0.7 if not history.empty else 0.5,
            'macro_monetary_value': 0.6
        }
        results.update(all_models)
        results.update({f"{k}_confidence": v for k, v in confidence_scores.items()})
        
        logging.info(f"Valuation completed for {model}: intrinsic_value=${results['intrinsic_value']:,.2f}")
        st.success(f"Valuation completed for {model}: Intrinsic value ${results['intrinsic_value']:,.2f}")
        return results
    
    except Exception as e:
        logging.error(f"Error in valuation: {str(e)}")
        st.error(f"Valuation failed: {str(e)}")
        return {
            'intrinsic_value': 65000.0,  # Realistic fallback for 2025
            'current_price': current_price,
            'safe_buy_price': 65000.0 * (1 - mos / 100),
            'undervaluation': 0,
            'verdict': 'Hold',
            'score': 50,
            'error': f'Calculation failed: {str(e)}',
            'source': 'fallback'
        }

def get_verdict(undervaluation):
    """Determine buy/sell/hold verdict with tighter thresholds."""
    if undervaluation > 15:
        return "Strong Buy"
    elif undervaluation > 5:
        return "Buy"
    elif undervaluation > -10:
        return "Hold"
    else:
        return "Sell"

def calculate_score(inputs, results):
    """Calculate an overall score based on metrics, weighted for 2025."""
    score = 50
    if results['undervaluation'] > 0:
        score += min(results['undervaluation'] * 0.5, 20)  # Reduced weight for stability
    if inputs['fear_greed'] < 30:
        score += 15
    elif inputs['fear_greed'] > 70:
        score -= 15
    if inputs['sopr'] < 0.95:
        score += 10
    elif inputs['sopr'] > 1.15:
        score -= 10
    if inputs['puell_multiple'] < 0.4:
        score += 10
    elif inputs['puell_multiple'] > 3:
        score -= 10
    if inputs['rsi'] < 30:
        score += 10
    elif inputs['rsi'] > 70:
        score -= 10
    return max(min(score, 100), 0)

def s2f_model(inputs, history):
    """Stock-to-Flow model: price based on scarcity, calibrated for 2025."""
    try:
        coeffs = calibrate_s2f(history, inputs['s2f_intercept'], inputs['s2f_slope'])
        stock = inputs['circulating_supply']
        flow = inputs['block_reward'] * inputs['blocks_per_day'] * 365
        sf = stock / flow if flow > 0 else 100
        intrinsic_value = np.exp(coeffs[0] + coeffs[1] * np.log(sf)) / inputs['circulating_supply']
        return {'intrinsic_value': max(intrinsic_value, 0), 'stock_to_flow': sf}
    except Exception as e:
        logging.error(f"S2F model failed: {str(e)}")
        return {'intrinsic_value': 65000.0, 'stock_to_flow': 100}

def metcalfe_law(inputs, history):
    """Metcalfe's Law: value proportional to square of active addresses."""
    try:
        coeff = calibrate_metcalfe(history, inputs['metcalfe_coeff'])
        n = inputs['active_addresses']
        intrinsic_value = (n ** 2) * coeff / inputs['circulating_supply']
        return {'intrinsic_value': max(intrinsic_value, 0), 'metcalfe_coeff': coeff}
    except Exception as e:
        logging.error(f"Metcalfe model failed: {str(e)}")
        return {'intrinsic_value': 65000.0, 'metcalfe_coeff': inputs['metcalfe_coeff']}

def nvt_model(inputs, history):
    """NVT: market cap to transaction volume, benchmarked to historical median."""
    try:
        target_nvt = calibrate_nvt(history)
        nvt = inputs['market_cap'] / inputs['transaction_volume'] if inputs['transaction_volume'] > 0 else target_nvt
        intrinsic_value = inputs['current_price'] * (target_nvt / nvt)
        return {'intrinsic_value': max(intrinsic_value, 0), 'nvt_ratio': nvt}
    except Exception as e:
        logging.error(f"NVT model failed: {str(e)}")
        return {'intrinsic_value': 65000.0, 'nvt_ratio': 50}

def pi_cycle(inputs):
    """Pi Cycle Top: uses 111-day and 350-day MAs, adjusted for 2025."""
    try:
        # Use historical data to estimate MAs if available
        history = fetch_history(period='1y')
        ma_111 = history['Close'].rolling(111).mean().iloc[-1] if not history.empty else inputs['50_day_ma'] * 1.05
        ma_350 = history['Close'].rolling(350).mean().iloc[-1] if not history.empty else inputs['200_day_ma'] * 1.1
        signal = 'Buy' if inputs['current_price'] < ma_111 else 'Sell' if inputs['current_price'] > ma_350 else 'Hold'
        adjustment = 1.1 if signal == 'Buy' else 0.9 if signal == 'Sell' else 1.0
        intrinsic_value = inputs['current_price'] * adjustment
        return {'intrinsic_value': max(intrinsic_value, 0), 'pi_cycle_signal': signal}
    except Exception as e:
        logging.error(f"Pi Cycle model failed: {str(e)}")
        return {'intrinsic_value': 65000.0, 'pi_cycle_signal': 'Hold'}

def reverse_s2f(inputs, history):
    """Reverse S2F: implied growth to reach target price."""
    try:
        coeffs = calibrate_s2f(history, inputs['s2f_intercept'], inputs['s2f_slope'])
        stock = inputs['circulating_supply']
        flow = inputs['block_reward'] * inputs['blocks_per_day'] * 365
        sf = stock / flow if flow > 0 else 100
        target_price = inputs['current_price'] * 1.3  # Reduced for realism
        implied_sf = (np.log(target_price * inputs['circulating_supply']) - coeffs[0]) / coeffs[1]
        intrinsic_value = target_price
        return {'intrinsic_value': max(intrinsic_value, 0), 'implied_sf': implied_sf}
    except Exception as e:
        logging.error(f"Reverse S2F model failed: {str(e)}")
        return {'intrinsic_value': 65000.0, 'implied_sf': 100}

def msc_model(inputs):
    """Market Sentiment Composite: Weighted sentiment index."""
    try:
        sentiment_index = (inputs['fear_greed'] / 100 * 0.5 + inputs['sentiment_score'] * 0.3 + (inputs['social_volume'] / 20000) * 0.2)
        intrinsic_value = inputs['current_price'] * (1 + sentiment_index * 0.5)  # Reduced impact
        return {'intrinsic_value': max(intrinsic_value, 0), 'sentiment_index': sentiment_index}
    except Exception as e:
        logging.error(f"MSC model failed: {str(e)}")
        return {'intrinsic_value': 65000.0, 'sentiment_index': 0.5}

def energy_value_model(inputs):
    """Bitcoin Energy Value: Floor price from mining costs."""
    try:
        intrinsic_value = inputs['mining_cost'] * 1.2  # Slight premium for profitability
        return {'intrinsic_value': max(intrinsic_value, 0)}
    except Exception as e:
        logging.error(f"Energy Value model failed: {str(e)}")
        return {'intrinsic_value': 65000.0}

def rvmr_model(inputs):
    """RVMR: Realized cap vs miner revenue."""
    try:
        miner_revenue = inputs['block_reward'] * inputs['blocks_per_day'] * 365 * inputs['current_price']
        rvmr = inputs['realized_cap'] / miner_revenue if miner_revenue > 0 else 50
        intrinsic_value = inputs['current_price'] * (50 / rvmr)
        return {'intrinsic_value': max(intrinsic_value, 0), 'rvmr': rvmr}
    except Exception as e:
        logging.error(f"RVMR model failed: {str(e)}")
        return {'intrinsic_value': 65000.0, 'rvmr': 50}

def mayer_multiple(inputs):
    """Mayer Multiple: Price to 200-day MA."""
    try:
        mayer = inputs['current_price'] / inputs['200_day_ma'] if inputs['200_day_ma'] > 0 else 1.0
        target_multiple = 1.4  # Adjusted for 2025 market conditions
        intrinsic_value = inputs['200_day_ma'] * target_multiple
        return {'intrinsic_value': max(intrinsic_value, 0), 'mayer_multiple': mayer}
    except Exception as e:
        logging.error(f"Mayer Multiple model failed: {str(e)}")
        return {'intrinsic_value': 65000.0, 'mayer_multiple': 1.0}

def hash_ribbons(inputs, history):
    """Hash Ribbons: Miner capitulation via hash rate MAs."""
    try:
        if history.empty or 'hash_rate' not in history.columns:
            logging.warning("Empty or invalid history for Hash Ribbons")
            st.warning("Hash Ribbons calculation failed: no historical data")
            return {'intrinsic_value': 65000.0, 'hash_ribbon_signal': 'Hold'}
        hash_rate_30d = history['hash_rate_30d'].iloc[-1]
        hash_rate_60d = history['hash_rate_60d'].iloc[-1]
        signal = 'Buy' if hash_rate_30d > hash_rate_60d else 'Sell' if hash_rate_30d < hash_rate_60d * 0.95 else 'Hold'
        adjustment = 1.15 if signal == 'Buy' else 0.85 if signal == 'Sell' else 1.0
        intrinsic_value = inputs['current_price'] * adjustment
        return {'intrinsic_value': max(intrinsic_value, 0), 'hash_ribbon_signal': signal}
    except Exception as e:
        logging.error(f"Hash Ribbons model failed: {str(e)}")
        st.error(f"Hash Ribbons model failed: {str(e)}")
        return {'intrinsic_value': 65000.0, 'hash_ribbon_signal': 'Hold'}

def macro_monetary_model(inputs):
    """Macro Monetary Model: Adjusts Metcalfe's base value by macro factors."""
    try:
        n = inputs['active_addresses']
        base_value = (n ** 2) * inputs['metcalfe_coeff'] / inputs['circulating_supply']
        inflation_premium = inputs['us_inflation'] / 100 * 0.3  # Reduced sensitivity
        real_rate = inputs['fed_rate'] - inputs['us_inflation']
        rate_discount = max(real_rate / 100 * 0.5, 0)  # Adjusted weight
        intrinsic_value = base_value * (1 + inflation_premium - rate_discount)
        return {
            'intrinsic_value': max(intrinsic_value, 0),
            'inflation_premium': inflation_premium,
            'real_rate_discount': rate_discount
        }
    except Exception as e:
        logging.error(f"Macro Monetary model failed: {str(e)}")
        st.error(f"Macro Monetary model failed: {str(e)}")
        return {'intrinsic_value': 65000.0, 'inflation_premium': 0, 'real_rate_discount': 0}
