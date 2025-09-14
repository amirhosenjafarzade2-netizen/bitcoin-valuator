import pandas as pd
from datetime import datetime
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_inputs(inputs):
    """
    Validate inputs for valuation models.
    Args:
        inputs (dict): Dictionary of input parameters.
    Returns:
        tuple: (is_valid: bool, errors: list of str)
    """
    errors = []
    
    # Required keys
    required_keys = [
        'model', 'current_price', 'total_supply', 'circulating_supply', 'next_halving_date',
        'margin_of_safety', 'hash_rate', 'active_addresses', 'transaction_volume', 'mvrv',
        'sopr', 'realized_cap', 'puell_multiple', 'mining_cost', 'fear_greed',
        'social_volume', 'sentiment_score', 'us_inflation', 'fed_rate', 'sp_correlation',
        'gold_price', 'rsi', '50_day_ma', '200_day_ma', 'desired_return',
        'monte_carlo_runs', 'volatility_adj', 'growth_adj', 'beta', 'market_cap',
        's2f_intercept', 's2f_slope', 'metcalfe_coeff', 'block_reward', 'blocks_per_day',
        'electricity_cost'
    ]
    
    # Check missing or None values
    for key in required_keys:
        if key not in inputs or inputs[key] is None:
            errors.append(f"Missing or None value for {key}")
    
    # Type checking
    numeric_keys = [
        'current_price', 'total_supply', 'circulating_supply', 'margin_of_safety',
        'hash_rate', 'active_addresses', 'transaction_volume', 'mvrv', 'sopr',
        'realized_cap', 'puell_multiple', 'mining_cost', 'social_volume',
        'sentiment_score', 'us_inflation', 'fed_rate', 'sp_correlation', 'gold_price',
        'rsi', '50_day_ma', '200_day_ma', 'desired_return', 'volatility_adj',
        'growth_adj', 'beta', 'market_cap', 's2f_intercept', 's2f_slope',
        'metcalfe_coeff', 'block_reward', 'blocks_per_day', 'electricity_cost'
    ]
    for key in numeric_keys:
        if key in inputs and inputs[key] is not None:
            try:
                inputs[key] = float(inputs[key])
            except (TypeError, ValueError):
                errors.append(f"Invalid type for {key}: expected float, got {type(inputs[key]).__name__}")
    
    if 'fear_greed' in inputs and inputs['fear_greed'] is not None:
        try:
            inputs['fear_greed'] = int(inputs['fear_greed'])
        except (TypeError, ValueError):
            errors.append(f"Invalid type for fear_greed: expected int, got {type(inputs['fear_greed']).__name__}")
    
    if 'monte_carlo_runs' in inputs and inputs['monte_carlo_runs'] is not None:
        try:
            inputs['monte_carlo_runs'] = int(inputs['monte_carlo_runs'])
        except (TypeError, ValueError):
            errors.append(f"Invalid type for monte_carlo_runs: expected int, got {type(inputs['monte_carlo_runs']).__name__}")
    
    if 'next_halving_date' in inputs and inputs['next_halving_date'] is not None:
        if not isinstance(inputs['next_halving_date'], datetime):
            try:
                inputs['next_halving_date'] = datetime.strptime(str(inputs['next_halving_date']), '%Y-%m-%d')
            except (TypeError, ValueError):
                errors.append(f"Invalid type for next_halving_date: expected datetime, got {type(inputs['next_halving_date']).__name__}")
    
    # Numeric constraints
    if 'current_price' in inputs and isinstance(inputs['current_price'], (int, float)) and inputs['current_price'] <= 0:
        errors.append("Current price must be positive")
    elif inputs.get('current_price', 0) < 10000 or inputs.get('current_price', 0) > 100000:
        logging.warning(f"Current price ${inputs['current_price']:.2f} is unusual, expected $60,000-$65,000")
    
    if 'circulating_supply' in inputs and isinstance(inputs['circulating_supply'], (int, float)) and inputs['circulating_supply'] <= 0:
        errors.append("Circulating supply must be positive")
    elif inputs.get('circulating_supply', 0) < 19000000 or inputs.get('circulating_supply', 0) > 21000000:
        logging.warning(f"Circulating supply {inputs['circulating_supply']} is unusual, expected 19.5M-20M")
    
    if 'total_supply' in inputs and isinstance(inputs['total_supply'], (int, float)) and inputs['total_supply'] <= 0:
        errors.append("Total supply must be positive")
    elif inputs.get('total_supply', 0) != 21000000:
        logging.warning(f"Total supply {inputs['total_supply']} is not 21M")
    
    if 'hash_rate' in inputs and isinstance(inputs['hash_rate'], (int, float)) and inputs['hash_rate'] < 0:
        errors.append("Hash rate cannot be negative")
    elif inputs.get('hash_rate', 0) < 400 or inputs.get('hash_rate', 0) > 700:
        logging.warning(f"Hash rate {inputs['hash_rate']} EH/s is unusual, expected 500-600")
    
    if 'active_addresses' in inputs and isinstance(inputs['active_addresses'], (int, float)) and inputs['active_addresses'] < 0:
        errors.append("Active addresses cannot be negative")
    elif inputs.get('active_addresses', 0) < 500000 or inputs.get('active_addresses', 0) > 1500000:
        logging.warning(f"Active addresses {inputs['active_addresses']} is unusual, expected 800K-1M")
    
    if 'transaction_volume' in inputs and isinstance(inputs['transaction_volume'], (int, float)) and inputs['transaction_volume'] < 0:
        errors.append("Transaction volume cannot be negative")
    elif inputs.get('transaction_volume', 0) < 5e8 or inputs.get('transaction_volume', 0) > 1e10:
        logging.warning(f"Transaction volume ${inputs['transaction_volume']:.2e} is unusual, expected $1B-$5B")
    
    if 'mvrv' in inputs and isinstance(inputs['mvrv'], (int, float)) and inputs['mvrv'] < 0:
        errors.append("MVRV cannot be negative")
    elif inputs.get('mvrv', 0) < 0.5 or inputs.get('mvrv', 0) > 5.0:
        logging.warning(f"MVRV {inputs['mvrv']} is unusual, expected 1.5-2.5")
    
    if 'sopr' in inputs and isinstance(inputs['sopr'], (int, float)) and inputs['sopr'] < 0:
        errors.append("SOPR cannot be negative")
    elif inputs.get('sopr', 0) < 0.5 or inputs.get('sopr', 0) > 1.5:
        logging.warning(f"SOPR {inputs['sopr']} is unusual, expected 0.9-1.1")
    
    if 'realized_cap' in inputs and isinstance(inputs['realized_cap'], (int, float)) and inputs['realized_cap'] < 0:
        errors.append("Realized cap cannot be negative")
    elif inputs.get('realized_cap', 0) < 4e11 or inputs.get('realized_cap', 0) > 1e12:
        logging.warning(f"Realized cap ${inputs['realized_cap']:.2e} is unusual, expected $600B-$800B")
    
    if 'puell_multiple' in inputs and isinstance(inputs['puell_multiple'], (int, float)) and inputs['puell_multiple'] < 0:
        errors.append("Puell multiple cannot be negative")
    elif inputs.get('puell_multiple', 0) < 0.3 or inputs.get('puell_multiple', 0) > 5.0:
        logging.warning(f"Puell multiple {inputs['puell_multiple']} is unusual, expected 0.5-1.5")
    
    if 'mining_cost' in inputs and isinstance(inputs['mining_cost'], (int, float)) and inputs['mining_cost'] < 0:
        errors.append("Mining cost cannot be negative")
    elif inputs.get('mining_cost', 0) < 5000 or inputs.get('mining_cost', 0) > 15000:
        logging.warning(f"Mining cost ${inputs['mining_cost']} is unusual, expected $5,000-$15,000")
    
    if 'fear_greed' in inputs and isinstance(inputs['fear_greed'], int) and (inputs['fear_greed'] < 0 or inputs['fear_greed'] > 100):
        errors.append("Fear & Greed must be between 0 and 100")
    elif inputs.get('fear_greed', 0) < 20 or inputs.get('fear_greed', 0) > 80:
        logging.warning(f"Fear & Greed {inputs['fear_greed']} is unusual, expected 40-60")
    
    if 'social_volume' in inputs and isinstance(inputs['social_volume'], (int, float)) and inputs['social_volume'] < 0:
        errors.append("Social volume cannot be negative")
    
    if 'sentiment_score' in inputs and isinstance(inputs['sentiment_score'], (int, float)) and (inputs['sentiment_score'] < -1 or inputs['sentiment_score'] > 1):
        errors.append("Sentiment score must be between -1 and 1")
    
    if 'us_inflation' in inputs and isinstance(inputs['us_inflation'], (int, float)) and inputs['us_inflation'] < 0:
        errors.append("US inflation cannot be negative")
    elif inputs.get('us_inflation', 0) > 10:
        logging.warning(f"US inflation {inputs['us_inflation']}% is unusual, expected 2.5-3.5%")
    
    if 'fed_rate' in inputs and isinstance(inputs['fed_rate'], (int, float)) and inputs['fed_rate'] < 0:
        errors.append("Fed rate cannot be negative")
    elif inputs.get('fed_rate', 0) > 10:
        logging.warning(f"Fed rate {inputs['fed_rate']}% is unusual, expected 4.5-5.0%")
    
    if 'sp_correlation' in inputs and isinstance(inputs['sp_correlation'], (int, float)) and (inputs['sp_correlation'] < 0 or inputs['sp_correlation'] > 1):
        errors.append("S&P 500 correlation must be between 0 and 1")
    
    if 'gold_price' in inputs and isinstance(inputs['gold_price'], (int, float)) and inputs['gold_price'] < 0:
        errors.append("Gold price cannot be negative")
    elif inputs.get('gold_price', 0) < 1500 or inputs.get('gold_price', 0) > 3000:
        logging.warning(f"Gold price ${inputs['gold_price']} is unusual, expected $2,000-$2,500")
    
    if 'rsi' in inputs and isinstance(inputs['rsi'], (int, float)) and (inputs['rsi'] < 0 or inputs['rsi'] > 100):
        errors.append("RSI must be between 0 and 100")
    elif inputs.get('rsi', 0) < 10 or inputs.get('rsi', 0) > 90:
        logging.warning(f"RSI {inputs['rsi']} is unusual, expected 40-60")
    
    if '50_day_ma' in inputs and isinstance(inputs['50_day_ma'], (int, float)) and inputs['50_day_ma'] < 0:
        errors.append("50-day MA cannot be negative")
    elif inputs.get('50_day_ma', 0) < 40000 or inputs.get('50_day_ma', 0) > 80000:
        logging.warning(f"50-day MA ${inputs['50_day_ma']} is unusual, expected $55,000-$65,000")
    
    if '200_day_ma' in inputs and isinstance(inputs['200_day_ma'], (int, float)) and inputs['200_day_ma'] < 0:
        errors.append("200-day MA cannot be negative")
    elif inputs.get('200_day_ma', 0) < 40000 or inputs.get('200_day_ma', 0) > 80000:
        logging.warning(f"200-day MA ${inputs['200_day_ma']} is unusual, expected $50,000-$60,000")
    
    if 'desired_return' in inputs and isinstance(inputs['desired_return'], (int, float)) and inputs['desired_return'] < 0:
        errors.append("Desired return cannot be negative")
    
    if 'monte_carlo_runs' in inputs and isinstance(inputs['monte_carlo_runs'], int) and inputs['monte_carlo_runs'] < 100:
        errors.append("Monte Carlo runs must be at least 100")
    
    if 'volatility_adj' in inputs and isinstance(inputs['volatility_adj'], (int, float)) and inputs['volatility_adj'] < 0:
        errors.append("Volatility adjustment cannot be negative")
    elif inputs.get('volatility_adj', 0) > 50:
        logging.warning(f"Volatility adjustment {inputs['volatility_adj']}% is unusual, expected 10-30%")
    
    if 'growth_adj' in inputs and isinstance(inputs['growth_adj'], (int, float)) and inputs['growth_adj'] < 0:
        errors.append("Growth adjustment cannot be negative")
    elif inputs.get('growth_adj', 0) > 50:
        logging.warning(f"Growth adjustment {inputs['growth_adj']}% is unusual, expected 10-30%")
    
    if 'beta' in inputs and isinstance(inputs['beta'], (int, float)) and inputs['beta'] < 0:
        errors.append("Beta cannot be negative")
    
    if 'market_cap' in inputs and isinstance(inputs['market_cap'], (int, float)) and inputs['market_cap'] < 0:
        errors.append("Market cap cannot be negative")
    elif abs(inputs.get('market_cap', 0) - inputs.get('current_price', 0) * inputs.get('circulating_supply', 0)) > 1e9:
        logging.warning(f"Market cap ${inputs['market_cap']:.2e} inconsistent with price*supply")
    
    if 's2f_intercept' in inputs and isinstance(inputs['s2f_intercept'], (int, float)):
        if inputs['s2f_intercept'] < -10 or inputs['s2f_intercept'] > 10:
            logging.warning(f"S2F intercept {inputs['s2f_intercept']} is unusual, expected -1.84 for ln(sf) model")
    
    if 's2f_slope' in inputs and isinstance(inputs['s2f_slope'], (int, float)) and inputs['s2f_slope'] < 0:
        errors.append("S2F slope cannot be negative")
    elif inputs.get('s2f_slope', 0) > 10:
        logging.warning(f"S2F slope {inputs['s2f_slope']} is unusual, expected 3.96 for ln(sf) model")
    
    if 'metcalfe_coeff' in inputs and isinstance(inputs['metcalfe_coeff'], (int, float)) and inputs['metcalfe_coeff'] < 0:
        errors.append("Metcalfe coefficient cannot be negative")
    elif inputs.get('metcalfe_coeff', 0) < 0.1 or inputs.get('metcalfe_coeff', 0) > 10:
        logging.warning(f"Metcalfe coefficient {inputs['metcalfe_coeff']} is unusual, expected 1.0 for ~$50,000-$70,000/BTC")
    
    if 'block_reward' in inputs and isinstance(inputs['block_reward'], (int, float)) and inputs['block_reward'] < 0:
        errors.append("Block reward cannot be negative")
    elif inputs.get('block_reward', 0) not in [50.0, 25.0, 12.5, 6.25, 3.125]:
        logging.warning(f"Block reward {inputs['block_reward']} is unusual, expected 3.125 post-2024 halving")
    
    if 'blocks_per_day' in inputs and isinstance(inputs['blocks_per_day'], (int, float)) and inputs['blocks_per_day'] < 0:
        errors.append("Blocks per day cannot be negative")
    elif inputs.get('blocks_per_day', 0) < 100 or inputs.get('blocks_per_day', 0) > 200:
        logging.warning(f"Blocks per day {inputs['blocks_per_day']} is unusual, expected ~144")
    
    if 'electricity_cost' in inputs and isinstance(inputs['electricity_cost'], (int, float)) and inputs['electricity_cost'] < 0:
        errors.append("Electricity cost cannot be negative")
    elif inputs.get('electricity_cost', 0) > 1.0:
        logging.warning(f"Electricity cost ${inputs['electricity_cost']} is unusual, expected $0.05-$0.10/kWh")
    
    return len(errors) == 0, errors

def export_portfolio(portfolio):
    """
    Export portfolio to CSV.
    Args:
        portfolio (pd.DataFrame): Portfolio DataFrame.
    """
    try:
        portfolio.to_csv("portfolio_export.csv", index=False)
        logging.info("Portfolio exported to portfolio_export.csv")
    except Exception as e:
        logging.error(f"Portfolio export failed: {str(e)}")
        raise Exception(f"Failed to export portfolio: {str(e)}")

def generate_pdf_report(results, portfolio, model_comp_fig):
    """
    Generate a PDF report of valuation results and portfolio.
    Args:
        results (dict): Valuation results.
        portfolio (pd.DataFrame): Portfolio DataFrame.
        model_comp_fig: Plotly figure for model comparison.
    Returns:
        bytes: PDF content.
    """
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(100, 750, "Bitcoin Valuation Report")
        c.drawString(100, 730, f"Model: {results.get('model', '-')}")
        c.drawString(100, 710, f"Intrinsic Value: ${results.get('intrinsic_value', 0):.2f}")
        c.drawString(100, 690, f"Undervaluation: {results.get('undervaluation', 0):.2f}%")
        c.drawString(100, 670, f"Verdict: {results.get('verdict', '-')}")
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        logging.error(f"PDF report generation failed: {str(e)}")
        raise Exception(f"Failed to generate PDF report: {str(e)}")
```
