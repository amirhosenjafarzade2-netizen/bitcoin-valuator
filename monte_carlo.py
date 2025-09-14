import numpy as np
from valuation_models import calculate_valuation
from utils import validate_inputs
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def run_monte_carlo(inputs, num_runs=1000, volatility_adj=30.0, growth_adj=20.0):
    """
    Run Monte Carlo simulation by varying volatility, growth rate, hash rate, fear_greed, realized_cap, us_inflation, and fed_rate.
    Returns a dictionary with results.
    """
    try:
        if num_runs <= 0:
            return {
                'values': np.array([]),
                'avg_value': 0,
                'std_dev': 0,
                'prob_undervalued': 0,
                'num_runs': 0
            }
        
        model = inputs['model']
        base_volatility = inputs['volatility_adj']
        base_growth = inputs['growth_adj']
        base_hash_rate = inputs['hash_rate']
        base_fear_greed = inputs['fear_greed']
        base_realized_cap = inputs['realized_cap']
        base_us_inflation = inputs['us_inflation']
        base_fed_rate = inputs['fed_rate']
        
        volatility_variations = np.clip(
            np.random.normal(base_volatility, volatility_adj / 2, num_runs),
            max(0, base_volatility - volatility_adj),
            base_volatility + volatility_adj
        )
        growth_variations = np.clip(
            np.random.normal(base_growth, growth_adj / 2, num_runs),
            max(0, base_growth - growth_adj),
            base_growth + growth_adj
        )
        hash_rate_variations = np.clip(
            np.random.normal(base_hash_rate, base_hash_rate * 0.1, num_runs),
            base_hash_rate * 0.5,
            base_hash_rate * 1.5
        )
        fear_greed_variations = np.clip(
            np.random.normal(base_fear_greed, 20, num_runs),
            0, 100
        )
        realized_cap_variations = np.clip(
            np.random.normal(base_realized_cap, base_realized_cap * 0.1, num_runs),
            base_realized_cap * 0.5,
            base_realized_cap * 1.5
        )
        us_inflation_variations = np.clip(
            np.random.normal(base_us_inflation, 1.0, num_runs),
            max(0, base_us_inflation - 2),
            base_us_inflation + 2
        )
        fed_rate_variations = np.clip(
            np.random.normal(base_fed_rate, 1.0, num_runs),
            max(0, base_fed_rate - 2),
            base_fed_rate + 2
        )
        
        intrinsic_values = []
        for i in range(num_runs):
            sim_inputs = inputs.copy()
            sim_inputs['volatility_adj'] = volatility_variations[i]
            sim_inputs['growth_adj'] = growth_variations[i]
            sim_inputs['hash_rate'] = hash_rate_variations[i]
            sim_inputs['fear_greed'] = fear_greed_variations[i]
            sim_inputs['realized_cap'] = realized_cap_variations[i]
            sim_inputs['us_inflation'] = us_inflation_variations[i]
            sim_inputs['fed_rate'] = fed_rate_variations[i]
            sim_inputs['mining_cost'] = sim_inputs['hash_rate'] * 1000 * sim_inputs['electricity_cost'] * 24 * 365 / (sim_inputs['block_reward'] * sim_inputs['blocks_per_day'])
            
            if validate_inputs(sim_inputs):
                sim_results = calculate_valuation(sim_inputs)
                intrinsic_values.append(sim_results.get('intrinsic_value', 0))
            else:
                intrinsic_values.append(0)
        
        intrinsic_values = np.array(intrinsic_values)
        avg_value = np.mean(intrinsic_values)
        std_dev = np.std(intrinsic_values)
        current_price = inputs['current_price']
        prob_undervalued = np.mean(intrinsic_values > current_price) * 100 if current_price > 0 else 0
        
        return {
            'values': intrinsic_values,
            'avg_value': avg_value,
            'std_dev': std_dev,
            'prob_undervalued': prob_undervalued,
            'num_runs': num_runs
        }
    
    except Exception as e:
        logging.error(f"Error in Monte Carlo simulation: {str(e)}")
        return {
            'values': np.array([]),
            'avg_value': 0,
            'std_dev': 0,
            'prob_undervalued': 0,
            'num_runs': 0
        }
