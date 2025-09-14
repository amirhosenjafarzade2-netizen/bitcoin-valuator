import numpy as np
from valuation_models import calculate_valuation

def run_monte_carlo(inputs, num_runs=1000, volatility_adj=30.0, growth_adj=20.0):
    """
    Run Monte Carlo simulation by varying volatility and growth rate.
    Returns a dictionary with results.
    """
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
    
    intrinsic_values = []
    for i in range(num_runs):
        sim_inputs = inputs.copy()
        sim_inputs['volatility_adj'] = volatility_variations[i]
        sim_inputs['growth_adj'] = growth_variations[i]
        
        if validate_sim_inputs(sim_inputs, model):
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

def validate_sim_inputs(inputs, model):
    """
    Validate inputs for Monte Carlo runs.
    """
    if inputs['volatility_adj'] < 0 or inputs['growth_adj'] < 0:
        return False
    return True
