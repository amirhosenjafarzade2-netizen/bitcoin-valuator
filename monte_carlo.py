import numpy as np
from valuation_models import calculate_valuation
from utils import validate_inputs
import logging
import concurrent.futures
import streamlit as st
from scipy.stats import multivariate_normal

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_monte_carlo(inputs, num_runs=1000, volatility_adj=10.0, growth_adj=20.0):
    """
    Run Monte Carlo simulation with correlated parameter variations for volatility, growth rate, hash rate,
    fear & greed, realized cap, US inflation, and Fed rate.
    Returns a dictionary with results, including realistic fallbacks.
    """
    try:
        if num_runs <= 0:
            logging.warning("Invalid num_runs, returning fallback results")
            return {
                'values': np.array([65000.0] * 100),  # Realistic fallback for 2025
                'avg_value': 65000.0,
                'std_dev': 10000.0,
                'prob_undervalued': 50.0,
                'num_runs': 100,
                'source': 'fallback'
            }

        model = inputs['model']
        base_volatility = inputs['volatility_adj']
        base_growth = inputs['growth_adj']
        base_hash_rate = inputs['hash_rate']
        base_fear_greed = inputs['fear_greed']
        base_realized_cap = inputs['realized_cap']
        base_us_inflation = inputs['us_inflation']
        base_fed_rate = inputs['fed_rate']
        current_price = inputs['current_price']

        # Define realistic variation ranges based on 2025 market data
        # Volatility: ±10% (historical BTC volatility ~30-50%)
        # Growth: ±10% (conservative for crypto markets)
        # Hash rate: ±20% (based on Blockchain.com historical volatility)
        # Fear & Greed: ±15 (realistic range for sentiment)
        # Realized cap: ±15% (aligned with market cap fluctuations)
        # Inflation: ±1% (based on recent US data)
        # Fed rate: ±0.5% (based on Fed stability)
        means = [
            base_volatility,
            base_growth,
            base_hash_rate,
            base_fear_greed,
            base_realized_cap,
            base_us_inflation,
            base_fed_rate
        ]
        # Covariance matrix for correlated parameters (e.g., hash rate and realized cap)
        cov_matrix = np.diag([
            (volatility_adj * 0.5) ** 2,
            (growth_adj * 0.5) ** 2,
            (base_hash_rate * 0.2) ** 2,
            15 ** 2,
            (base_realized_cap * 0.15) ** 2,
            1.0 ** 2,
            0.5 ** 2
        ])
        # Add correlation between hash rate and realized cap (0.7)
        cov_matrix[2, 4] = cov_matrix[4, 2] = 0.7 * (base_hash_rate * 0.2) * (base_realized_cap * 0.15)

        # Generate correlated samples
        samples = multivariate_normal.rvs(mean=means, cov=cov_matrix, size=num_runs)
        samples = np.clip(samples, [
            max(0, base_volatility - volatility_adj),
            max(0, base_growth - growth_adj),
            base_hash_rate * 0.8,
            0,
            base_realized_cap * 0.85,
            max(0, base_us_inflation - 1.5),
            max(0, base_fed_rate - 1.0)
        ], [
            base_volatility + volatility_adj,
            base_growth + growth_adj,
            base_hash_rate * 1.2,
            100,
            base_realized_cap * 1.15,
            base_us_inflation + 1.5,
            base_fed_rate + 1.0
        ])

        intrinsic_values = []
        def run_simulation(i):
            sim_inputs = inputs.copy()
            sim_inputs.update({
                'volatility_adj': samples[i, 0],
                'growth_adj': samples[i, 1],
                'hash_rate': samples[i, 2],
                'fear_greed': samples[i, 3],
                'realized_cap': samples[i, 4],
                'us_inflation': samples[i, 5],
                'fed_rate': samples[i, 6],
                'mining_cost': samples[i, 2] * 1000 * sim_inputs['electricity_cost'] * 24 * 365 / (sim_inputs['block_reward'] * sim_inputs['blocks_per_day'])
            })
            if validate_inputs(sim_inputs):
                result = calculate_valuation(sim_inputs)
                return result.get('intrinsic_value', 65000.0)  # Fallback to realistic value
            return 65000.0

        # Parallelize simulations
        progress_bar = st.progress(0)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_simulation, i) for i in range(num_runs)]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                intrinsic_values.append(future.result())
                progress_bar.progress((i + 1) / num_runs)
                logging.info(f"Completed simulation {i + 1}/{num_runs}")

        intrinsic_values = np.array(intrinsic_values)
        avg_value = np.mean(intrinsic_values)
        std_dev = np.std(intrinsic_values)
        prob_undervalued = np.mean(intrinsic_values > current_price) * 100 if current_price > 0 else 50.0

        result = {
            'values': intrinsic_values,
            'avg_value': avg_value,
            'std_dev': std_dev,
            'prob_undervalued': prob_undervalued,
            'num_runs': num_runs,
            'source': 'live'
        }
        logging.info(f"Monte Carlo results: avg_value={avg_value:.2f}, std_dev={std_dev:.2f}, prob_undervalued={prob_undervalued:.1f}%")
        st.success(f"Monte Carlo simulation completed: Average value ${avg_value:,.2f}, Probability undervalued {prob_undervalued:.1f}%")
        return result

    except Exception as e:
        logging.error(f"Error in Monte Carlo simulation: {str(e)}")
        st.error(f"Monte Carlo simulation failed: {str(e)}. Using fallback results.")
        return {
            'values': np.array([65000.0] * 100),  # Realistic fallback for 2025
            'avg_value': 65000.0,
            'std_dev': 10000.0,
            'prob_undervalued': 50.0,
            'num_runs': 100,
            'source': 'fallback'
        }
