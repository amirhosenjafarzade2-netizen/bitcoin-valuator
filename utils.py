from datetime import datetime

def validate_inputs(inputs):
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
    if 'circulating_supply' in inputs and isinstance(inputs['circulating_supply'], (int, float)) and inputs['circulating_supply'] <= 0:
        errors.append("Circulating supply must be positive")
    if 'total_supply' in inputs and isinstance(inputs['total_supply'], (int, float)) and inputs['total_supply'] <= 0:
        errors.append("Total supply must be positive")
    if 'hash_rate' in inputs and isinstance(inputs['hash_rate'], (int, float)) and inputs['hash_rate'] < 0:
        errors.append("Hash rate cannot be negative")
    if 'active_addresses' in inputs and isinstance(inputs['active_addresses'], (int, float)) and inputs['active_addresses'] < 0:
        errors.append("Active addresses cannot be negative")
    if 'transaction_volume' in inputs and isinstance(inputs['transaction_volume'], (int, float)) and inputs['transaction_volume'] < 0:
        errors.append("Transaction volume cannot be negative")
    if 'mvrv' in inputs and isinstance(inputs['mvrv'], (int, float)) and inputs['mvrv'] < 0:
        errors.append("MVRV cannot be negative")
    if 'sopr' in inputs and isinstance(inputs['sopr'], (int, float)) and inputs['sopr'] < 0:
        errors.append("SOPR cannot be negative")
    if 'realized_cap' in inputs and isinstance(inputs['realized_cap'], (int, float)) and inputs['realized_cap'] < 0:
        errors.append("Realized cap cannot be negative")
    if 'puell_multiple' in inputs and isinstance(inputs['puell_multiple'], (int, float)) and inputs['puell_multiple'] < 0:
        errors.append("Puell multiple cannot be negative")
    if 'mining_cost' in inputs and isinstance(inputs['mining_cost'], (int, float)) and inputs['mining_cost'] < 0:
        errors.append("Mining cost cannot be negative")
    if 'fear_greed' in inputs and isinstance(inputs['fear_greed'], int) and (inputs['fear_greed'] < 0 or inputs['fear_greed'] > 100):
        errors.append("Fear & Greed must be between 0 and 100")
    if 'social_volume' in inputs and isinstance(inputs['social_volume'], (int, float)) and inputs['social_volume'] < 0:
        errors.append("Social volume cannot be negative")
    if 'sentiment_score' in inputs and isinstance(inputs['sentiment_score'], (int, float)) and (inputs['sentiment_score'] < -1 or inputs['sentiment_score'] > 1):
        errors.append("Sentiment score must be between -1 and 1")
    if 'us_inflation' in inputs and isinstance(inputs['us_inflation'], (int, float)) and inputs['us_inflation'] < 0:
        errors.append("US inflation cannot be negative")
    if 'fed_rate' in inputs and isinstance(inputs['fed_rate'], (int, float)) and inputs['fed_rate'] < 0:
        errors.append("Fed rate cannot be negative")
    if 'sp_correlation' in inputs and isinstance(inputs['sp_correlation'], (int, float)) and (inputs['sp_correlation'] < 0 or inputs['sp_correlation'] > 1):
        errors.append("S&P 500 correlation must be between 0 and 1")
    if 'gold_price' in inputs and isinstance(inputs['gold_price'], (int, float)) and inputs['gold_price'] < 0:
        errors.append("Gold price cannot be negative")
    if 'rsi' in inputs and isinstance(inputs['rsi'], (int, float)) and (inputs['rsi'] < 0 or inputs['rsi'] > 100):
        errors.append("RSI must be between 0 and 100")
    if '50_day_ma' in inputs and isinstance(inputs['50_day_ma'], (int, float)) and inputs['50_day_ma'] < 0:
        errors.append("50-day MA cannot be negative")
    if '200_day_ma' in inputs and isinstance(inputs['200_day_ma'], (int, float)) and inputs['200_day_ma'] < 0:
        errors.append("200-day MA cannot be negative")
    if 'desired_return' in inputs and isinstance(inputs['desired_return'], (int, float)) and inputs['desired_return'] < 0:
        errors.append("Desired return cannot be negative")
    if 'monte_carlo_runs' in inputs and isinstance(inputs['monte_carlo_runs'], int) and inputs['monte_carlo_runs'] < 100:
        errors.append("Monte Carlo runs must be at least 100")
    if 'volatility_adj' in inputs and isinstance(inputs['volatility_adj'], (int, float)) and inputs['volatility_adj'] < 0:
        errors.append("Volatility adjustment cannot be negative")
    if 'growth_adj' in inputs and isinstance(inputs['growth_adj'], (int, float)) and inputs['growth_adj'] < 0:
        errors.append("Growth adjustment cannot be negative")
    if 'beta' in inputs and isinstance(inputs['beta'], (int, float)) and inputs['beta'] < 0:
        errors.append("Beta cannot be negative")
    if 'market_cap' in inputs and isinstance(inputs['market_cap'], (int, float)) and inputs['market_cap'] < 0:
        errors.append("Market cap cannot be negative")
    if 's2f_intercept' in inputs and isinstance(inputs['s2f_intercept'], (int, float)) and inputs['s2f_intercept'] < 0:
        errors.append("S2F intercept cannot be negative")
    if 's2f_slope' in inputs and isinstance(inputs['s2f_slope'], (int, float)) and inputs['s2f_slope'] < 0:
        errors.append("S2F slope cannot be negative")
    if 'metcalfe_coeff' in inputs and isinstance(inputs['metcalfe_coeff'], (int, float)) and inputs['metcalfe_coeff'] < 0:
        errors.append("Metcalfe coefficient cannot be negative")
    if 'block_reward' in inputs and isinstance(inputs['block_reward'], (int, float)) and inputs['block_reward'] < 0:
        errors.append("Block reward cannot be negative")
    if 'blocks_per_day' in inputs and isinstance(inputs['blocks_per_day'], (int, float)) and inputs['blocks_per_day'] < 0:
        errors.append("Blocks per day cannot be negative")
    if 'electricity_cost' in inputs and isinstance(inputs['electricity_cost'], (int, float)) and inputs['electricity_cost'] < 0:
        errors.append("Electricity cost cannot be negative")
    
    return len(errors) == 0, errors
