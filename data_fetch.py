import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

@st.cache_resource
def get_yfinance_ticker(symbol):
    return yf.Ticker(symbol)

@st.cache_data(ttl=600)
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_bitcoin_data(electricity_cost=0.05):
    """
    Fetch Bitcoin data from CoinGecko, CoinMarketCap, Kraken, or Binance.
    electricity_cost: Cost per kWh for mining cost estimation ($/kWh).
    Returns a dictionary with all required metrics.
    """
    data = {}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    # Initialize defaults (September 2025, realistic)
    data.update({
        'current_price': 65000.0,  # Kraken, expect $60,000-$65,000
        'market_cap': 1.28e12,  # Derived: 65000 * 19700000
        'total_volume': 5e10,  # 24h volume
        'circulating_supply': 19700000,  # Blockchain.com, expect 19.5M-20M
        'total_supply': 21000000,  # Fixed
        'social_volume': 10000,  # Default
        'sentiment_score': 0.5,  # Default, -1 to 1
        'hash_rate': 550.0,  # Blockchain.com, expect 500-600 EH/s
        'active_addresses': 900000,  # Blockchain.com, expect 800K-1M
        'transaction_volume': 2e9,  # Blockchain.com, expect $1B-$5B
        'mvrv': 2.0,  # Blockchain.com, expect 1.5-2.5
        'sopr': 1.0,  # Blockchain.com, expect 0.9-1.1
        'puell_multiple': 1.0,  # Blockchain.com, expect 0.5-1.5
        'realized_cap': 6.5e11,  # Blockchain.com, expect $600B-$800B
        'mining_cost': 10000.0,  # Calculated, expect $5,000-$15,000
        'electricity_cost': electricity_cost,
        'next_halving_date': datetime(2028, 4, 1),  # Estimate
        'fear_greed': 50,  # Alternative.me, expect 40-60
        'gold_price': 2200.0,  # Yahoo Finance, expect $2,000-$2,500
        'sp_correlation': 0.5,  # Calculated, expect 0.4-0.6
        'us_inflation': 2.8,  # Web-scraped, expect 2.5-3.5%
        'fed_rate': 4.75,  # Web-scraped, expect 4.5-5.0%
        '50_day_ma': 62000.0,  # Yahoo Finance, expect $55,000-$65,000
        '200_day_ma': 58000.0,  # Yahoo Finance, expect $50,000-$60,000
        'rsi': 50.0,  # Yahoo Finance, expect 40-60
        'beta': 1.5,  # Default, vs S&P 500
        'desired_return': 15.0,  # Default
        'margin_of_safety': 25.0,  # Default
        'monte_carlo_runs': 1000,  # Default
        'volatility_adj': 10.0,  # Reduced for realistic impact
        'growth_adj': 20.0,  # Default
        's2f_intercept': -1.84,  # Adjusted for ln(sf), per PlanB
        's2f_slope': 3.96,  # Adjusted for ln(sf), per PlanB
        'metcalfe_coeff': 1.0,  # Adjusted for ~$50,000-$70,000/BTC
        'block_reward': 3.125,  # Post-2024 halving
        'blocks_per_day': 144  # Blockchain.com, ~144
    })

    # Try CoinGecko
    try:
        cg_url = "https://api.coingecko.com/api/v3/coins/bitcoin"
        cg_params = {'x_cg_api_key': os.getenv('COINGECKO_API_KEY', '')}
        cg_response = requests.get(cg_url, headers=headers, params=cg_params, timeout=10).json()
        market_data = cg_response['market_data']
        
        data['current_price'] = market_data['current_price']['usd']
        data['market_cap'] = market_data['market_cap']['usd']
        data['total_volume'] = market_data['total_volume']['usd']
        data['circulating_supply'] = market_data['circulating_supply']
        data['total_supply'] = market_data['max_supply'] or 21000000
        community = cg_response['community_data']
        data['social_volume'] = community['reddit_average_posts_48h'] + community['twitter_followers'] / 1000
        up = cg_response['sentiment_votes_up_percentage']
        down = cg_response['sentiment_votes_down_percentage']
        data['sentiment_score'] = (up - down) / 100 if up and down else 0.5
        logging.info(f"Fetched from CoinGecko: price={data['current_price']}, market_cap={data['market_cap']}")
        st.success(f"Fetched market data from CoinGecko: ${data['current_price']:.2f}")
    
    except Exception as e:
        logging.warning(f"CoinGecko failed: {str(e)}. Trying CoinMarketCap...")
        
        # Try CoinMarketCap
        try:
            cmc_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            cmc_params = {'symbol': 'BTC', 'convert': 'USD'}
            cmc_headers = {'X-CMC_PRO_API_KEY': os.getenv('COINMARKETCAP_API_KEY', ''), **headers}
            cmc_response = requests.get(cmc_url, headers=cmc_headers, params=cmc_params, timeout=10).json()
            btc_data = cmc_response['data']['BTC']
            
            data['current_price'] = btc_data['quote']['USD']['price']
            data['market_cap'] = btc_data['quote']['USD']['market_cap']
            data['total_volume'] = btc_data['quote']['USD']['volume_24h']
            data['circulating_supply'] = btc_data['circulating_supply']
            data['total_supply'] = btc_data['max_supply'] or 21000000
            logging.info(f"Fetched from CoinMarketCap: price={data['current_price']}, market_cap={data['market_cap']}")
            st.success(f"Fetched market data from CoinMarketCap: ${data['current_price']:.2f}")
        
        except Exception as e:
            logging.warning(f"CoinMarketCap failed: {str(e)}. Trying Kraken...")
            
            # Try Kraken
            try:
                kraken_url = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
                kraken_response = requests.get(kraken_url, headers=headers, timeout=10).json()
                if kraken_response.get('error'):
                    raise Exception(f"Kraken error: {kraken_response['error']}")
                btc_data = kraken_response['result']['XXBTZUSD']
                price = float(btc_data['c'][0])
                if price <= 0:
                    raise Exception("Invalid Kraken price")
                
                data['current_price'] = price
                data['market_cap'] = price * data['circulating_supply']
                data['total_volume'] = float(btc_data['v'][1]) * price
                logging.info(f"Fetched from Kraken: price={price}, market_cap={data['market_cap']}")
                st.success(f"Fetched market data from Kraken: ${price:.2f}")
            
            except Exception as e:
                logging.warning(f"Kraken failed: {str(e)}. Trying Binance...")
                
                # Try Binance
                try:
                    binance_url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
                    binance_response = requests.get(binance_url, headers=headers, timeout=10).json()
                    if 'code' in binance_response:
                        raise Exception(f"Binance error: {binance_response['msg']}")
                    price = float(binance_response['lastPrice'])
                    if price <= 0:
                        raise Exception("Invalid Binance price")
                    
                    data['current_price'] = price
                    data['market_cap'] = price * data['circulating_supply']
                    data['total_volume'] = float(binance_response['volume']) * price
                    logging.info(f"Fetched from Binance: price={price}, market_cap={data['market_cap']}")
                    st.success(f"Fetched market data from Binance: ${price:.2f}")
                
                except Exception as e:
                    logging.error(f"Binance failed: {str(e)}. Using defaults.")
                    st.error("Unable to fetch market data. Using defaults.")

    # Blockchain.com for on-chain
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_latest(chart_name, timespan='1days'):
        try:
            url = f"https://api.blockchain.info/charts/{chart_name}?format=json&timespan={timespan}"
            response = requests.get(url, headers=headers).json()
            value = response['values'][-1]['y']
            if value <= 0:
                raise Exception(f"Invalid {chart_name} value")
            logging.info(f"Fetched {chart_name}: {value}")
            return value
        except Exception as e:
            logging.warning(f"Error fetching {chart_name}: {str(e)}")
            return data.get(chart_name, 0.0)  # Fallback to default

    data['hash_rate'] = get_latest('hash-rate') or data['hash_rate']
    data['active_addresses'] = get_latest('n-unique-addresses') or data['active_addresses']
    data['transaction_volume'] = get_latest('estimated-transaction-volume-usd') or data['transaction_volume']
    data['mvrv'] = get_latest('mvrv') or data['mvrv']
    data['sopr'] = get_latest('sopr') or data['sopr']
    data['puell_multiple'] = get_latest('puell_multiple') or data['puell_multiple']
    data['realized_cap'] = data['market_cap'] / data['mvrv'] if data['mvrv'] > 0 else data['realized_cap']
    
    # Mining cost estimation
    try:
        def estimate_mining_cost(hash_rate, electricity_cost):
            if hash_rate <= 0 or electricity_cost <= 0:
                raise ValueError("Invalid hash_rate or electricity_cost")
            # 0.1 TWh/EH/s, convert EH/s to TH/s, 1 TH/s ≈ 30 W
            power_consumption = hash_rate * 0.1 * 1e12 * 30 / 1e6  # MW
            mining_cost = power_consumption * electricity_cost * 24 * 365 / (data['block_reward'] * data['blocks_per_day'])
            logging.info(f"Calculated mining_cost: {mining_cost}")
            return min(max(mining_cost, 5000.0), 15000.0)  # Cap between $5,000-$15,000
        data['mining_cost'] = estimate_mining_cost(data['hash_rate'], electricity_cost)
    except Exception as e:
        logging.error(f"Error calculating mining_cost: {str(e)}")
        data['mining_cost'] = 10000.0
    
    # Next halving
    try:
        height = requests.get('https://blockchain.info/q/getblockcount', headers=headers).json()
        current_cycle = height // 210000
        next_halving_block = (current_cycle + 1) * 210000
        blocks_left = next_halving_block - height
        minutes_left = blocks_left * 10
        days_left = minutes_left / 1440
        data['next_halving_date'] = datetime.now() + timedelta(days=days_left)
        logging.info(f"Calculated next_halving_date: {data['next_halving_date']}")
    except Exception as e:
        logging.error(f"Error calculating halving: {str(e)}")
    
    # Fear & Greed
    try:
        fng_url = "https://api.alternative.me/fng/?limit=1"
        fng_response = requests.get(fng_url, headers=headers).json()
        data['fear_greed'] = int(fng_response['data'][0]['value'])
        logging.info(f"Fetched fear_greed: {data['fear_greed']}")
    except Exception as e:
        logging.error(f"Error fetching Fear & Greed: {str(e)}")
    
    # Macro: Gold price
    try:
        gold = get_yfinance_ticker('GC=F')
        data['gold_price'] = gold.info.get('currentPrice', gold.info.get('regularMarketPrice', 2200.0))
        logging.info(f"Fetched gold_price: {data['gold_price']}")
    except Exception as e:
        logging.error(f"Error fetching gold price: {str(e)}")
    
    # Macro: S&P 500 correlation
    try:
        end = datetime.now()
        start = end - timedelta(days=365)
        btc_hist = yf.download('BTC-USD', start=start, end=end)['Close']
        sp_hist = yf.download('^GSPC', start=start, end=end)['Close']
        correlation = btc_hist.corr(sp_hist)
        data['sp_correlation'] = correlation if not np.isnan(correlation) else 0.5
        logging.info(f"Calculated sp_correlation: {data['sp_correlation']}")
    except Exception as e:
        logging.error(f"Error calculating S&P correlation: {str(e)}")
    
    # Macro: US Inflation
    try:
        inf_url = "https://www.usinflationcalculator.com/inflation/current-inflation-rates/"
        response = requests.get(inf_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        latest_row = table.find_all('tr')[-1]
        cols = latest_row.find_all('td')
        data['us_inflation'] = float(cols[-1].text.strip().replace('%', '')) or 2.8
        logging.info(f"Fetched us_inflation: {data['us_inflation']}")
    except Exception as e:
        logging.error(f"Error fetching inflation rate: {str(e)}")
    
    # Macro: Fed Interest Rate
    try:
        fed_url = "https://www.federalreserve.gov/releases/h15/"
        response = requests.get(fed_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', id='h15table')
        fed_rate = float(table.find_all('tr')[-1].find_all('td')[-1].text.strip())
        data['fed_rate'] = fed_rate or 4.75
        logging.info(f"Fetched fed_rate: {data['fed_rate']}")
    except Exception as e:
        logging.error(f"Error fetching Fed rate: {str(e)}")
    
    # Technical
    try:
        hist = yf.download('BTC-USD', period='1y')['Close']
        data['50_day_ma'] = hist.rolling(50).mean().iloc[-1] if not hist.empty else data['current_price'] * 0.95
        data['200_day_ma'] = hist.rolling(200).mean().iloc[-1] if not hist.empty else data['current_price'] * 0.9
        delta = hist.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - 100 / (1 + rs).iloc[-1] if not rs.empty else 50.0
        logging.info(f"Fetched technicals: 50_day_ma={data['50_day_ma']}, 200_day_ma={data['200_day_ma']}, rsi={data['rsi']}")
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {str(e)}")
    
    logging.info(f"Final data: {data}")
    return data

@st.cache_data(ttl=86400)
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_history(period='5y'):
    """
    Fetch historical price and on-chain data for Bitcoin, including hash rate MAs for Hash Ribbons.
    """
    try:
        df = yf.download('BTC-USD', period=period)
        df['circulating_supply'] = 19700000  # Default, updated below if available
        df['block_reward'] = 3.125  # Post-2024 halving
        df['blocks_per_day'] = 144
        for metric in ['n-unique-addresses', 'estimated-transaction-volume-usd', 'mvrv', 'sopr', 'puell_multiple', 'hash-rate']:
            try:
                url = f"https://api.blockchain.info/charts/{metric}?format=json&timespan={period}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers).json()
                values = pd.DataFrame(response['values'])
                values['x'] = pd.to_datetime(values['x'], unit='s')
                values.set_index('x', inplace=True)
                df[metric] = values['y'].reindex(df.index, method='ffill')
            except Exception as e:
                logging.error(f"Error fetching historical {metric}: {str(e)}")
                df[metric] = data.get(metric, 0.0)
        
        # Add Hash Ribbons MAs
        if 'hash-rate' in df.columns and not df['hash-rate'].isna().all():
            df['hash_rate_30d'] = df['hash-rate'].rolling(30).mean()
            df['hash_rate_60d'] = df['hash-rate'].rolling(60).mean()
        else:
            df['hash_rate_30d'] = 550.0
            df['hash_rate_60d'] = 550.0
        
        # Update circulating_supply and block_reward for historical accuracy
        halving_dates = [
            (datetime(2012, 11, 28), 50.0),
            (datetime(2016, 7, 9), 25.0),
            (datetime(2020, 5, 11), 12.5),
            (datetime(2024, 4, 20), 6.25),
            (datetime(2024, 4, 21), 3.125)  # Post-2024 halving
        ]
        for date, reward in halving_dates:
            df.loc[df.index < date, 'block_reward'] = reward
            df.loc[df.index < date, 'circulating_supply'] = min(19700000, 21000000 - (21000000 - df['circulating_supply'].iloc[-1]) * (1 - (df.index[-1] - date).days / (datetime(2028, 4, 1) - date).days))
        
        logging.info("Fetched historical data successfully")
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data: {str(e)}")
        st.warning("Unable to fetch historical data.")
        return pd.DataFrame()
