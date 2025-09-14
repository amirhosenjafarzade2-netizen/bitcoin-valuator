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
    Fetch Bitcoin data from CoinGecko, CoinMarketCap, or Kraken (in order).
    electricity_cost: Cost per kWh for mining cost estimation ($/kWh).
    Returns a dictionary with relevant metrics.
    """
    data = {}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

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
        data['sentiment_score'] = (up - down) / 100 if up and down else 0.0
        logging.info("Successfully fetched data from CoinGecko")
        st.success("Fetched market data from CoinGecko")
    
    except Exception as e:
        logging.error(f"CoinGecko failed: {str(e)}")
        st.warning("CoinGecko failed. Trying CoinMarketCap...")
        
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
            data['social_volume'] = 10000  # No direct equivalent
            data['sentiment_score'] = 0.5  # No direct equivalent
            logging.info("Successfully fetched data from CoinMarketCap")
            st.success("Fetched market data from CoinMarketCap")
        
        except Exception as e:
            logging.error(f"CoinMarketCap failed: {str(e)}")
            st.warning("CoinMarketCap failed. Trying Kraken...")
            
            # Try Kraken
            try:
                kraken_url = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
                kraken_response = requests.get(kraken_url, headers=headers, timeout=10).json()
                btc_data = kraken_response['result']['XXBTZUSD']
                
                data['current_price'] = float(btc_data['c'][0])
                data['market_cap'] = float(btc_data['c'][0]) * 19700000  # Approximate circulating supply
                data['total_volume'] = float(btc_data['v'][1]) * float(btc_data['c'][0])  # 24h volume
                data['circulating_supply'] = 19700000
                data['total_supply'] = 21000000
                data['social_volume'] = 10000
                data['sentiment_score'] = 0.5
                logging.info("Successfully fetched data from Kraken")
                st.success("Fetched market data from Kraken")
            
            except Exception as e:
                logging.error(f"Kraken failed: {str(e)}")
                st.error("Unable to fetch market data from all sources. Using defaults. Check API keys or network.")
                data['current_price'] = 60000.0
                data['market_cap'] = 1.2e12
                data['total_volume'] = 5e10
                data['circulating_supply'] = 19700000
                data['total_supply'] = 21000000
                data['social_volume'] = 10000
                data['sentiment_score'] = 0.5
    
    # Blockchain.com for on-chain
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_latest(chart_name, timespan='1days'):
        try:
            url = f"https://api.blockchain.info/charts/{chart_name}?format=json&timespan={timespan}"
            response = requests.get(url, headers=headers).json()
            return response['values'][-1]['y']
        except Exception as e:
            logging.error(f"Error fetching {chart_name}: {str(e)}")
            return 0.0
    
    data['hash_rate'] = get_latest('hash-rate') or 500.0
    data['active_addresses'] = get_latest('n-unique-addresses') or 1000000
    data['transaction_volume'] = get_latest('estimated-transaction-volume-usd') or 1e9
    data['mvrv'] = get_latest('mvrv') or 2.0
    data['sopr'] = get_latest('sopr') or 1.0
    data['puell_multiple'] = get_latest('puell_multiple') or 1.0
    data['realized_cap'] = data['market_cap'] / data['mvrv'] if data['mvrv'] > 0 else 6e11
    
    # Mining cost estimation
    def estimate_mining_cost(hash_rate, electricity_cost):
        power_consumption = hash_rate * 1000
        return power_consumption * electricity_cost * 24 * 365 / (6.25 * 144)
    data['mining_cost'] = estimate_mining_cost(data['hash_rate'], electricity_cost)
    data['electricity_cost'] = electricity_cost
    
    # Next halving
    try:
        height = requests.get('https://blockchain.info/q/getblockcount', headers=headers).json()
        current_cycle = height // 210000
        next_halving_block = (current_cycle + 1) * 210000
        blocks_left = next_halving_block - height
        minutes_left = blocks_left * 10
        days_left = minutes_left / 1440
        data['next_halving_date'] = datetime.now() + timedelta(days=days_left)
    except Exception as e:
        logging.error(f"Error calculating halving: {str(e)}")
        data['next_halving_date'] = datetime(2028, 4, 1)
    
    # Fear & Greed
    try:
        fng_url = "https://api.alternative.me/fng/?limit=1"
        fng_response = requests.get(fng_url, headers=headers).json()
        data['fear_greed'] = int(fng_response['data'][0]['value'])
    except Exception as e:
        logging.error(f"Error fetching Fear & Greed: {str(e)}")
        data['fear_greed'] = 50
    
    # Macro: Gold price
    try:
        gold = get_yfinance_ticker('GC=F')
        data['gold_price'] = gold.info.get('currentPrice', gold.info.get('regularMarketPrice', 2000.0))
    except Exception as e:
        logging.error(f"Error fetching gold price: {str(e)}")
        data['gold_price'] = 2000.0
    
    # Macro: S&P 500 correlation
    try:
        end = datetime.now()
        start = end - timedelta(days=365)
        btc_hist = yf.download('BTC-USD', start=start, end=end)['Close']
        sp_hist = yf.download('^GSPC', start=start, end=end)['Close']
        correlation = btc_hist.corr(sp_hist)
        data['sp_correlation'] = correlation if not np.isnan(correlation) else 0.5
    except Exception as e:
        logging.error(f"Error calculating S&P correlation: {str(e)}")
        data['sp_correlation'] = 0.5
    
    # Macro: US Inflation
    try:
        inf_url = "https://www.usinflationcalculator.com/inflation/current-inflation-rates/"
        response = requests.get(inf_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        latest_row = table.find_all('tr')[-1]
        cols = latest_row.find_all('td')
        data['us_inflation'] = float(cols[-1].text.strip().replace('%', '')) or 3.0
    except Exception as e:
        logging.error(f"Error fetching inflation rate: {str(e)}")
        data['us_inflation'] = 3.0
    
    # Macro: Fed Interest Rate
    try:
        fed_url = "https://www.federalreserve.gov/releases/h15/"
        response = requests.get(fed_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', id='h15table')
        fed_rate = float(table.find_all('tr')[-1].find_all('td')[-1].text.strip())
        data['fed_rate'] = fed_rate or 5.0
    except Exception as e:
        logging.error(f"Error fetching Fed rate: {str(e)}")
        data['fed_rate'] = 5.0
    
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
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {str(e)}")
        data['50_day_ma'] = data['current_price'] * 0.95
        data['200_day_ma'] = data['current_price'] * 0.9
        data['rsi'] = 50.0
    
    # Defaults
    data['beta'] = 1.5
    data['desired_return'] = 15.0
    data['margin_of_safety'] = 25.0
    data['monte_carlo_runs'] = 1000
    data['volatility_adj'] = 30.0
    data['growth_adj'] = 20.0
    data['s2f_intercept'] = 14.6
    data['s2f_slope'] = 0.05
    data['metcalfe_coeff'] = 0.0001
    data['block_reward'] = 6.25
    data['blocks_per_day'] = 144
    
    return data

@st.cache_data(ttl=86400)
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_history(period='5y'):
    """
    Fetch historical price and on-chain data for Bitcoin, including hash rate MAs for Hash Ribbons.
    """
    try:
        df = yf.download('BTC-USD', period=period)
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
                df[metric] = 0.0
        
        # Add Hash Ribbons MAs
        if 'hash-rate' in df.columns and not df['hash-rate'].isna().all():
            df['hash_rate_30d'] = df['hash-rate'].rolling(30).mean()
            df['hash_rate_60d'] = df['hash-rate'].rolling(60).mean()
        else:
            df['hash_rate_30d'] = 0.0
            df['hash_rate_60d'] = 0.0
        
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data: {str(e)}")
        st.warning("Unable to fetch historical data.")
        return pd.DataFrame()
