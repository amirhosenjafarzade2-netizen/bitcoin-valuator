import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import streamlit as st

@st.cache_data(ttl=3600)
def fetch_bitcoin_data():
    """
    Fetch Bitcoin data from various sources.
    Returns a dictionary with relevant metrics.
    """
    data = {}
    
    # CoinGecko for market data
    try:
        cg_url = "https://api.coingecko.com/api/v3/coins/bitcoin"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        cg_response = requests.get(cg_url, headers=headers).json()
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
        
    except Exception as e:
        st.error(f"Error fetching from CoinGecko: {str(e)}")
        data['current_price'] = 60000.0
        data['market_cap'] = 1.2e12
        data['total_volume'] = 5e10
        data['circulating_supply'] = 19700000
        data['total_supply'] = 21000000
        data['social_volume'] = 10000
        data['sentiment_score'] = 0.5
    
    # Blockchain.com for on-chain
    def get_latest(chart_name):
        try:
            url = f"https://api.blockchain.info/charts/{chart_name}?format=json&timespan=1days"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers).json()
            return response['values'][-1]['y']
        except:
            return 0.0
    
    data['hash_rate'] = get_latest('hash-rate') or 500.0
    data['active_addresses'] = get_latest('n-unique-addresses') or 1000000
    data['transaction_volume'] = get_latest('estimated-transaction-volume-usd') or 1e9
    data['mvrv'] = get_latest('mvrv') or 2.0
    data['sopr'] = get_latest('sopr') or 1.0
    data['puell_multiple'] = get_latest('puell_multiple') or 1.0
    data['realized_cap'] = data['market_cap'] / data['mvrv'] if data['mvrv'] > 0 else 6e11
    
    # Mining cost estimation
    data['mining_cost'] = 20000.0
    
    # Next halving
    try:
        height = requests.get('https://blockchain.info/q/getblockcount', headers=headers).json()
        current_cycle = height // 210000
        next_halving_block = (current_cycle + 1) * 210000
        blocks_left = next_halving_block - height
        minutes_left = blocks_left * 10
        days_left = minutes_left / 1440
        data['next_halving_date'] = datetime.now() + timedelta(days=days_left)
    except:
        data['next_halving_date'] = datetime(2028, 4, 1)
    
    # Fear & Greed
    try:
        fng_url = "https://api.alternative.me/fng/?limit=1"
        fng_response = requests.get(fng_url, headers=headers).json()
        data['fear_greed'] = int(fng_response['data'][0]['value'])
    except:
        data['fear_greed'] = 50
    
    # Macro: Gold price
    try:
        gold = yf.Ticker('GC=F')
        data['gold_price'] = gold.info.get('currentPrice', gold.info.get('regularMarketPrice', 2000.0))
    except:
        data['gold_price'] = 2000.0
    
    # Macro: S&P 500 correlation
    try:
        end = datetime.now()
        start = end - timedelta(days=365)
        btc_hist = yf.download('BTC-USD', start=start, end=end)['Close']
        sp_hist = yf.download('^GSPC', start=start, end=end)['Close']
        correlation = btc_hist.corr(sp_hist)
        data['sp_correlation'] = correlation if not np.isnan(correlation) else 0.5
    except:
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
    except:
        data['us_inflation'] = 3.0
    
    # Macro: Fed Interest Rate
    try:
        fed_url = "https://www.federalreserve.gov/releases/h15/"
        response = requests.get(fed_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', id='h15table')
        fed_rate = float(table.find_all('tr')[-1].find_all('td')[-1].text.strip())
        data['fed_rate'] = fed_rate or 5.0
    except:
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
    except:
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
    
    return data

@st.cache_data(ttl=3600)
def fetch_history(period='5y'):
    """
    Fetch historical price data for Bitcoin.
    """
    try:
        return yf.download('BTC-USD', period=period)
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()
