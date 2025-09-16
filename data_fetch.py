import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from cachetools import TTLCache

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Initialize caches (5 min for price data, 30 min for on-chain)
price_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes
onchain_cache = TTLCache(maxsize=100, ttl=1800)  # 30 minutes

@st.cache_resource
def get_yfinance_ticker(symbol):
    return yf.Ticker(symbol)

async def fetch_url(session, url, headers, params=None, timeout=10):
    """Helper function for async HTTP requests."""
    try:
        async with session.get(url, headers=headers, params=params, timeout=timeout) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}")
            return await response.json()
    except Exception as e:
        logging.warning(f"Failed to fetch {url}: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_bitcoin_data(electricity_cost=0.05):
    """
    Fetch Bitcoin data from CoinGecko, CoinMarketCap, Kraken, or Binance concurrently.
    electricity_cost: Cost per kWh for mining cost estimation ($/kWh).
    Returns a dictionary with all required metrics.
    """
    data = {
        'current_price': 65000.0,  # Realistic for Sep 2025
        'market_cap': 1.28e12,  # 65000 * 19.7M
        'total_volume': 5e10,  # 24h volume
        'circulating_supply': 19700000,  # Post-2024 halving
        'total_supply': 21000000,  # Fixed
        'social_volume': 10000,  # Default
        'sentiment_score': 0.5,  # -1 to 1
        'hash_rate': 750.0,  # Blockchain.com, realistic for 2025
        'active_addresses': 1050000,  # Blockchain.com
        'transaction_volume': 15.2e9,  # $15.2B, realistic
        'mvrv': 2.0,  # Blockchain.com, 1.5-2.5
        'sopr': 1.0,  # Blockchain.com, 0.9-1.1
        'puell_multiple': 1.0,  # Blockchain.com, 0.5-1.5
        'realized_cap': 680e9,  # $680B, realistic
        'mining_cost': 10000.0,  # $5,000-$15,000
        'electricity_cost': electricity_cost,
        'next_halving_date': datetime(2028, 4, 1),  # Post-2024 estimate
        'fear_greed': 50,  # Alternative.me, 40-60
        'gold_price': 2500.0,  # Yahoo Finance, realistic
        'sp_correlation': 0.5,  # Calculated, 0.4-0.6
        'us_inflation': 2.8,  # Web-scraped, 2.5-3.5%
        'fed_rate': 4.75,  # Web-scraped, 4.5-5.0%
        '50_day_ma': 62000.0,  # Yahoo Finance
        '200_day_ma': 58000.0,  # Yahoo Finance
        'rsi': 50.0,  # Yahoo Finance, 40-60
        'beta': 1.5,  # vs S&P 500
        'desired_return': 15.0,  # Default
        'margin_of_safety': 25.0,  # Default
        'monte_carlo_runs': 1000,  # Default
        'volatility_adj': 10.0,  # Realistic impact
        'growth_adj': 20.0,  # Default
        's2f_intercept': -1.84,  # PlanB adjusted
        's2f_slope': 3.96,  # PlanB adjusted
        'metcalfe_coeff': 15.0,  # Calibrated for 2025
        'block_reward': 3.125,  # Post-2024 halving
        'blocks_per_day': 144,  # Blockchain.com
        'source': 'fallback'  # Indicate fallback data
    }

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    async with aiohttp.ClientSession() as session:
        # Concurrent API calls
        tasks = [
            fetch_url(session, "https://api.coingecko.com/api/v3/coins/bitcoin",
                     headers, {'x_cg_api_key': os.getenv('COINGECKO_API_KEY', '')}),
            fetch_url(session, "https://api.kraken.com/0/public/Ticker?pair=XBTUSD", headers),
            fetch_url(session, "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", headers),
            fetch_url(session, "https://api.alternative.me/fng/?limit=1", headers)
        ]
        cg_data, kraken_data, binance_data, fng_data = await asyncio.gather(*tasks, return_exceptions=True)

        # CoinGecko
        if cg_data and not isinstance(cg_data, Exception):
            market_data = cg_data['market_data']
            data.update({
                'current_price': market_data['current_price']['usd'],
                'market_cap': market_data['market_cap']['usd'],
                'total_volume': market_data['total_volume']['usd'],
                'circulating_supply': market_data['circulating_supply'],
                'total_supply': market_data['max_supply'] or 21000000,
                'social_volume': cg_data['community_data']['reddit_average_posts_48h'] + cg_data['community_data']['twitter_followers'] / 1000,
                'sentiment_score': (cg_data['sentiment_votes_up_percentage'] - cg_data['sentiment_votes_down_percentage']) / 100 if cg_data.get('sentiment_votes_up_percentage') else 0.5,
                'source': 'coingecko'
            })
            price_cache['bitcoin'] = data
            logging.info(f"Fetched from CoinGecko: price={data['current_price']}, market_cap={data['market_cap']}")
            st.success(f"Fetched market data from CoinGecko: ${data['current_price']:.2f}")

        # Kraken fallback
        elif kraken_data and not isinstance(kraken_data, Exception) and not kraken_data.get('error'):
            btc_data = kraken_data['result']['XXBTZUSD']
            price = float(btc_data['c'][0])
            if price > 0:
                data.update({
                    'current_price': price,
                    'market_cap': price * data['circulating_supply'],
                    'total_volume': float(btc_data['v'][1]) * price,
                    'source': 'kraken'
                })
                price_cache['bitcoin'] = data
                logging.info(f"Fetched from Kraken: price={price}, market_cap={data['market_cap']}")
                st.success(f"Fetched market data from Kraken: ${price:.2f}")

        # Binance fallback
        elif binance_data and not isinstance(binance_data, Exception) and 'code' not in binance_data:
            price = float(binance_data['lastPrice'])
            if price > 0:
                data.update({
                    'current_price': price,
                    'market_cap': price * data['circulating_supply'],
                    'total_volume': float(binance_data['volume']) * price,
                    'source': 'binance'
                })
                price_cache['bitcoin'] = data
                logging.info(f"Fetched from Binance: price={price}, market_cap={data['market_cap']}")
                st.success(f"Fetched market data from Binance: ${price:.2f}")

        else:
            logging.error("All market data APIs failed. Using fallback.")
            st.error("Unable to fetch market data. Using fallback.")

        # Fear & Greed
        if fng_data and not isinstance(fng_data, Exception):
            data['fear_greed'] = int(fng_data['data'][0]['value'])
            logging.info(f"Fetched fear_greed: {data['fear_greed']}")

    # Blockchain.com on-chain data
    async def fetch_onchain_metric(chart_name, timespan='1days'):
        cache_key = f"{chart_name}_{timespan}"
        if cache_key in onchain_cache:
            return onchain_cache[cache_key]
        try:
            url = f"https://api.blockchain.info/charts/{chart_name}?format=json&timespan={timespan}"
            async with aiohttp.ClientSession() as session:
                response = await fetch_url(session, url, headers)
                value = response['values'][-1]['y']
                if value <= 0:
                    raise Exception(f"Invalid {chart_name} value")
                onchain_cache[cache_key] = value
                logging.info(f"Fetched {chart_name}: {value}")
                return value
        except Exception as e:
            logging.warning(f"Error fetching {chart_name}: {str(e)}")
            return data.get(chart_name, 0.0)

    onchain_tasks = [
        fetch_onchain_metric('hash-rate'),
        fetch_onchain_metric('n-unique-addresses'),
        fetch_onchain_metric('estimated-transaction-volume-usd'),
        fetch_onchain_metric('mvrv'),
        fetch_onchain_metric('sopr'),
        fetch_onchain_metric('puell_multiple')
    ]
    hash_rate, active_addresses, transaction_volume, mvrv, sopr, puell = await asyncio.gather(*onchain_tasks)
    data.update({
        'hash_rate': hash_rate or data['hash_rate'],
        'active_addresses': active_addresses or data['active_addresses'],
        'transaction_volume': transaction_volume or data['transaction_volume'],
        'mvrv': mvrv or data['mvrv'],
        'sopr': sopr or data['sopr'],
        'puell_multiple': puell or data['puell_multiple'],
        'realized_cap': data['market_cap'] / mvrv if mvrv > 0 else data['realized_cap']
    })

    # Mining cost estimation
    try:
        def estimate_mining_cost(hash_rate, electricity_cost):
            if hash_rate <= 0 or electricity_cost <= 0:
                raise ValueError("Invalid hash_rate or electricity_cost")
            power_consumption = hash_rate * 0.1 * 1e12 * 30 / 1e6  # MW
            mining_cost = power_consumption * electricity_cost * 24 * 365 / (data['block_reward'] * data['blocks_per_day'])
            return min(max(round(mining_cost), 5000.0), 15000.0)
        data['mining_cost'] = estimate_mining_cost(data['hash_rate'], electricity_cost)
        logging.info(f"Calculated mining_cost: {data['mining_cost']}")
    except Exception as e:
        logging.error(f"Error calculating mining_cost: {str(e)}")

    # Next halving
    try:
        async with aiohttp.ClientSession() as session:
            response = await fetch_url(session, 'https://blockchain.info/q/getblockcount', headers)
            height = response
            current_cycle = height // 210000
            next_halving_block = (current_cycle + 1) * 210000
            blocks_left = next_halving_block - height
            days_left = blocks_left * 10 / 1440
            data['next_halving_date'] = datetime.now() + timedelta(days=days_left)
            logging.info(f"Calculated next_halving_date: {data['next_halving_date']}")
    except Exception as e:
        logging.error(f"Error calculating halving: {str(e)}")

    # Macro: Gold price
    try:
        gold = get_yfinance_ticker('GC=F')
        data['gold_price'] = gold.info.get('currentPrice', gold.info.get('regularMarketPrice', 2500.0))
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
        async with aiohttp.ClientSession() as session:
            response = await session.get("https://www.usinflationcalculator.com/inflation/current-inflation-rates/", headers=headers)
            soup = BeautifulSoup(await response.text(), 'html.parser')
            table = soup.find('table')
            latest_row = table.find_all('tr')[-1]
            cols = latest_row.find_all('td')
            data['us_inflation'] = float(cols[-1].text.strip().replace('%', '')) or 2.8
            logging.info(f"Fetched us_inflation: {data['us_inflation']}")
    except Exception as e:
        logging.error(f"Error fetching inflation rate: {str(e)}")

    # Macro: Fed Interest Rate
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.get("https://www.federalreserve.gov/releases/h15/", headers=headers)
            soup = BeautifulSoup(await response.text(), 'html.parser')
            table = soup.find('table', id='h15table')
            fed_rate = float(table.find_all('tr')[-1].find_all('td')[-1].text.strip())
            data['fed_rate'] = fed_rate or 4.75
            logging.info(f"Fetched fed_rate: {data['fed_rate']}")
    except Exception as e:
        logging.error(f"Error fetching Fed rate: {str(e)}")

    # Technical indicators
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

@st.cache_data(ttl=86400)  # Cache for 24 hours
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_history(period='5y'):
    """
    Fetch historical price and on-chain data for Bitcoin, including hash rate MAs for Hash Ribbons.
    Includes data sampling to reduce payload size.
    """
    try:
        # Fetch historical price data
        async with aiohttp.ClientSession() as session:
            days = {'1mo': 30, '3mo': 90, '1y': 365, '5y': 1825}.get(period, 1825)
            url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
            response = await fetch_url(session, url, {'User-Agent': 'Mozilla/5.0'}, {'x_cg_api_key': os.getenv('COINGECKO_API_KEY', '')})
            if not response:
                raise Exception("Failed to fetch historical price data")
            prices = pd.DataFrame(response['prices'], columns=['timestamp', 'price'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices.set_index('timestamp', inplace=True)

        df = yf.download('BTC-USD', period=period)
        df['price'] = prices['price'].reindex(df.index, method='ffill')
        df['circulating_supply'] = 19700000
        df['block_reward'] = 3.125
        df['blocks_per_day'] = 144

        # Fetch on-chain data concurrently
        async def fetch_historical_metric(chart_name):
            cache_key = f"historical_{chart_name}_{period}"
            if cache_key in onchain_cache:
                return onchain_cache[cache_key]
            try:
                url = f"https://api.blockchain.info/charts/{chart_name}?format=json&timespan={period}"
                async with aiohttp.ClientSession() as session:
                    response = await fetch_url(session, url, {'User-Agent': 'Mozilla/5.0'})
                    values = pd.DataFrame(response['values'])
                    values['x'] = pd.to_datetime(values['x'], unit='s')
                    values.set_index('x', inplace=True)
                    onchain_cache[cache_key] = values['y']
                    return values['y']
            except Exception as e:
                logging.warning(f"Error fetching historical {chart_name}: {str(e)}")
                return pd.Series(data.get(chart_name, 0.0), index=df.index)

        metrics = ['hash-rate', 'n-unique-addresses', 'estimated-transaction-volume-usd', 'mvrv', 'sopr', 'puell_multiple']
        results = await asyncio.gather(*[fetch_historical_metric(metric) for metric in metrics])
        for metric, values in zip(metrics, results):
            df[metric] = values.reindex(df.index, method='ffill')

        # Add Hash Ribbons MAs
        if 'hash-rate' in df.columns and not df['hash-rate'].isna().all():
            df['hash_rate_30d'] = df['hash-rate'].rolling(30).mean()
            df['hash_rate_60d'] = df['hash-rate'].rolling(60).mean()
        else:
            df['hash_rate_30d'] = 750.0
            df['hash_rate_60d'] = 750.0

        # Update circulating supply and block reward
        halving_dates = [
            (datetime(2012, 11, 28), 50.0),
            (datetime(2016, 7, 9), 25.0),
            (datetime(2020, 5, 11), 12.5),
            (datetime(2024, 4, 20), 6.25),
            (datetime(2024, 4, 21), 3.125)
        ]
        for date, reward in halving_dates:
            df.loc[df.index < date, 'block_reward'] = reward
            df.loc[df.index < date, 'circulating_supply'] = min(19700000, 21000000 - (21000000 - df['circulating_supply'].iloc[-1]) * (1 - (df.index[-1] - date).days / (datetime(2028, 4, 1) - date).days))

        # Sample data to reduce payload
        sample_rate = {'1mo': 2, '3mo': 5, '1y': 10, '5y': 20}.get(period, 20)
        df = df[::sample_rate]  # Sample every nth row
        logging.info(f"Fetched and sampled historical data: {len(df)} points")
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data: {str(e)}")
        st.error(f"Failed to fetch historical data: {str(e)}. Using fallback.")
        # Fallback data
        fallback_data = pd.DataFrame([
            {'date': '2024-10-01', 'price': 62500, 'hash-rate': 750.0, 'n-unique-addresses': 1050000, 'estimated-transaction-volume-usd': 15.2e9, 'mvrv': 2.0, 'sopr': 1.0, 'puell_multiple': 1.0, 'hash_rate_30d': 750.0, 'hash_rate_60d': 750.0, 'circulating_supply': 19700000, 'block_reward': 3.125},
            {'date': '2024-11-01', 'price': 69800, 'hash-rate': 760.0, 'n-unique-addresses': 1060000, 'estimated-transaction-volume-usd': 15.5e9, 'mvrv': 2.1, 'sopr': 1.05, 'puell_multiple': 1.1, 'hash_rate_30d': 755.0, 'hash_rate_60d': 752.0, 'circulating_supply': 19700000, 'block_reward': 3.125}
        ])
        fallback_data['date'] = pd.to_datetime(fallback_data['date'])
        fallback_data.set_index('date', inplace=True)
        fallback_data['source'] = 'fallback'
        return fallback_data
