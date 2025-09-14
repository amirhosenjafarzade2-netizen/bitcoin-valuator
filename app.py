import streamlit as st
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from data_fetch import fetch_bitcoin_data, fetch_history
from valuation_models import calculate_valuation
from visualizations import plot_heatmap, plot_monte_carlo, plot_model_comparison, plot_onchain_metrics, plot_sentiment_analysis, plot_price_history, plot_macro_correlations
from utils import validate_inputs, export_portfolio, generate_pdf_report
from monte_carlo import run_monte_carlo
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Bitcoin Valuation Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load CSS
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    logging.error("styles.css not found")
    st.warning("styles.css not found. Using default styling.")

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Asset', 'Intrinsic Value', 'Undervaluation %', 'Verdict', 'Beta'])
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

# Fetch data
data = fetch_bitcoin_data(electricity_cost=0.05)
logging.info(f"Fetched data: {data}")

st.title("Bitcoin Valuation Dashboard")
st.markdown("Analyze Bitcoin using models like Stock-to-Flow, Metcalfe's Law, and NVT. *Not financial advice. Verify all inputs.*")

tab1, tab2 = st.tabs(["Valuation Dashboard", "On-Chain & Sentiment Analysis"])

with tab1:
    with st.sidebar:
        st.header("Input Parameters")
        
        debug_mode = st.checkbox("Debug Mode", value=False, help="Show detailed input and validation info")
        
        with st.expander("Model Descriptions"):
            st.markdown("""
            - **Stock-to-Flow (S2F)**: Predicts price based on scarcity (supply issuance).
            - **Metcalfe's Law**: Values network by active users squared.
            - **Network Value to Transactions (NVT)**: Like P/E ratio, detects over/undervaluation.
            - **Pi Cycle Top Indicator**: Uses moving averages to signal market tops/bottoms.
            - **Reverse S2F**: Implies growth needed for target price.
            - **Market Sentiment Composite (MSC)**: Weighted sentiment index adjusting price.
            - **Bitcoin Energy Value Model**: Floor price based on mining energy costs.
            - **RVMR**: Realized cap vs miner revenue for mining economics.
            - **Mayer Multiple**: Price relative to 200-day MA for fair value.
            - **Hash Ribbons**: Hash rate MAs for miner capitulation signals.
            - **Macro Monetary Model**: Adjusts Metcalfe's value by inflation and real rates.
            """)
        
        model = st.selectbox(
            "Valuation Model",
            ["Stock-to-Flow (S2F)", "Metcalfe's Law", "Network Value to Transactions (NVT)", 
             "Pi Cycle Top Indicator", "Reverse S2F", "Market Sentiment Composite (MSC)",
             "Bitcoin Energy Value Model", "RVMR", "Mayer Multiple", "Hash Ribbons", "Macro Monetary Model"],
            help="Select a model to analyze Bitcoin."
        )
        
        use_fetched_data = st.checkbox("Initialize with Fetched Data", value=True, help="Initialize inputs with API-fetched data (edit to override incorrect values)")
        
        with st.expander("Fetched Data Overview"):
            st.write("All Metrics (Fetched or Default):")
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    st.write(f"- {key.replace('_', ' ').title()}: {float(value):.2f}")
                elif isinstance(value, datetime):
                    st.write(f"- {key.replace('_', ' ').title()}: {value.strftime('%Y-%m-%d')}")
                else:
                    st.write(f"- {key.replace('_', ' ').title()}: {value}")
        
        with st.expander("Core Inputs"):
            current_price_fetched = float(data.get('current_price', 60000.0))
            st.write(f"Fetched Current Price: ${current_price_fetched:.2f} (Kraken, expect $60,000-$65,000)")
            if current_price_fetched < 10000 or current_price_fetched > 100000:
                st.warning("Current price seems unusual. Verify and edit if needed.")
            current_price = st.number_input("Current Price (USD)", min_value=0.01, value=current_price_fetched if use_fetched_data else 60000.0, help="Current BTC price in USD (Kraken)")
            
            total_supply_fetched = float(data.get('total_supply', 21000000.0))
            st.write(f"Fetched Total Supply: {total_supply_fetched:.0f} BTC (Fixed at 21M)")
            total_supply = st.number_input("Total Supply (BTC)", min_value=0.0, value=total_supply_fetched if use_fetched_data else 21000000.0, help="Maximum BTC supply (21M)")
            
            circulating_supply_fetched = float(data.get('circulating_supply', 19700000.0))
            st.write(f"Fetched Circulating Supply: {circulating_supply_fetched:.0f} BTC (Blockchain.com, expect 19.5M-20M)")
            if circulating_supply_fetched < 19000000 or circulating_supply_fetched > 21000000:
                st.warning("Circulating supply seems unusual. Verify and edit if needed.")
            circulating_supply = st.number_input("Circulating Supply (BTC)", min_value=0.0, value=circulating_supply_fetched if use_fetched_data else 19700000.0, help="Current circulating BTC (Blockchain.com)")
            
            next_halving_date_fetched = data.get('next_halving_date', datetime(2028, 4, 1))
            st.write(f"Fetched Next Halving Date: {next_halving_date_fetched.strftime('%Y-%m-%d')} (Estimate)")
            next_halving_date = st.date_input("Next Halving Date", value=next_halving_date_fetched if use_fetched_data else datetime(2028, 4, 1), help="Estimated date of next halving (~April 2028)")
            
            margin_of_safety_fetched = float(data.get('margin_of_safety', 25.0))
            st.write(f"Fetched Margin of Safety: {margin_of_safety_fetched:.2f}% (Default)")
            margin_of_safety = st.number_input("Margin of Safety (%)", min_value=0.0, max_value=100.0, value=margin_of_safety_fetched if use_fetched_data else 25.0, help="Discount for risk (0-100%)")
        
        with st.expander("On-Chain Inputs"):
            hash_rate_fetched = float(data.get('hash_rate', 500.0))
            st.write(f"Fetched Hash Rate: {hash_rate_fetched:.2f} EH/s (Blockchain.com, expect 500-600)")
            if hash_rate_fetched < 400 or hash_rate_fetched > 700:
                st.warning("Hash rate seems unusual. Verify and edit if needed.")
            hash_rate = st.number_input("Hash Rate (EH/s)", min_value=0.0, value=hash_rate_fetched if use_fetched_data else 500.0, help="Network hash rate (Blockchain.com)")
            
            active_addresses_fetched = float(data.get('active_addresses', 1000000.0))
            st.write(f"Fetched Active Addresses: {active_addresses_fetched:.0f} (Blockchain.com, expect 800K-1M)")
            if active_addresses_fetched < 500000 or active_addresses_fetched > 1500000:
                st.warning("Active addresses seem unusual. Verify and edit if needed.")
            active_addresses = st.number_input("Active Addresses (Daily)", min_value=0.0, value=active_addresses_fetched if use_fetched_data else 1000000.0, help="Daily active wallet addresses (Blockchain.com)")
            
            transaction_volume_fetched = float(data.get('transaction_volume', 1e9))
            st.write(f"Fetched Transaction Volume: ${transaction_volume_fetched:.2e} (Blockchain.com, expect $1B-$5B)")
            if transaction_volume_fetched < 5e8 or transaction_volume_fetched > 1e10:
                st.warning("Transaction volume seems unusual. Verify and edit if needed.")
            transaction_volume = st.number_input("Transaction Volume (USD, Daily)", min_value=0.0, value=transaction_volume_fetched if use_fetched_data else 1e9, help="Daily USD transaction volume (Blockchain.com)")
            
            mvrv_fetched = float(data.get('mvrv', 2.0))
            st.write(f"Fetched MVRV: {mvrv_fetched:.2f} (Blockchain.com, expect 1.5-2.5)")
            if mvrv_fetched < 0.5 or mvrv_fetched > 5.0:
                st.warning("MVRV seems unusual. Verify and edit if needed.")
            mvrv = st.number_input("MVRV Ratio", min_value=0.0, value=mvrv_fetched if use_fetched_data else 2.0, help="Market Value to Realized Value (Blockchain.com)")
            
            sopr_fetched = float(data.get('sopr', 1.0))
            st.write(f"Fetched SOPR: {sopr_fetched:.2f} (Blockchain.com, expect 0.9-1.1)")
            if sopr_fetched < 0.5 or sopr_fetched > 1.5:
                st.warning("SOPR seems unusual. Verify and edit if needed.")
            sopr = st.number_input("SOPR", min_value=0.0, value=sopr_fetched if use_fetched_data else 1.0, help="Spent Output Profit Ratio (Blockchain.com, ~1)")
            
            realized_cap_fetched = float(data.get('realized_cap', 6e11))
            st.write(f"Fetched Realized Cap: ${realized_cap_fetched:.2e} (Blockchain.com, expect $600B-$800B)")
            if realized_cap_fetched < 4e11 or realized_cap_fetched > 1e12:
                st.warning("Realized cap seems unusual. Verify and edit if needed.")
            realized_cap = st.number_input("Realized Cap (USD)", min_value=0.0, value=realized_cap_fetched if use_fetched_data else 6e11, help="Total value of all BTC at purchase price (Blockchain.com)")
            
            puell_multiple_fetched = float(data.get('puell_multiple', 1.0))
            st.write(f"Fetched Puell Multiple: {puell_multiple_fetched:.2f} (Blockchain.com, expect 0.5-1.5)")
            if puell_multiple_fetched < 0.3 or puell_multiple_fetched > 5.0:
                st.warning("Puell multiple seems unusual. Verify and edit if needed.")
            puell_multiple = st.number_input("Puell Multiple", min_value=0.0, value=puell_multiple_fetched if use_fetched_data else 1.0, help="Miners' revenue vs historical avg (Blockchain.com, 0.3-5)")
            
            electricity_cost_fetched = float(data.get('electricity_cost', 0.05))
            st.write(f"Fetched Electricity Cost: ${electricity_cost_fetched:.2f}/kWh (Default)")
            electricity_cost = st.number_input("Electricity Cost ($/kWh)", min_value=0.0, max_value=1.0, value=electricity_cost_fetched if use_fetched_data else 0.05, help="Cost per kWh for mining cost estimation")
            
            block_reward_fetched = float(data.get('block_reward', 3.125))
            st.write(f"Fetched Block Reward: {block_reward_fetched:.2f} BTC (Blockchain.com, expect 3.125 post-2024 halving)")
            block_reward = st.number_input("Block Reward (BTC)", min_value=0.0, max_value=50.0, value=block_reward_fetched if use_fetched_data else 3.125, help="Current block reward per block (Blockchain.com)")
            
            blocks_per_day_fetched = float(data.get('blocks_per_day', 144.0))
            st.write(f"Fetched Blocks Per Day: {blocks_per_day_fetched:.0f} (Blockchain.com, expect ~144)")
            blocks_per_day = st.number_input("Blocks Per Day", min_value=100.0, max_value=200.0, value=blocks_per_day_fetched if use_fetched_data else 144.0, help="Approx blocks mined per day (Blockchain.com)")
        
        with st.expander("Model-Specific Inputs"):
            s2f_intercept_fetched = float(data.get('s2f_intercept', 14.6))
            st.write(f"Fetched S2F Intercept: {s2f_intercept_fetched:.2f} (Default)")
            s2f_intercept = st.number_input("S2F Intercept", min_value=0.0, max_value=100.0, value=s2f_intercept_fetched if use_fetched_data else 14.6, help="S2F model intercept (default 14.6)")
            
            s2f_slope_fetched = float(data.get('s2f_slope', 0.05))
            st.write(f"Fetched S2F Slope: {s2f_slope_fetched:.2f} (Default)")
            s2f_slope = st.number_input("S2F Slope", min_value=0.0, max_value=1.0, value=s2f_slope_fetched if use_fetched_data else 0.05, help="S2F model slope (default 0.05)")
            
            metcalfe_coeff_fetched = float(data.get('metcalfe_coeff', 0.0001))
            st.write(f"Fetched Metcalfe Coefficient: {metcalfe_coeff_fetched:.4f} (Default, adjust to 0.001 for higher valuation)")
            if metcalfe_coeff_fetched < 0.00001 or metcalfe_coeff_fetched > 0.01:
                st.warning("Metcalfe coefficient seems unusual. Verify and edit if needed.")
            metcalfe_coeff = st.number_input("Metcalfe Coefficient", min_value=0.0, max_value=0.01, value=metcalfe_coeff_fetched if use_fetched_data else 0.0001, help="Scaling factor for Metcalfe's Law (default 0.0001)")
        
        with st.expander("Sentiment Inputs"):
            fear_greed_fetched = int(data.get('fear_greed', 50))
            st.write(f"Fetched Fear & Greed: {fear_greed_fetched} (Alternative.me, expect 40-60)")
            if fear_greed_fetched < 20 or fear_greed_fetched > 80:
                st.warning("Fear & Greed index seems unusual. Verify and edit if needed.")
            fear_greed = st.number_input("Fear & Greed Index (0-100)", min_value=0, max_value=100, value=fear_greed_fetched if use_fetched_data else 50, help="0=Extreme Fear, 100=Extreme Greed (Alternative.me)")
            
            social_volume_fetched = float(data.get('social_volume', 10000.0))
            st.write(f"Fetched Social Volume: {social_volume_fetched:.0f} (Default)")
            social_volume = st.number_input("Social Volume (Mentions/Day)", min_value=0.0, value=social_volume_fetched if use_fetched_data else 10000.0, help="Social media mentions (default 10,000)")
            
            sentiment_score_fetched = float(data.get('sentiment_score', 0.5))
            st.write(f"Fetched Sentiment Score: {sentiment_score_fetched:.2f} (Default)")
            if sentiment_score_fetched < -1 or sentiment_score_fetched > 1:
                st.warning("Sentiment score seems unusual. Verify and edit if needed.")
            sentiment_score = st.number_input("Sentiment Score (-1 to 1)", min_value=-1.0, max_value=1.0, value=sentiment_score_fetched if use_fetched_data else 0.5, help="Positive=Bullish, Negative=Bearish (default 0.5)")
        
        with st.expander("Macro Inputs"):
            us_inflation_fetched = float(data.get('us_inflation', 3.0))
            st.write(f"Fetched US Inflation: {us_inflation_fetched:.2f}% (Web-scraped, expect 2.5-3.5%)")
            if us_inflation_fetched < 1 or us_inflation_fetched > 10:
                st.warning("US inflation seems unusual. Verify and edit if needed.")
            us_inflation = st.number_input("US Inflation Rate (%)", min_value=0.0, max_value=50.0, value=us_inflation_fetched if use_fetched_data else 3.0, help="Annual US inflation rate (web-scraped)")
            
            fed_rate_fetched = float(data.get('fed_rate', 5.0))
            st.write(f"Fetched Fed Rate: {fed_rate_fetched:.2f}% (Web-scraped, expect 4.5-5.0%)")
            if fed_rate_fetched < 2 or fed_rate_fetched > 10:
                st.warning("Fed rate seems unusual. Verify and edit if needed.")
            fed_rate = st.number_input("Fed Interest Rate (%)", min_value=0.0, max_value=50.0, value=fed_rate_fetched if use_fetched_data else 5.0, help="Federal Reserve interest rate (web-scraped)")
            
            sp_correlation_fetched = float(data.get('sp_correlation', 0.5))
            st.write(f"Fetched S&P 500 Correlation: {sp_correlation_fetched:.2f} (Calculated, expect 0.4-0.6)")
            if sp_correlation_fetched < 0 or sp_correlation_fetched > 1:
                st.warning("S&P 500 correlation seems unusual. Verify and edit if needed.")
            sp_correlation = st.number_input("S&P 500 Correlation (0-1)", min_value=0.0, max_value=1.0, value=sp_correlation_fetched if use_fetched_data else 0.5, help="BTC-S&P 500 correlation (calculated)")
            
            gold_price_fetched = float(data.get('gold_price', 2000.0))
            st.write(f"Fetched Gold Price: ${gold_price_fetched:.2f} (Yahoo Finance, expect $2,000-$2,500)")
            if gold_price_fetched < 1500 or gold_price_fetched > 3000:
                st.warning("Gold price seems unusual. Verify and edit if needed.")
            gold_price = st.number_input("Gold Price (USD/oz)", min_value=0.0, value=gold_price_fetched if use_fetched_data else 2000.0, help="Gold price for comparison (Yahoo Finance)")
        
        with st.expander("Technical Inputs"):
            rsi_fetched = float(data.get('rsi', 50.0))
            st.write(f"Fetched RSI: {rsi_fetched:.2f} (Yahoo Finance, expect 40-60)")
            if rsi_fetched < 10 or rsi_fetched > 90:
                st.warning("RSI seems unusual. Verify and edit if needed.")
            rsi = st.number_input("RSI (14-day)", min_value=0.0, max_value=100.0, value=rsi_fetched if use_fetched_data else 50.0, help="Overbought >70, Oversold <30 (Yahoo Finance)")
            
            ma_50_fetched = float(data.get('50_day_ma', 57000.0))
            st.write(f"Fetched 50-Day MA: ${ma_50_fetched:.2f} (Yahoo Finance, expect $55,000-$65,000)")
            if ma_50_fetched < 40000 or ma_50_fetched > 80000:
                st.warning("50-day MA seems unusual. Verify and edit if needed.")
            ma_50 = st.number_input("50-Day MA", min_value=0.0, value=ma_50_fetched if use_fetched_data else 57000.0, help="50-day moving average (Yahoo Finance)")
            
            ma_200_fetched = float(data.get('200_day_ma', 54000.0))
            st.write(f"Fetched 200-Day MA: ${ma_200_fetched:.2f} (Yahoo Finance, expect $50,000-$60,000)")
            if ma_200_fetched < 40000 or ma_200_fetched > 80000:
                st.warning("200-day MA seems unusual. Verify and edit if needed.")
            ma_200 = st.number_input("200-Day MA", min_value=0.0, value=ma_200_fetched if use_fetched_data else 54000.0, help="200-day moving average (Yahoo Finance)")
        
        with st.expander("Monte Carlo Settings"):
            monte_carlo_runs_fetched = int(data.get('monte_carlo_runs', 1000))
            st.write(f"Fetched Monte Carlo Runs: {monte_carlo_runs_fetched} (Default)")
            monte_carlo_runs = st.number_input("Number of Runs", min_value=100, max_value=2000, value=monte_carlo_runs_fetched if use_fetched_data else 1000, help="100-2000 runs (default 1000)")
            
            volatility_adj_fetched = float(data.get('volatility_adj', 30.0))
            st.write(f"Fetched Volatility Adjustment: {volatility_adj_fetched:.2f}% (Default)")
            volatility_adj = st.number_input("Volatility Adjustment Range (±%)", min_value=0.0, max_value=50.0, value=volatility_adj_fetched if use_fetched_data else 30.0, help="Volatility variation (default 30%)")
            
            growth_adj_fetched = float(data.get('growth_adj', 20.0))
            st.write(f"Fetched Growth Adjustment: {growth_adj_fetched:.2f}% (Default)")
            growth_adj = st.number_input("Growth Adjustment Range (±%)", min_value=0.0, max_value=50.0, value=growth_adj_fetched if use_fetched_data else 20.0, help="Growth rate variation (default 20%)")
        
        beta_fetched = float(data.get('beta', 1.5))
        st.write(f"Fetched Beta: {beta_fetched:.2f} (Default)")
        beta = st.number_input("Beta (vs. Market)", min_value=0.0, value=beta_fetched if use_fetched_data else 1.5, help="BTC's market risk vs S&P 500 (default 1.5)")
        
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.rerun()
        
        calculate = st.button("Calculate")
        add_to_portfolio = st.button("Add to Portfolio")
        export = st.button("Export Portfolio")
        download_report = st.button("Download Report")
        
        if debug_mode:
            with st.expander("Debug Logs"):
                try:
                    with open('app.log', 'r') as f:
                        st.text(f.read())
                except FileNotFoundError:
                    st.write("No logs found.")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.header("Valuation Dashboard")
        if calculate:
            with st.spinner("Calculating valuation..."):
                inputs = {
                    'model': model,
                    'current_price': float(current_price),
                    'total_supply': float(total_supply),
                    'circulating_supply': float(circulating_supply),
                    'next_halving_date': next_halving_date,
                    'margin_of_safety': float(margin_of_safety),
                    'hash_rate': float(hash_rate),
                    'active_addresses': float(active_addresses),
                    'transaction_volume': float(transaction_volume),
                    'mvrv': float(mvrv),
                    'sopr': float(sopr),
                    'realized_cap': float(realized_cap),
                    'puell_multiple': float(puell_multiple),
                    'mining_cost': float(data.get('mining_cost', 10000.0)),
                    'fear_greed': int(fear_greed),
                    'social_volume': float(social_volume),
                    'sentiment_score': float(sentiment_score),
                    'us_inflation': float(us_inflation),
                    'fed_rate': float(fed_rate),
                    'sp_correlation': float(sp_correlation),
                    'gold_price': float(gold_price),
                    'rsi': float(rsi),
                    '50_day_ma': float(ma_50),
                    '200_day_ma': float(ma_200),
                    'desired_return': float(data.get('desired_return', 15.0)),
                    'monte_carlo_runs': int(monte_carlo_runs),
                    'volatility_adj': float(volatility_adj),
                    'growth_adj': float(growth_adj),
                    'beta': float(beta),
                    'market_cap': float(current_price * circulating_supply),
                    's2f_intercept': float(s2f_intercept),
                    's2f_slope': float(s2f_slope),
                    'metcalfe_coeff': float(metcalfe_coeff),
                    'block_reward': float(block_reward),
                    'blocks_per_day': float(blocks_per_day),
                    'electricity_cost': float(electricity_cost)
                }
                
                logging.info(f"Inputs for valuation: {inputs}")
                logging.info(f"Input types: { {k: type(v).__name__ for k, v in inputs.items()} }")
                if debug_mode:
                    st.write("Inputs for Valuation:", inputs)
                    st.write("Input Types:", {k: type(v).__name__ for k, v in inputs.items()})
                
                try:
                    validation_result, validation_errors = validate_inputs(inputs)
                    if not validation_result:
                        st.error(f"Invalid inputs: {validation_errors}. Check values and logs.")
                        logging.error(f"Validation failed: {validation_errors}, Inputs: {inputs}")
                    else:
                        try:
                            st.session_state.results = {}
                            results = calculate_valuation(inputs)
                            st.session_state.results = results
                            st.session_state.data = inputs
                            logging.info(f"Valuation results: {results}")
                            if debug_mode:
                                st.write("Valuation Results:", results)
                            
                            st.metric("Score", f"{results.get('score', 0)}/100")
                            st.metric("Model", results.get('model', '-'))
                            st.metric("Current Price", f"${results.get('current_price', 0):.2f}")
                            st.metric("Intrinsic Value (Today)", f"${results.get('intrinsic_value', 0):.2f}")
                            st.metric("Safe Buy Price (after MOS)", f"${results.get('safe_buy_price', 0):.2f}")
                            st.metric("Undervaluation %", f"{results.get('undervaluation', 0):.2f}%")
                            st.metric("NVT Ratio", f"{results.get('nvt_ratio', 0):.2f}")
                            st.metric("MVRV Z-Score", f"{results.get('mvrv_z_score', 0):.2f}")
                            st.metric("SOPR Signal", results.get('sopr_signal', '-'))
                            st.metric("Puell Multiple Signal", results.get('puell_signal', '-'))
                            st.metric("Mining Cost vs Price", f"{results.get('mining_cost_vs_price', 0):.2f}%")
                            st.metric("Overall Score", f"{results.get('score', 0)}/100")
                            st.metric("Verdict", results.get('verdict', '-'))
                            st.metric("S2F Projected Value", f"${results.get('s2f_value', 0):.2f}")
                            st.metric("Metcalfe Value", f"${results.get('metcalfe_value', 0):.2f}")
                            st.metric("NVT Value", f"${results.get('nvt_value', 0):.2f}")
                            st.metric("Pi Cycle Value", f"${results.get('pi_cycle_value', 0):.2f}")
                            st.metric("Reverse S2F Value", f"${results.get('reverse_s2f_value', 0):.2f}")
                            st.metric("MSC Value", f"${results.get('msc_value', 0):.2f}")
                            st.metric("Energy Value", f"${results.get('energy_value', 0):.2f}")
                            st.metric("RVMR Value", f"${results.get('rvmr_value', 0):.2f}")
                            st.metric("Mayer Multiple", f"{results.get('mayer_multiple', 0):.2f}")
                            st.metric("Hash Ribbons Signal", results.get('hash_ribbon_signal', '-'))
                            st.metric("Macro Monetary Value", f"${results.get('macro_monetary_value', 0):.2f}")
                        except Exception as e:
                            st.error(f"Valuation calculation failed: {str(e)}. Check logs for details.")
                            logging.error(f"Valuation error: {str(e)}, Inputs: {inputs}")
                except Exception as e:
                    st.error(f"Validation failed: {str(e)}. Check logs for details.")
                    logging.error(f"Validation error: {str(e)}, Inputs: {inputs}")
    
    with col_right:
        st.header("Portfolio Overview")
        portfolio_beta = st.session_state.portfolio['Beta'].mean() if not st.session_state.portfolio.empty else 0
        expected_return = portfolio_beta * 8.0
        st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
        st.metric("Portfolio Expected Return", f"{expected_return:.2f}%")
        
        if add_to_portfolio and 'results' in st.session_state and st.session_state.results:
            new_row = pd.DataFrame([{
                'Asset': 'BTC',
                'Intrinsic Value': st.session_state.results.get('intrinsic_value', 0),
                'Undervaluation %': st.session_state.results.get('undervaluation', 0),
                'Verdict': st.session_state.results.get('verdict', '-'),
                'Beta': beta
            }])
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
        
        st.dataframe(st.session_state.portfolio, use_container_width=True)
        
        if export:
            export_portfolio(st.session_state.portfolio)
        
        if download_report and 'results' in st.session_state and st.session_state.results:
            model_comp = pd.DataFrame({
                'Model': ['S2F', 'Metcalfe', 'NVT', 'Pi Cycle', 'Reverse S2F', 'MSC', 'Energy', 'RVMR', 'Mayer', 'Hash Ribbons', 'Macro Monetary'],
                'Intrinsic Value': [
                    st.session_state.results.get('s2f_value', 0),
                    st.session_state.results.get('metcalfe_value', 0),
                    st.session_state.results.get('nvt_value', 0),
                    st.session_state.results.get('pi_cycle_value', 0),
                    st.session_state.results.get('reverse_s2f_value', 0),
                    st.session_state.results.get('msc_value', 0),
                    st.session_state.results.get('energy_value', 0),
                    st.session_state.results.get('rvmr_value', 0),
                    st.session_state.results.get('mayer_multiple_value', 0),
                    st.session_state.results.get('hash_ribbons_value', 0),
                    st.session_state.results.get('macro_monetary_value', 0)
                ]
            })
            model_comp_fig = plot_model_comparison(model_comp)
            pdf = generate_pdf_report(st.session_state.results, st.session_state.portfolio, model_comp_fig)
            if pdf:
                st.download_button(
                    label="Download PDF Report",
                    data=pdf,
                    file_name="bitcoin_valuation_report.pdf",
                    mime="application/pdf"
                )
        
        st.header("Scenario Analysis")
        with st.expander("Adjust Scenarios"):
            bull_adj = st.slider("Bull Case Adjustment (%)", -50.0, 50.0, 20.0, help="Adjust intrinsic value for bullish scenario")
            bear_adj = st.slider("Bear Case Adjustment (%)", -50.0, 50.0, -20.0, help="Adjust intrinsic value for bearish scenario")
        scenarios = pd.DataFrame({
            'Scenario': ['Base Case', 'Bull Case', 'Bear Case'],
            'Intrinsic Value': [
                st.session_state.results.get('intrinsic_value', 0),
                st.session_state.results.get('intrinsic_value', 0) * (1 + bull_adj/100),
                st.session_state.results.get('intrinsic_value', 0) * (1 + bear_adj/100)
            ],
            'Undervaluation %': [
                st.session_state.results.get('undervaluation', 0),
                st.session_state.results.get('undervaluation', 0) + bull_adj,
                st.session_state.results.get('undervaluation', 0) + bear_adj
            ]
        })
        st.dataframe(scenarios, use_container_width=True)
        
        st.header("Sensitivity Analysis (Heatmap)")
        if 'results' in st.session_state and st.session_state.results:
            heatmap = plot_heatmap(st.session_state.results.get('intrinsic_value', 0), volatility_adj, growth_adj)
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)
        
        st.header("Monte Carlo Simulation")
        if 'results' in st.session_state and st.session_state.results:
            with st.spinner("Running Monte Carlo simulation..."):
                with ThreadPoolExecutor() as executor:
                    mc_results = executor.submit(run_monte_carlo, st.session_state.data, monte_carlo_runs, volatility_adj, growth_adj).result()
            st.metric("Average Intrinsic Value", f"${mc_results.get('avg_value', 0):.2f}")
            st.metric("Std Dev", f"${mc_results.get('std_dev', 0):.2f}")
            st.metric("Probability Undervalued (> Current Price)", f"{mc_results.get('prob_undervalued', 0):.2f}%")
            mc_plot = plot_monte_carlo(mc_results)
            if mc_plot:
                st.plotly_chart(mc_plot, use_container_width=True)
        
        st.header("Model Comparison")
        if 'results' in st.session_state and st.session_state.results:
            model_comp = pd.DataFrame({
                'Model': ['S2F', 'Metcalfe', 'NVT', 'Pi Cycle', 'Reverse S2F', 'MSC', 'Energy', 'RVMR', 'Mayer', 'Hash Ribbons', 'Macro Monetary'],
                'Intrinsic Value': [
                    st.session_state.results.get('s2f_value', 0),
                    st.session_state.results.get('metcalfe_value', 0),
                    st.session_state.results.get('nvt_value', 0),
                    st.session_state.results.get('pi_cycle_value', 0),
                    st.session_state.results.get('reverse_s2f_value', 0),
                    st.session_state.results.get('msc_value', 0),
                    st.session_state.results.get('energy_value', 0),
                    st.session_state.results.get('rvmr_value', 0),
                    st.session_state.results.get('mayer_multiple_value', 0),
                    st.session_state.results.get('hash_ribbons_value', 0),
                    st.session_state.results.get('macro_monetary_value', 0)
                ]
            })
            model_comp_fig = plot_model_comparison(model_comp)
            if model_comp_fig:
                st.plotly_chart(model_comp_fig, use_container_width=True)

with tab2:
    st.header("On-Chain & Sentiment Analysis")
    history = fetch_history()
    if not history.empty:
        price_plot = plot_price_history(history)
        if price_plot:
            st.plotly_chart(price_plot, use_container_width=True)
        
        onchain_plot = plot_onchain_metrics(st.session_state.data, history)
        if onchain_plot:
            st.plotly_chart(onchain_plot, use_container_width=True)
        
        sentiment_plot = plot_sentiment_analysis(st.session_state.data, history)
        if sentiment_plot:
            st.plotly_chart(sentiment_plot, use_container_width=True)
        
        macro_plot = plot_macro_correlations(st.session_state.data, history)
        if macro_plot:
            st.plotly_chart(macro_plot, use_container_width=True)
    
st.markdown("---")
st.markdown("*Disclaimer: This tool is for informational purposes only and not financial advice.*")
