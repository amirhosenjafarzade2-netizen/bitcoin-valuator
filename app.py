import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import asyncio
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor

try:
    from data_fetch import fetch_bitcoin_data, fetch_history
    from valuation_models import calculate_valuation
    from visualization import (
        plot_heatmap, plot_monte_carlo, plot_model_comparison,
        plot_onchain_metrics, plot_sentiment_analysis,
        plot_price_history, plot_macro_correlations
    )
    from utils import validate_inputs, export_portfolio, generate_pdf_report
    from monte_carlo import run_monte_carlo
except ImportError as e:
    st.error(f"Failed to import modules: {str(e)}. Ensure all required files are in the project directory.")
    logging.error(f"Import error: {str(e)}")
    st.stop()

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize cache
cache = TTLCache(maxsize=100, ttl=3600)  # Cache for 1 hour

# Set page config
st.set_page_config(page_title="Bitcoin Valuation Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load CSS with Bitcoin branding
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    logging.error("styles.css not found")
    st.warning("styles.css not found. Using default styling.")
    st.markdown("""
        <style>
        .stButton>button { background-color: #F7931A; color: white; border-radius: 5px; }
        .stMetric { background-color: #1A3C6D; color: white; padding: 10px; border-radius: 5px; }
        .stSidebar { background-color: #F5F5F5; }
        </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Asset', 'Intrinsic Value', 'Undervaluation %', 'Verdict', 'Beta'])
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Async data fetching with caching
async def fetch_data_with_cache():
    cache_key = 'bitcoin_data'
    if cache_key in cache:
        logging.info("Using cached Bitcoin data")
        return cache[cache_key]
    try:
        data = await fetch_bitcoin_data(electricity_cost=0.05)
        cache[cache_key] = data
        logging.info(f"Fetched and cached data: {data}")
        return data
    except Exception as e:
        logging.error(f"Data fetch error: {str(e)}")
        st.error(f"Failed to fetch data: {str(e)}. Using default values.")
        return {
            'current_price': 115000.0,  # 2025 estimate
            'total_supply': 21000000.0,
            'circulating_supply': 19780000.0,  # Post-2024 halving
            'next_halving_date': datetime(2028, 4, 1),
            'margin_of_safety': 20.0,
            'hash_rate': 750.0,  # Conservative 2025 estimate
            'active_addresses': 1050000.0,  # 2025 estimate
            'transaction_volume': 3e9,  # 2025 estimate
            'mvrv': 2.2,
            'sopr': 1.0,
            'realized_cap': 7.5e11,
            'puell_multiple': 1.2,
            'mining_cost': 15000.0,
            'fear_greed': 55,
            'social_volume': 15000.0,
            'sentiment_score': 0.6,
            'us_inflation': 2.8,
            'fed_rate': 4.75,
            'sp_correlation': 0.45,
            'gold_price': 2500.0,
            'rsi': 50.0,
            '50_day_ma': 110000.0,
            '200_day_ma': 100000.0,
            'desired_return': 15.0,
            'monte_carlo_runs': 1000,
            'volatility_adj': 15.0,
            'growth_adj': 25.0,
            'beta': 1.6,
            's2f_intercept': -1.84,
            's2f_slope': 3.96,
            'metcalfe_coeff': 15.0,  # Calibrated for 2025
            'block_reward': 3.125,  # Post-2024 halving
            'blocks_per_day': 144.0,
            'electricity_cost': 0.05
        }

# Run async fetch
data = asyncio.run(fetch_data_with_cache())

st.title("Bitcoin Valuation Dashboard")
st.markdown("Analyze Bitcoin using advanced models like S2F, Metcalfe, and NVT. *Not financial advice. Verify inputs.*")

# Dark mode toggle
st.session_state.dark_mode = st.sidebar.checkbox("Dark Mode", value=st.session_state.dark_mode, help="Toggle dark/light theme")

# Tabs
tab1, tab2 = st.tabs(["Valuation Dashboard", "On-Chain & Sentiment Analysis"])

with tab1:
    with st.sidebar:
        st.header("Input Parameters")
        
        debug_mode = st.checkbox("Debug Mode", value=False, help="Show detailed input/output info")
        
        with st.expander("Model Descriptions"):
            st.markdown("""
            - **Stock-to-Flow (S2F)**: Price based on scarcity (post-2024: 3.125 BTC reward).
            - **Metcalfe's Law**: Network value via active users squared (exponent 1.69).
            - **Network Value to Transactions (NVT)**: P/E-like ratio (target NVT=15).
            - **Pi Cycle Top Indicator**: Uses 111-day/350-day MAs for market signals.
            - **Reverse S2F**: Implied growth for target price (1.5x current).
            - **Market Sentiment Composite (MSC)**: Weighted fear/greed and social sentiment.
            - **Bitcoin Energy Value Model**: Floor price from mining costs (750 EH/s).
            - **RVMR**: Realized cap vs miner revenue (post-2024 production).
            - **Mayer Multiple**: Price vs 200-day MA (target multiple 1.5).
            - **Hash Ribbons**: Miner capitulation via hash rate MAs (30d vs 60d).
            - **Macro Monetary Model**: Adjusts Metcalfe by inflation (2.8%) and rates (4.75%).
            - **Ensemble**: Weighted average (50% S2F, 30% Metcalfe, 20% NVT).
            """)
        
        model = st.selectbox(
            "Valuation Model",
            ["Stock-to-Flow (S2F)", "Metcalfe's Law", "Network Value to Transactions (NVT)", 
             "Pi Cycle Top Indicator", "Reverse S2F", "Market Sentiment Composite (MSC)",
             "Bitcoin Energy Value Model", "RVMR", "Mayer Multiple", "Hash Ribbons", 
             "Macro Monetary Model", "Ensemble"],
            help="Select a model or ensemble for valuation."
        )
        
        use_fetched_data = st.checkbox("Use Fetched Data", value=True, help="Initialize with API data (edit to override)")
        
        with st.expander("Data Overview"):
            st.write("Fetched Metrics (2025 Estimates):")
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    st.write(f"- {key.replace('_', ' ').title()}: {float(value):,.2f}")
                elif isinstance(value, datetime):
                    st.write(f"- {key.replace('_', ' ').title()}: {value.strftime('%Y-%m-%d')}")
                else:
                    st.write(f"- {key.replace('_', ' ').title()}: {value}")
        
        with st.expander("Core Inputs"):
            current_price = st.number_input("Current Price (USD)", min_value=0.01, value=float(data.get('current_price', 115000.0)), help="BTC price (Kraken, ~$115,000)")
            total_supply = st.number_input("Total Supply (BTC)", min_value=0.0, value=float(data.get('total_supply', 21000000.0)), help="Max supply (21M)")
            circulating_supply = st.number_input("Circulating Supply (BTC)", min_value=0.0, value=float(data.get('circulating_supply', 19780000.0)), help="Circulating BTC (~19.78M)")
            next_halving_date = st.date_input("Next Halving Date", value=data.get('next_halving_date', datetime(2028, 4, 1)), help="Next halving (~April 2028)")
            margin_of_safety = st.number_input("Margin of Safety (%)", min_value=0.0, max_value=100.0, value=float(data.get('margin_of_safety', 20.0)), help="Risk discount (0-100%)")
        
        with st.expander("On-Chain Inputs"):
            hash_rate = st.number_input("Hash Rate (EH/s)", min_value=0.0, value=float(data.get('hash_rate', 750.0)), help="Network hash rate (~750 EH/s)")
            active_addresses = st.number_input("Active Addresses (Daily)", min_value=0.0, value=float(data.get('active_addresses', 1050000.0)), help="Daily active addresses (~1.05M)")
            transaction_volume = st.number_input("Transaction Volume (USD, Daily)", min_value=0.0, value=float(data.get('transaction_volume', 3e9)), help="Daily USD volume (~$3B)")
            mvrv = st.number_input("MVRV Ratio", min_value=0.0, value=float(data.get('mvrv', 2.2)), help="Market vs Realized Value (~2.2)")
            sopr = st.number_input("SOPR", min_value=0.0, value=float(data.get('sopr', 1.0)), help="Spent Output Profit Ratio (~1.0)")
            realized_cap = st.number_input("Realized Cap (USD)", min_value=0.0, value=float(data.get('realized_cap', 7.5e11)), help="Value at purchase price (~$750B)")
            puell_multiple = st.number_input("Puell Multiple", min_value=0.0, value=float(data.get('puell_multiple', 1.2)), help="Miners' revenue vs avg (~1.2)")
            electricity_cost = st.number_input("Electricity Cost ($/kWh)", min_value=0.0, max_value=1.0, value=float(data.get('electricity_cost', 0.05)), help="Mining cost per kWh")
            block_reward = st.number_input("Block Reward (BTC)", min_value=0.0, max_value=50.0, value=float(data.get('block_reward', 3.125)), help="Block reward (3.125 post-2024)")
            blocks_per_day = st.number_input("Blocks Per Day", min_value=100.0, max_value=200.0, value=float(data.get('blocks_per_day', 144.0)), help="Blocks mined daily (~144)")
        
        with st.expander("Model-Specific Inputs"):
            s2f_intercept = st.number_input("S2F Intercept", min_value=-10.0, max_value=10.0, value=float(data.get('s2f_intercept', -1.84)), help="S2F ln(sf) intercept (-1.84)")
            s2f_slope = st.number_input("S2F Slope", min_value=0.0, max_value=10.0, value=float(data.get('s2f_slope', 3.96)), help="S2F ln(sf) slope (3.96)")
            metcalfe_coeff = st.number_input("Metcalfe Coefficient", min_value=0.0, max_value=20.0, value=float(data.get('metcalfe_coeff', 15.0)), help="Metcalfe scaling (~15.0)")
        
        with st.expander("Sentiment Inputs"):
            fear_greed = st.number_input("Fear & Greed Index (0-100)", min_value=0, max_value=100, value=int(data.get('fear_greed', 55)), help="0=Fear, 100=Greed (~55)")
            social_volume = st.number_input("Social Volume (Mentions/Day)", min_value=0.0, value=float(data.get('social_volume', 15000.0)), help="Social media mentions (~15,000)")
            sentiment_score = st.number_input("Sentiment Score (-1 to 1)", min_value=-1.0, max_value=1.0, value=float(data.get('sentiment_score', 0.6)), help="Bullish=1, Bearish=-1 (~0.6)")
        
        with st.expander("Macro Inputs"):
            us_inflation = st.number_input("US Inflation Rate (%)", min_value=0.0, max_value=50.0, value=float(data.get('us_inflation', 2.8)), help="Annual inflation (~2.8%)")
            fed_rate = st.number_input("Fed Interest Rate (%)", min_value=0.0, max_value=50.0, value=float(data.get('fed_rate', 4.75)), help="Fed rate (~4.75%)")
            sp_correlation = st.number_input("S&P 500 Correlation (0-1)", min_value=0.0, max_value=1.0, value=float(data.get('sp_correlation', 0.45)), help="BTC-S&P correlation (~0.45)")
            gold_price = st.number_input("Gold Price (USD/oz)", min_value=0.0, value=float(data.get('gold_price', 2500.0)), help="Gold price (~$2,500)")
        
        with st.expander("Technical Inputs"):
            rsi = st.number_input("RSI (14-day)", min_value=0.0, max_value=100.0, value=float(data.get('rsi', 50.0)), help="Overbought >70, Oversold <30 (~50)")
            ma_50 = st.number_input("50-Day MA", min_value=0.0, value=float(data.get('50_day_ma', 110000.0)), help="50-day MA (~$110,000)")
            ma_200 = st.number_input("200-Day MA", min_value=0.0, value=float(data.get('200_day_ma', 100000.0)), help="200-day MA (~$100,000)")
        
        with st.expander("Monte Carlo Settings"):
            monte_carlo_runs = st.number_input("Number of Runs", min_value=100, max_value=2000, value=int(data.get('monte_carlo_runs', 1000)), help="100-2000 runs (~1000)")
            volatility_adj = st.number_input("Volatility Adjustment (±%)", min_value=0.0, max_value=50.0, value=float(data.get('volatility_adj', 15.0)), help="Volatility range (~15%)")
            growth_adj = st.number_input("Growth Adjustment (±%)", min_value=0.0, max_value=50.0, value=float(data.get('growth_adj', 25.0)), help="Growth range (~25%)")
        
        beta = st.number_input("Beta (vs. Market)", min_value=0.0, value=float(data.get('beta', 1.6)), help="BTC market risk (~1.6)")
        
        if st.button("Clear Cache"):
            cache.clear()
            st.cache_data.clear()
            st.rerun()
        
        calculate = st.button("Calculate Valuation", type="primary")
        add_to_portfolio = st.button("Add to Portfolio")
        export = st.button("Export Portfolio")
        download_report = st.button("Download PDF Report")
        
        if debug_mode:
            with st.expander("Debug Logs"):
                try:
                    with open('app.log', 'r') as f:
                        st.text(f.read())
                except FileNotFoundError:
                    st.write("No logs found.")
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.header("Valuation Results")
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
                    'mining_cost': float(data.get('mining_cost', 15000.0)),
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
                
                logging.info(f"Inputs: {inputs}")
                if debug_mode:
                    st.write("Inputs:", inputs)
                
                try:
                    is_valid, errors, sanitized_inputs = validate_inputs(inputs)
                    if not is_valid:
                        st.error(f"Invalid inputs: {errors}")
                        logging.error(f"Validation failed: {errors}")
                    else:
                        results = calculate_valuation(sanitized_inputs)
                        st.session_state.results = results
                        st.session_state.data = sanitized_inputs
                        logging.info(f"Valuation results: {results}")
                        if debug_mode:
                            st.write("Results:", results)
                        
                        # Display metrics with accuracy
                        st.subheader("Key Metrics")
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Score", f"{results.get('score', 0)}/100")
                            st.metric("Current Price", f"${results.get('current_price', 0):,.2f}")
                            st.metric("Intrinsic Value", f"${results.get('intrinsic_value', 0):,.2f}")
                            st.metric("Safe Buy Price", f"${results.get('safe_buy_price', 0):,.2f}")
                        with cols[1]:
                            st.metric("Undervaluation", f"{results.get('undervaluation', 0):.2f}%")
                            st.metric("NVT Ratio", f"{results.get('nvt_ratio', 0):.2f}")
                            st.metric("MVRV Z-Score", f"{results.get('mvrv_z_score', 0):.2f}")
                            st.metric("Verdict", results.get('verdict', '-'))
                        with cols[2]:
                            st.metric("SOPR Signal", results.get('sopr_signal', '-'))
                            st.metric("Puell Signal", results.get('puell_signal', '-'))
                            st.metric("Mining Cost vs Price", f"{results.get('mining_cost_vs_price', 0):.2f}%")
                            st.metric("Hash Ribbons Signal", results.get('hash_ribbon_signal', '-'))
                        
                        # Model comparison table
                        st.subheader("Model Comparison")
                        model_comp = pd.DataFrame({
                            'Model': ['S2F', 'Metcalfe', 'NVT', 'Pi Cycle', 'Reverse S2F', 'MSC', 
                                    'Energy', 'RVMR', 'Mayer Multiple', 'Hash Ribbons', 'Macro Monetary'],
                            'Intrinsic Value': [
                                results.get('s2f_value', 0),
                                results.get('metcalfe_value', 0),
                                results.get('nvt_value', 0),
                                results.get('pi_cycle_value', 0),
                                results.get('reverse_s2f_value', 0),
                                results.get('msc_value', 0),
                                results.get('energy_value', 0),
                                results.get('rvmr_value', 0),
                                results.get('mayer_multiple_value', 0),
                                results.get('hash_ribbons_value', 0),
                                results.get('macro_monetary_value', 0)
                            ],
                            'Accuracy (%)': [
                                results.get('s2f_value_accuracy', 0),
                                results.get('metcalfe_value_accuracy', 0),
                                results.get('nvt_value_accuracy', 0),
                                results.get('pi_cycle_value_accuracy', 0),
                                results.get('reverse_s2f_value_accuracy', 0),
                                results.get('msc_value_accuracy', 0),
                                results.get('energy_value_accuracy', 0),
                                results.get('rvmr_value_accuracy', 0),
                                results.get('mayer_multiple_value_accuracy', 0),
                                results.get('hash_ribbons_value_accuracy', 0),
                                results.get('macro_monetary_value_accuracy', 0)
                            ]
                        })
                        st.dataframe(model_comp, use_container_width=True)
                except Exception as e:
                    st.error(f"Valuation failed: {str(e)}")
                    logging.error(f"Valuation error: {str(e)}")
    
    with col_right:
        st.header("Portfolio Overview")
        portfolio_beta = st.session_state.portfolio['Beta'].mean() if not st.session_state.portfolio.empty else 0
        expected_return = portfolio_beta * 8.0
        st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
        st.metric("Expected Return", f"{expected_return:.2f}%")
        
        if add_to_portfolio and 'results' in st.session_state and st.session_state.results:
            new_row = pd.DataFrame([{
                'Asset': 'BTC',
                'Intrinsic Value': st.session_state.results.get('intrinsic_value', 0),
                'Undervaluation %': st.session_state.results.get('undervaluation', 0),
                'Verdict': st.session_state.results.get('verdict', '-'),
                'Beta': beta
            }])
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
            st.success("Added to portfolio")
        
        st.dataframe(st.session_state.portfolio, use_container_width=True)
        
        if export:
            try:
                export_portfolio(st.session_state.portfolio)
                st.success("Portfolio exported to portfolio_export.csv")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
                logging.error(f"Export error: {str(e)}")
        
        if download_report and 'results' in st.session_state and st.session_state.results:
            try:
                model_comp_fig = plot_model_comparison(pd.DataFrame({
                    'Model': ['S2F', 'Metcalfe', 'NVT', 'Pi Cycle', 'Reverse S2F', 'MSC', 
                            'Energy', 'RVMR', 'Mayer Multiple', 'Hash Ribbons', 'Macro Monetary'],
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
                }))
                pdf = generate_pdf_report(st.session_state.results, st.session_state.portfolio, model_comp_fig)
                if pdf:
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf,
                        file_name="bitcoin_valuation_report.pdf",
                        mime="application/pdf"
                    )
                    st.success("PDF report generated")
            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")
                logging.error(f"Report error: {str(e)}")
        
        st.header("Scenario Analysis")
        with st.expander("Adjust Scenarios"):
            bull_adj = st.slider("Bull Case Adjustment (%)", -50.0, 50.0, 25.0, help="Bullish scenario adjustment")
            bear_adj = st.slider("Bear Case Adjustment (%)", -50.0, 50.0, -25.0, help="Bearish scenario adjustment")
        if 'results' in st.session_state and st.session_state.results:
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
        
        st.header("Sensitivity Analysis")
        if 'results' in st.session_state and st.session_state.results:
            try:
                heatmap = plot_heatmap(st.session_state.results.get('intrinsic_value', 0), volatility_adj, growth_adj)
                if heatmap:
                    st.plotly_chart(heatmap, use_container_width=True)
            except Exception as e:
                st.error(f"Heatmap generation failed: {str(e)}")
                logging.error(f"Heatmap error: {str(e)}")
        
        st.header("Monte Carlo Simulation")
        if 'results' in st.session_state and st.session_state.results:
            with st.spinner("Running Monte Carlo simulation..."):
                try:
                    with ThreadPoolExecutor() as executor:
                        mc_results = executor.submit(run_monte_carlo, st.session_state.data, monte_carlo_runs, volatility_adj, growth_adj).result()
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Average Intrinsic Value", f"${mc_results.get('avg_value', 0):,.2f}")
                    with cols[1]:
                        st.metric("Std Dev", f"${mc_results.get('std_dev', 0):,.2f}")
                    with cols[2]:
                        st.metric("Prob Undervalued", f"{mc_results.get('prob_undervalued', 0):.2f}%")
                    mc_plot = plot_monte_carlo(mc_results)
                    if mc_plot:
                        st.plotly_chart(mc_plot, use_container_width=True)
                except Exception as e:
                    st.error(f"Monte Carlo simulation failed: {str(e)}")
                    logging.error(f"Monte Carlo error: {str(e)}")

with tab2:
    st.header("On-Chain & Sentiment Analysis")
    try:
        cache_key = 'history_data'
        if cache_key in cache:
            history = cache[cache_key]
            logging.info("Using cached historical data")
        else:
            history = fetch_history(period='5y')
            cache[cache_key] = history
            logging.info("Fetched and cached historical data")
        
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
        else:
            st.warning("No historical data available")
    except Exception as e:
        st.error(f"On-chain & sentiment analysis failed: {str(e)}")
        logging.error(f"On-chain analysis error: {str(e)}")

st.markdown("---")
st.markdown("*Disclaimer: This tool is for informational purposes only and not financial advice.*")
