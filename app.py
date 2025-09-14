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
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Dark mode toggle
dark_mode = st.checkbox("Toggle Dark Mode", value=False)
st.markdown(f'<body class="{"dark-mode" if dark_mode else ""}">', unsafe_allow_html=True)

st.title("Bitcoin Valuation Dashboard")
st.markdown("Analyze Bitcoin using models like Stock-to-Flow, Metcalfe's Law, and NVT. *Not financial advice. Verify all inputs.*")

tab1, tab2 = st.tabs(["Valuation Dashboard", "On-Chain & Sentiment Analysis"])

with tab1:
    with st.sidebar:
        st.header("Input Parameters")
        
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
        
        with st.expander("Core Inputs"):
            desired_return = st.number_input("Desired Return (%)", min_value=0.0, max_value=50.0, value=15.0, help="Expected annual return (0-50%)")
            current_price = st.number_input("Current Price (USD)", min_value=0.01, value=60000.0, help="Current BTC price in USD")
            total_supply = st.number_input("Total Supply (BTC)", min_value=0.0, value=21000000.0, help="Maximum BTC supply (default: 21M)")
            circulating_supply = st.number_input("Circulating Supply (BTC)", min_value=0.0, value=19700000.0, help="Current circulating BTC")
            next_halving_date = st.date_input("Next Halving Date", value=datetime(2028, 4, 1), help="Estimated date of next halving")
            margin_of_safety = st.number_input("Margin of Safety (%)", min_value=0.0, max_value=100.0, value=25.0, help="Discount for risk (0-100%)")
        
        with st.expander("On-Chain Inputs"):
            hash_rate = st.number_input("Hash Rate (EH/s)", min_value=0.0, value=500.0, help="Network hash rate")
            active_addresses = st.number_input("Active Addresses (Daily)", min_value=0.0, value=1000000.0, help="Daily active wallet addresses")
            transaction_volume = st.number_input("Transaction Volume (USD, Daily)", min_value=0.0, value=1e9, help="Daily USD transaction volume")
            mvrv = st.number_input("MVRV Ratio", min_value=0.0, value=2.0, help="Market Value to Realized Value")
            sopr = st.number_input("SOPR", min_value=0.0, value=1.0, help="Spent Output Profit Ratio (~1)")
            realized_cap = st.number_input("Realized Cap (USD)", min_value=0.0, value=6e11, help="Total value of all BTC at purchase price")
            puell_multiple = st.number_input("Puell Multiple", min_value=0.0, value=1.0, help="Miners' revenue vs historical avg (0.3-5)")
            electricity_cost = st.number_input("Electricity Cost ($/kWh)", min_value=0.0, max_value=1.0, value=0.05, help="Cost per kWh for mining cost estimation")
            block_reward = st.number_input("Block Reward (BTC)", min_value=0.0, max_value=50.0, value=6.25, help="Current block reward per block")
            blocks_per_day = st.number_input("Blocks Per Day", min_value=100.0, max_value=200.0, value=144.0, help="Approx blocks mined per day")
        
        with st.expander("Model-Specific Inputs"):
            s2f_intercept = st.number_input("S2F Intercept", min_value=0.0, max_value=100.0, value=14.6, help="S2F model intercept for log(price) = intercept + slope * S/F")
            s2f_slope = st.number_input("S2F Slope", min_value=0.0, max_value=1.0, value=0.05, help="S2F model slope for log(price) = intercept + slope * S/F")
            metcalfe_coeff = st.number_input("Metcalfe Coefficient", min_value=0.0, max_value=0.01, value=0.0001, help="Scaling factor for Metcalfe's Law (value = coeff * addresses^2)")
        
        with st.expander("Sentiment Inputs"):
            fear_greed = st.number_input("Fear & Greed Index (0-100)", min_value=0, max_value=100, value=50, help="0=Extreme Fear, 100=Extreme Greed")
            social_volume = st.number_input("Social Volume (Mentions/Day)", min_value=0.0, value=10000.0, help="Social media mentions (X, Reddit)")
            sentiment_score = st.number_input("Sentiment Score (-1 to 1)", min_value=-1.0, max_value=1.0, value=0.5, help="Positive=Bullish, Negative=Bearish")
        
        with st.expander("Macro Inputs"):
            us_inflation = st.number_input("US Inflation Rate (%)", min_value=0.0, max_value=50.0, value=3.0, help="Annual US inflation rate")
            fed_rate = st.number_input("Fed Interest Rate (%)", min_value=0.0, max_value=50.0, value=5.0, help="Federal Reserve interest rate")
            sp_correlation = st.number_input("S&P 500 Correlation (0-1)", min_value=0.0, max_value=1.0, value=0.5, help="BTC-S&P 500 correlation")
            gold_price = st.number_input("Gold Price (USD/oz)", min_value=0.0, value=2000.0, help="Gold price for comparison")
        
        with st.expander("Technical Inputs"):
            rsi = st.number_input("RSI (14-day)", min_value=0.0, max_value=100.0, value=50.0, help="Overbought >70, Oversold <30")
            ma_50 = st.number_input("50-Day MA", min_value=0.0, value=57000.0, help="50-day moving average")
            ma_200 = st.number_input("200-Day MA", min_value=0.0, value=54000.0, help="200-day moving average")
        
        with st.expander("Monte Carlo Settings"):
            monte_carlo_runs = st.number_input("Number of Runs", min_value=100, max_value=2000, value=1000, help="100-2000 runs")
            volatility_adj = st.number_input("Volatility Adjustment Range (±%)", min_value=0.0, max_value=50.0, value=30.0, help="Volatility variation")
            growth_adj = st.number_input("Growth Adjustment Range (±%)", min_value=0.0, max_value=50.0, value=20.0, help="Growth rate variation")
        
        beta = st.number_input("Beta (vs. Market)", min_value=0.0, value=1.5, help="BTC's market risk vs S&P 500")
        
        data = fetch_bitcoin_data(electricity_cost)
        data.update({
            's2f_intercept': s2f_intercept,
            's2f_slope': s2f_slope,
            'metcalfe_coeff': metcalfe_coeff,
            'block_reward': block_reward,
            'blocks_per_day': blocks_per_day,
            'electricity_cost': electricity_cost
        })
        
        calculate = st.button("Calculate")
        add_to_portfolio = st.button("Add to Portfolio")
        export = st.button("Export Portfolio")
        download_report = st.button("Download Report")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.header("Valuation Dashboard")
        if calculate:
            with st.spinner("Calculating valuation..."):
                inputs = {
                    'model': model,
                    'current_price': current_price,
                    'total_supply': total_supply,
                    'circulating_supply': circulating_supply,
                    'next_halving_date': next_halving_date,
                    'margin_of_safety': margin_of_safety,
                    'hash_rate': hash_rate,
                    'active_addresses': active_addresses,
                    'transaction_volume': transaction_volume,
                    'mvrv': mvrv,
                    'sopr': sopr,
                    'realized_cap': realized_cap,
                    'puell_multiple': puell_multiple,
                    'mining_cost': data['mining_cost'],
                    'fear_greed': fear_greed,
                    'social_volume': social_volume,
                    'sentiment_score': sentiment_score,
                    'us_inflation': us_inflation,
                    'fed_rate': fed_rate,
                    'sp_correlation': sp_correlation,
                    'gold_price': gold_price,
                    'rsi': rsi,
                    '50_day_ma': ma_50,
                    '200_day_ma': ma_200,
                    'desired_return': desired_return,
                    'monte_carlo_runs': monte_carlo_runs,
                    'volatility_adj': volatility_adj,
                    'growth_adj': growth_adj,
                    'beta': beta,
                    'market_cap': current_price * circulating_supply,
                    's2f_intercept': s2f_intercept,
                    's2f_slope': s2f_slope,
                    'metcalfe_coeff': metcalfe_coeff,
                    'block_reward': block_reward,
                    'blocks_per_day': blocks_per_day,
                    'electricity_cost': electricity_cost
                }
                
                if validate_inputs(inputs):
                    results = calculate_valuation(inputs)
                    st.session_state.results = results
                    st.session_state.data = inputs
                    
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
    
    with col_right:
        st.header("Portfolio Overview")
        portfolio_beta = st.session_state.portfolio['Beta'].mean() if not st.session_state.portfolio.empty else 0
        expected_return = portfolio_beta * 8.0
        st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
        st.metric("Portfolio Expected Return", f"{expected_return:.2f}%")
        
        if add_to_portfolio and 'results' in st.session_state:
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
        
        if download_report and 'results' in st.session_state:
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
        if 'results' in st.session_state:
            heatmap = plot_heatmap(st.session_state.results.get('intrinsic_value', 0), volatility_adj, growth_adj)
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)
        
        st.header("Monte Carlo Simulation")
        if 'results' in st.session_state:
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
        if 'results' in st.session_state:
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
            comp_plot = plot_model_comparison(model_comp)
            if comp_plot:
                st.plotly_chart(comp_plot, use_container_width=True)

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
