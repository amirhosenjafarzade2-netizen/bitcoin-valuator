import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Bitcoin-themed color palette
BITCOIN_ORANGE = '#F7931A'
COMPLEMENTARY_COLORS = ['#1A3C6D', '#EF553B', '#00CC96', '#AB47BC']

def plot_heatmap(intrinsic_value, volatility, growth_rate):
    """
    Generate an interactive heatmap for sensitivity analysis with Bitcoin branding.
    """
    try:
        volatility_range = np.linspace(max(0.01, volatility - 10), min(50, volatility + 10), 10)
        growth_range = np.linspace(max(0, growth_rate - 10), min(50, growth_rate + 10), 10)
        
        z = np.zeros((len(growth_range), len(volatility_range)))
        for i, g in enumerate(growth_range):
            for j, v in enumerate(volatility_range):
                z[i, j] = intrinsic_value * (1 + g / 100) / (1 + v / 100)
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=volatility_range,
            y=growth_range,
            colorscale=[[0, '#1A3C6D'], [0.5, '#F7931A'], [1, '#EF553B']],
            hovertemplate='Volatility: %{x:.2f}%<br>Growth: %{y:.2f}%<br>Value: $%{z:,.2f}<extra></extra>',
            zmin=np.min(z) * 0.8,
            zmax=np.max(z) * 1.2
        ))
        
        fig.update_layout(
            title='Sensitivity Analysis: Intrinsic Value vs Volatility and Growth Rate',
            xaxis_title='Volatility (%)',
            yaxis_title='Growth Rate (%)',
            template='plotly_dark' if st.session_state.get('dark_mode', False) else 'plotly_white',
            margin=dict(l=20, r=20, t=80, b=20),
            height=500,
            xaxis=dict(gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            yaxis=dict(gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            hovermode='closest',
            responsive=True
        )
        
        logging.info("Heatmap generated successfully")
        st.success("Sensitivity analysis heatmap generated")
        return fig
    except Exception as e:
        logging.error(f"Error in heatmap: {str(e)}")
        st.error(f"Failed to generate heatmap: {str(e)}")
        return None

def plot_monte_carlo(mc_results):
    """
    Generate an interactive histogram for Monte Carlo simulation results.
    """
    try:
        values = mc_results.get('values', np.random.normal(115000, 20000, 1000))  # Updated for 2025
        avg_value = mc_results.get('avg_value', np.mean(values))
        std_dev = mc_results.get('std_dev', np.std(values))
        
        fig = px.histogram(
            x=values,
            nbins=50,
            title='Monte Carlo Simulation: Distribution of Intrinsic Values',
            labels={'x': 'Intrinsic Value ($)', 'y': 'Frequency'},
            template='plotly_dark' if st.session_state.get('dark_mode', False) else 'plotly_white',
            color_discrete_sequence=[BITCOIN_ORANGE]
        )
        
        fig.add_vline(x=avg_value, line_dash="dash", line_color=COMPLEMENTARY_COLORS[0], annotation_text="Mean", annotation_position="top")
        fig.add_vline(x=avg_value - std_dev, line_dash="dot", line_color=COMPLEMENTARY_COLORS[2], annotation_text="-1 SD", annotation_position="top left")
        fig.add_vline(x=avg_value + std_dev, line_dash="dot", line_color=COMPLEMENTARY_COLORS[2], annotation_text="+1 SD", annotation_position="top right")
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=80, b=20),
            height=500,
            xaxis=dict(gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            yaxis=dict(gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            hovermode='x unified',
            responsive=True
        )
        
        logging.info("Monte Carlo histogram generated successfully")
        st.success("Monte Carlo simulation histogram generated")
        return fig
    except Exception as e:
        logging.error(f"Error in Monte Carlo plot: {str(e)}")
        st.error(f"Failed to generate Monte Carlo histogram: {str(e)}")
        return None

def plot_model_comparison(model_comp_df):
    """
    Generate an interactive bar chart comparing intrinsic values across models.
    """
    try:
        if not isinstance(model_comp_df, pd.DataFrame) or model_comp_df.empty:
            raise ValueError("Invalid or empty model comparison DataFrame")
        
        fig = px.bar(
            model_comp_df,
            x='Model',
            y='Intrinsic Value',
            title='Model Comparison: Intrinsic Values',
            labels={'Intrinsic Value': 'Intrinsic Value ($)'},
            template='plotly_dark' if st.session_state.get('dark_mode', False) else 'plotly_white',
            color='Model',
            color_discrete_sequence=[BITCOIN_ORANGE] + COMPLEMENTARY_COLORS
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
            margin=dict(l=20, r=20, t=80, b=20),
            height=500,
            xaxis=dict(tickangle=45, gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            yaxis=dict(gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            hovermode='x unified',
            responsive=True
        )
        
        logging.info("Model comparison bar chart generated successfully")
        st.success("Model comparison bar chart generated")
        return fig
    except Exception as e:
        logging.error(f"Error in model comparison plot: {str(e)}")
        st.error(f"Failed to generate model comparison chart: {str(e)}")
        return None

def plot_onchain_metrics(data, history):
    """
    Generate an interactive dual-axis plot for on-chain metrics (NVT, MVRV) with price.
    """
    try:
        if history.empty or 'Close' not in history or 'estimated-transaction-volume-usd' not in history:
            st.warning("No valid historical data for on-chain metrics.")
            logging.warning("Invalid history data for on-chain metrics")
            # Generate synthetic data as fallback
            dates = pd.date_range(end=datetime(2025, 9, 16), periods=365)
            history = pd.DataFrame({
                'Close': np.random.normal(115000, 10000, 365),
                'estimated-transaction-volume-usd': np.random.normal(15e9, 2e9, 365)
            }, index=dates)
        
        df = history.copy()
        df['NVT'] = df['Close'] * data['circulating_supply'] / df['estimated-transaction-volume-usd']
        df['MVRV'] = df['Close'] * data['circulating_supply'] / data['realized_cap']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], name='BTC Price ($)', line=dict(color=BITCOIN_ORANGE), yaxis='y1'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['NVT'], name='NVT Ratio', line=dict(color=COMPLEMENTARY_COLORS[0]), yaxis='y2'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MVRV'], name='MVRV Ratio', line=dict(color=COMPLEMENTARY_COLORS[1]), yaxis='y2'
        ))
        
        fig.update_layout(
            title='On-Chain Metrics: BTC Price vs NVT and MVRV',
            xaxis_title='Date',
            yaxis=dict(title='BTC Price ($)', gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            yaxis2=dict(title='Ratio', overlaying='y', side='right', gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            template='plotly_dark' if st.session_state.get('dark_mode', False) else 'plotly_white',
            margin=dict(l=20, r=20, t=80, b=20),
            height=500,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
            responsive=True
        )
        
        logging.info("On-chain metrics plot generated successfully")
        st.success("On-chain metrics plot generated")
        return fig
    except Exception as e:
        logging.error(f"Error in on-chain plot: {str(e)}")
        st.error(f"Failed to generate on-chain metrics plot: {str(e)}")
        return None

def plot_sentiment_analysis(data, history):
    """
    Generate an interactive dual-axis plot for sentiment metrics with price.
    """
    try:
        if history.empty or 'Close' not in history:
            st.warning("No valid historical data for sentiment analysis.")
            logging.warning("Invalid history data for sentiment analysis")
            # Generate synthetic data as fallback
            dates = pd.date_range(end=datetime(2025, 9, 16), periods=365)
            history = pd.DataFrame({'Close': np.random.normal(115000, 10000, 365)}, index=dates)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history.index, y=history['Close'], name='BTC Price ($)', line=dict(color=BITCOIN_ORANGE), yaxis='y1'
        ))
        fig.add_trace(go.Scatter(
            x=history.index, y=[data['fear_greed']] * len(history), name='Fear & Greed Index', line=dict(color=COMPLEMENTARY_COLORS[1], dash='dash'), yaxis='y2'
        ))
        
        fig.update_layout(
            title='Sentiment Analysis: BTC Price vs Fear & Greed Index',
            xaxis_title='Date',
            yaxis=dict(title='BTC Price ($)', gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            yaxis2=dict(title='Fear & Greed Index', overlaying='y', side='right', gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            template='plotly_dark' if st.session_state.get('dark_mode', False) else 'plotly_white',
            margin=dict(l=20, r=20, t=80, b=20),
            height=500,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
            responsive=True
        )
        
        logging.info("Sentiment analysis plot generated successfully")
        st.success("Sentiment analysis plot generated")
        return fig
    except Exception as e:
        logging.error(f"Error in sentiment plot: {str(e)}")
        st.error(f"Failed to generate sentiment analysis plot: {str(e)}")
        return None

def plot_price_history(history):
    """
    Generate an interactive candlestick chart for Bitcoin price history.
    """
    try:
        if history.empty or not all(col in history for col in ['Open', 'High', 'Low', 'Close']):
            st.warning("No valid historical price data for candlestick chart.")
            logging.warning("Invalid history data for price history")
            # Generate synthetic data as fallback
            dates = pd.date_range(end=datetime(2025, 9, 16), periods=365)
            history = pd.DataFrame({
                'Open': np.random.normal(115000, 10000, 365),
                'High': np.random.normal(116000, 10000, 365),
                'Low': np.random.normal(114000, 10000, 365),
                'Close': np.random.normal(115000, 10000, 365),
                'Volume': np.random.normal(1e9, 2e8, 365)
            }, index=dates)
        
        fig = go.Figure(data=[
            go.Candlestick(
                x=history.index,
                open=history['Open'],
                high=history['High'],
                low=history['Low'],
                close=history['Close'],
                name='BTC Price',
                increasing_line_color=BITCOIN_ORANGE,
                decreasing_line_color=COMPLEMENTARY_COLORS[1]
            )
        ])
        
        fig.update_layout(
            title='Bitcoin Price History (Candlestick)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_dark' if st.session_state.get('dark_mode', False) else 'plotly_white',
            margin=dict(l=20, r=20, t=80, b=20),
            height=500,
            xaxis_rangeslider_visible=True,
            xaxis=dict(gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            yaxis=dict(gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            hovermode='x unified',
            responsive=True
        )
        
        logging.info("Price history candlestick chart generated successfully")
        st.success("Price history candlestick chart generated")
        return fig
    except Exception as e:
        logging.error(f"Error in price history plot: {str(e)}")
        st.error(f"Failed to generate price history chart: {str(e)}")
        return None

def plot_macro_correlations(data, history):
    """
    Generate an interactive dual-axis plot for BTC vs S&P 500 and gold price.
    """
    try:
        if history.empty or 'Close' not in history:
            st.warning("No valid historical data for macro correlations.")
            logging.warning("Invalid history data for macro correlations")
            # Generate synthetic data as fallback
            dates = pd.date_range(end=datetime(2025, 9, 16), periods=365)
            history = pd.DataFrame({
                'Close': np.random.normal(115000, 10000, 365),
                'Volume': np.random.normal(1e9, 2e8, 365)
            }, index=dates)
        
        # Simulate S&P 500 and gold data (replace with real data in app.py)
        sp_hist = pd.DataFrame({
            'Close': np.random.normal(5000, 500, len(history))
        }, index=history.index)
        gold_hist = pd.DataFrame({
            'Close': np.random.normal(2500, 200, len(history))
        }, index=history.index)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history.index, y=history['Close'], name='BTC Price ($)', line=dict(color=BITCOIN_ORANGE), yaxis='y1'
        ))
        fig.add_trace(go.Scatter(
            x=sp_hist.index, y=sp_hist['Close'], name='S&P 500', line=dict(color=COMPLEMENTARY_COLORS[0]), yaxis='y2'
        ))
        fig.add_trace(go.Scatter(
            x=gold_hist.index, y=gold_hist['Close'], name='Gold Price ($)', line=dict(color=COMPLEMENTARY_COLORS[2]), yaxis='y2'
        ))
        
        fig.update_layout(
            title='Macro Correlations: BTC vs S&P 500 and Gold',
            xaxis_title='Date',
            yaxis=dict(title='BTC Price ($)', gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            yaxis2=dict(title='S&P 500 / Gold ($)', overlaying='y', side='right', gridcolor='#444' if st.session_state.get('dark_mode', False) else '#ccc'),
            template='plotly_dark' if st.session_state.get('dark_mode', False) else 'plotly_white',
            margin=dict(l=20, r=20, t=80, b=20),
            height=500,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
            responsive=True
        )
        
        logging.info("Macro correlations plot generated successfully")
        st.success("Macro correlations plot generated")
        return fig
    except Exception as e:
        logging.error(f"Error in macro correlations plot: {str(e)}")
        st.error(f"Failed to generate macro correlations plot: {str(e)}")
        return None
