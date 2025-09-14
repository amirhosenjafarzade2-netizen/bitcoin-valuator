import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_heatmap(intrinsic_value, volatility, growth_rate):
    """
    Generate a static heatmap for sensitivity analysis.
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
            colorscale='Viridis',
            hovertemplate='Volatility: %{x:.2f}%<br>Growth: %{y:.2f}%<br>Value: $%{z:.2f}<extra></extra>',
            zmin=np.min(z) * 0.8,
            zmax=np.max(z) * 1.2
        ))
        
        fig.update_layout(
            title='Sensitivity Analysis: Intrinsic Value vs Volatility and Growth Rate',
            xaxis_title='Volatility (%)',
            yaxis_title='Growth Rate (%)',
            template='plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            height=400,
            staticPlot=True
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error in heatmap: {str(e)}")
        return None

def plot_monte_carlo(mc_results):
    """
    Generate a static histogram for Monte Carlo simulation results.
    """
    try:
        values = mc_results.get('values', np.random.normal(60000, 10000, 1000))
        avg_value = mc_results.get('avg_value', np.mean(values))
        std_dev = mc_results.get('std_dev', np.std(values))
        
        fig = px.histogram(
            x=values,
            nbins=50,
            title='Monte Carlo Simulation: Distribution of Intrinsic Values',
            labels={'x': 'Intrinsic Value ($)', 'y': 'Frequency'},
            template='plotly_white',
            color_discrete_sequence=['#636EFA']
        )
        
        fig.add_vline(x=avg_value, line_dash="dash", line_color="green", annotation_text="Mean", annotation_position="top")
        fig.add_vline(x=avg_value - std_dev, line_dash="dot", line_color="orange", annotation_text="-1 SD", annotation_position="top left")
        fig.add_vline(x=avg_value + std_dev, line_dash="dot", line_color="orange", annotation_text="+1 SD", annotation_position="top right")
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50),
            height=400,
            staticPlot=True
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error in Monte Carlo plot: {str(e)}")
        return None

def plot_model_comparison(model_comp_df):
    """
    Generate a static bar chart comparing intrinsic values across models.
    """
    try:
        fig = px.bar(
            model_comp_df,
            x='Model',
            y='Intrinsic Value',
            title='Model Comparison: Intrinsic Values',
            labels={'Intrinsic Value': 'Intrinsic Value ($)'},
            template='plotly_white',
            color='Model',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50),
            height=400,
            staticPlot=True
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error in model comparison plot: {str(e)}")
        return None

def plot_onchain_metrics(data, history):
    """
    Plot static on-chain metrics over time.
    """
    try:
        if history.empty:
            st.warning("No historical data available for on-chain metrics.")
            return None
        
        df = history.copy()
        df['NVT'] = df['Close'] * data['circulating_supply'] / df['estimated-transaction-volume-usd']
        df['MVRV'] = df['Close'] * data['circulating_supply'] / data['realized_cap']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['NVT'], name='NVT Ratio', line=dict(color='#636EFA')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MVRV'], name='MVRV Ratio', line=dict(color='#EF553B')))
        
        fig.update_layout(
            title='On-Chain Metrics: NVT and MVRV Over Time',
            xaxis_title='Date',
            yaxis_title='Ratio',
            template='plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            height=400,
            staticPlot=True,
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error in on-chain plot: {str(e)}")
        return None

def plot_sentiment_analysis(data, history):
    """
    Plot static sentiment metrics with price.
    """
    try:
        if history.empty:
            st.warning("No historical data available for sentiment analysis.")
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.index, y=history['Close'], name='Price ($)', line=dict(color='#636EFA')))
        fig.add_trace(go.Scatter(x=history.index, y=[data['fear_greed']] * len(history), name='Fear & Greed', line=dict(color='#EF553B', dash='dash')))
        
        fig.update_layout(
            title='Sentiment: Price vs Fear & Greed Index',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            height=400,
            staticPlot=True,
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error in sentiment plot: {str(e)}")
        return None

def plot_price_history(history):
    """
    Plot static Bitcoin price history.
    """
    try:
        if history.empty:
            st.warning("No historical price data available.")
            return None
        
        fig = px.line(
            history,
            x=history.index,
            y='Close',
            title='Bitcoin Price History',
            labels={'Close': 'Price ($)'},
            template='plotly_white'
        )
        
        fig.update_layout(
            margin=dict(l=50, r=50, t=80, b=50),
            height=400,
            staticPlot=True
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error in price history plot: {str(e)}")
        return None

def plot_macro_correlations(data, history):
    """
    Plot static Bitcoin price vs S&P 500.
    """
    try:
        if history.empty:
            st.warning("No historical data for macro correlations.")
            return None
        
        sp_hist = yf.download('^GSPC', period='5y')['Close']
        sp_hist = sp_hist.reindex(history.index, method='ffill')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.index, y=history['Close'], name='BTC Price', line=dict(color='#636EFA')))
        fig.add_trace(go.Scatter(x=sp_hist.index, y=sp_hist, name='S&P 500', line=dict(color='#EF553B')))
        
        fig.update_layout(
            title='Macro Correlations: BTC vs S&P 500',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            height=400,
            staticPlot=True
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error in macro correlations plot: {str(e)}")
        return None
