import pandas as pd
from datetime import datetime
import logging
import streamlit as st
from fpdf import FPDF
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_inputs(inputs):
    """
    Validate inputs with a streamlined dictionary-based approach, displaying errors in Streamlit.
    Returns (is_valid, errors, sanitized_inputs).
    """
    errors = []
    sanitized_inputs = inputs.copy()
    
    # Define validation rules: (type, required, range, warning_range, warning_message)
    validation_rules = {
        'model': (str, True, None, None, None),
        'current_price': (float, True, (0, float('inf')), (10000, 100000), "expected $60,000-$65,000"),
        'total_supply': (float, True, (0, float('inf')), (21000000, 21000000), "expected 21M"),
        'circulating_supply': (float, True, (0, float('inf')), (19000000, 21000000), "expected 19.5M-20M"),
        'next_halving_date': (datetime, True, None, None, None),
        'margin_of_safety': (float, True, (0, 100), None, None),
        'hash_rate': (float, True, (0, float('inf')), (400, 750), "expected 500-750 EH/s"),
        'active_addresses': (float, True, (0, float('inf')), (500000, 1500000), "expected 800K-1.5M"),
        'transaction_volume': (float, True, (0, float('inf')), (5e8, 2e10), "expected $1B-$20B"),
        'mvrv': (float, True, (0, float('inf')), (0.5, 5.0), "expected 1.5-2.5"),
        'sopr': (float, True, (0, float('inf')), (0.5, 1.5), "expected 0.9-1.1"),
        'realized_cap': (float, True, (0, float('inf')), (4e11, 1e12), "expected $600B-$1T"),
        'puell_multiple': (float, True, (0, float('inf')), (0.3, 5.0), "expected 0.5-1.5"),
        'mining_cost': (float, True, (0, float('inf')), (5000, 15000), "expected $5,000-$15,000"),
        'fear_greed': (int, True, (0, 100), (20, 80), "expected 40-60"),
        'social_volume': (float, True, (0, float('inf')), None, None),
        'sentiment_score': (float, True, (-1, 1), None, None),
        'us_inflation': (float, True, (0, float('inf')), (0, 10), "expected 2.5-3.5%"),
        'fed_rate': (float, True, (0, float('inf')), (0, 10), "expected 4.5-5.0%"),
        'sp_correlation': (float, True, (0, 1), None, None),
        'gold_price': (float, True, (0, float('inf')), (1500, 3000), "expected $2,000-$3,000"),
        'rsi': (float, True, (0, 100), (10, 90), "expected 40-60"),
        '50_day_ma': (float, True, (0, float('inf')), (40000, 80000), "expected $55,000-$65,000"),
        '200_day_ma': (float, True, (0, float('inf')), (40000, 80000), "expected $50,000-$60,000"),
        'desired_return': (float, True, (0, float('inf')), None, None),
        'monte_carlo_runs': (int, True, (100, float('inf')), None, None),
        'volatility_adj': (float, True, (0, float('inf')), (0, 50), "expected 10-30%"),
        'growth_adj': (float, True, (0, float('inf')), (0, 50), "expected 10-30%"),
        'beta': (float, True, (0, float('inf')), None, None),
        'market_cap': (float, True, (0, float('inf')), None, None),
        's2f_intercept': (float, True, (-10, 10), (-2, 2), "expected ~ -1.84"),
        's2f_slope': (float, True, (0, float('inf')), (0, 10), "expected ~3.96"),
        'metcalfe_coeff': (float, True, (0, float('inf')), (0.1, 10), "expected ~15.0"),
        'block_reward': (float, True, (0, float('inf')), (3.125, 3.125), "expected 3.125 post-2024"),
        'blocks_per_day': (float, True, (0, float('inf')), (100, 200), "expected ~144"),
        'electricity_cost': (float, True, (0, float('inf')), (0.05, 0.10), "expected $0.05-$0.10/kWh")
    }

    # Fallback values for sanitization (realistic for 2025)
    fallback_values = {
        'current_price': 65000.0,
        'total_supply': 21000000,
        'circulating_supply': 19700000,
        'next_halving_date': datetime(2028, 4, 1),
        'margin_of_safety': 25.0,
        'hash_rate': 750.0,
        'active_addresses': 1050000,
        'transaction_volume': 15.2e9,
        'mvrv': 2.0,
        'sopr': 1.0,
        'realized_cap': 680e9,
        'puell_multiple': 1.0,
        'mining_cost': 10000.0,
        'fear_greed': 50,
        'social_volume': 10000,
        'sentiment_score': 0.5,
        'us_inflation': 2.8,
        'fed_rate': 4.75,
        'sp_correlation': 0.5,
        'gold_price': 2500.0,
        'rsi': 50.0,
        '50_day_ma': 62000.0,
        '200_day_ma': 58000.0,
        'desired_return': 15.0,
        'monte_carlo_runs': 1000,
        'volatility_adj': 10.0,
        'growth_adj': 20.0,
        'beta': 1.5,
        'market_cap': 1.28e12,
        's2f_intercept': -1.84,
        's2f_slope': 3.96,
        'metcalfe_coeff': 15.0,
        'block_reward': 3.125,
        'blocks_per_day': 144,
        'electricity_cost': 0.05
    }

    for key, (expected_type, required, valid_range, warning_range, warning_message) in validation_rules.items():
        # Check for missing or None values
        if required and (key not in inputs or inputs[key] is None):
            errors.append(f"Missing or None value for {key}")
            sanitized_inputs[key] = fallback_values.get(key)
            continue

        # Type conversion and validation
        if key in inputs and inputs[key] is not None:
            try:
                if expected_type == datetime:
                    if not isinstance(inputs[key], datetime):
                        sanitized_inputs[key] = datetime.strptime(str(inputs[key]), '%Y-%m-%d')
                elif expected_type == int:
                    sanitized_inputs[key] = int(inputs[key])
                else:
                    sanitized_inputs[key] = float(inputs[key])
            except (TypeError, ValueError):
                errors.append(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(inputs[key]).__name__}")
                sanitized_inputs[key] = fallback_values.get(key)
                continue

            # Range validation
            if valid_range and isinstance(sanitized_inputs[key], (int, float)):
                min_val, max_val = valid_range
                if sanitized_inputs[key] < min_val or sanitized_inputs[key] > max_val:
                    errors.append(f"{key.replace('_', ' ').title()} must be between {min_val} and {max_val}")
                    sanitized_inputs[key] = fallback_values.get(key)

            # Warning for unusual values
            if warning_range and isinstance(sanitized_inputs[key], (int, float)):
                warn_min, warn_max = warning_range
                if sanitized_inputs[key] < warn_min or sanitized_inputs[key] > warn_max:
                    logging.warning(f"{key.replace('_', ' ').title()} {sanitized_inputs[key]} is unusual, {warning_message}")

    # Additional consistency check for market cap
    if 'market_cap' in sanitized_inputs and 'current_price' in sanitized_inputs and 'circulating_supply' in sanitized_inputs:
        expected_market_cap = sanitized_inputs['current_price'] * sanitized_inputs['circulating_supply']
        if abs(sanitized_inputs['market_cap'] - expected_market_cap) > 1e9:
            logging.warning(f"Market cap ${sanitized_inputs['market_cap']:.2e} inconsistent with price*supply")
            sanitized_inputs['market_cap'] = expected_market_cap

    if errors:
        st.error(f"Input validation errors: {'; '.join(errors)}")
    else:
        st.success("Input validation successful")
    
    return len(errors) == 0, errors, sanitized_inputs

def export_portfolio(portfolio):
    """
    Export portfolio to CSV with user feedback.
    """
    try:
        portfolio.to_csv("portfolio_export.csv", index=False)
        logging.info("Portfolio exported to portfolio_export.csv")
        st.success("Portfolio exported successfully to portfolio_export.csv")
        return True
    except Exception as e:
        logging.error(f"Portfolio export failed: {str(e)}")
        st.error(f"Failed to export portfolio: {str(e)}")
        return False

def parse_uploaded_file(file):
    """
    Parse uploaded CSV file for custom data, returning a DataFrame.
    """
    try:
        if file is None:
            raise ValueError("No file uploaded")
        df = pd.read_csv(file)
        required_columns = ['Asset', 'Intrinsic Value', 'Undervaluation %', 'Verdict', 'Beta']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Uploaded CSV must contain columns: {', '.join(required_columns)}")
        logging.info("Parsed uploaded CSV successfully")
        st.success("Uploaded CSV parsed successfully")
        return df
    except Exception as e:
        logging.error(f"Failed to parse uploaded file: {str(e)}")
        st.error(f"Failed to parse uploaded file: {str(e)}")
        return None

def sample_data(df, period='5y'):
    """
    Sample DataFrame to reduce size based on period, inspired by second program's sampling.
    """
    try:
        sample_rate = {'1mo': 2, '3mo': 5, '1y': 10, '5y': 20}.get(period, 20)
        sampled_df = df.iloc[::sample_rate].copy()
        logging.info(f"Sampled data: {len(df)} rows reduced to {len(sampled_df)} rows for period {period}")
        st.info(f"Sampled historical data to {len(sampled_df)} points for {period}")
        return sampled_df
    except Exception as e:
        logging.error(f"Data sampling failed: {str(e)}")
        st.error(f"Data sampling failed: {str(e)}")
        return df

def generate_pdf_report(results, portfolio, model_comp_fig):
    """
    Generate a PDF report with valuation results, portfolio, and model comparison chart.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", size=16)
        pdf.cell(200, 10, txt="Bitcoin Valuation Report", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(10)
        
        # Valuation Results
        pdf.set_font("Arial", "B", size=12)
        pdf.cell(200, 10, txt="Valuation Results", ln=True)
        pdf.set_font("Arial", size=10)
        for key, value in results.items():
            if isinstance(value, (int, float)):
                pdf.cell(200, 10, txt=f"{key.replace('_', ' ').title()}: ${value:,.2f}", ln=True)
            else:
                pdf.cell(200, 10, txt=f"{key.replace('_', ' ').title()}: {value}", ln=True)
        
        # Portfolio Overview
        pdf.ln(10)
        pdf.set_font("Arial", "B", size=12)
        pdf.cell(200, 10, txt="Portfolio Overview", ln=True)
        pdf.set_font("Arial", size=10)
        for _, row in portfolio.iterrows():
            pdf.cell(200, 10, txt=f"Asset: {row['Asset']}, Intrinsic Value: ${row['Intrinsic Value']:,.2f}, "
                                  f"Undervaluation: {row['Undervaluation %']:.2f}%, Verdict: {row['Verdict']}, "
                                  f"Beta: {row['Beta']:.2f}", ln=True)
        
        # Add Model Comparison Chart
        if model_comp_fig:
            pdf.ln(10)
            pdf.set_font("Arial", "B", size=12)
            pdf.cell(200, 10, txt="Model Comparison Chart", ln=True)
            # Save figure to buffer
            buffer = BytesIO()
            model_comp_fig.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            # Encode image to base64
            img_str = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            # Embed image in PDF
            pdf.image(img_str, x=10, y=None, w=190, type='PNG')
        
        pdf_file = "bitcoin_valuation_report.pdf"
        pdf.output(pdf_file)
        with open(pdf_file, "rb") as f:
            pdf_data = f.read()
        logging.info("PDF report generated successfully")
        st.success("PDF report generated successfully")
        return pdf_data
    except Exception as e:
        logging.error(f"PDF report generation failed: {str(e)}")
        st.error(f"Failed to generate PDF report: {str(e)}")
        return None
