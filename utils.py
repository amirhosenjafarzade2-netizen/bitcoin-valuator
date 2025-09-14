import streamlit as st
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io
import plotly.io as pio
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_inputs(inputs):
    """
    Validate input parameters.
    Returns True if valid, False otherwise.
    """
    ranges = {
        'current_price': (0, float('inf')),
        'circulating_supply': (0, float('inf')),
        'total_supply': (0, float('inf')),
        'desired_return': (0, 50),
        'margin_of_safety': (0, 100),
        'hash_rate': (0, float('inf')),
        'active_addresses': (0, float('inf')),
        'transaction_volume': (0, float('inf')),
        'mvrv': (0, float('inf')),
        'sopr': (0, float('inf')),
        'realized_cap': (0, float('inf')),
        'puell_multiple': (0, float('inf')),
        'mining_cost': (0, float('inf')),
        'fear_greed': (0, 100),
        'social_volume': (0, float('inf')),
        'sentiment_score': (-1, 1),
        'us_inflation': (0, 50),
        'fed_rate': (0, 50),
        'sp_correlation': (0, 1),
        'gold_price': (0, float('inf')),
        'rsi': (0, 100),
        '50_day_ma': (0, float('inf')),
        '200_day_ma': (0, float('inf')),
        'monte_carlo_runs': (100, 2000),
        'volatility_adj': (0, 50),
        'growth_adj': (0, 50),
        'beta': (0, float('inf')),
        's2f_intercept': (0, 100),
        's2f_slope': (0, 1),
        'metcalfe_coeff': (0, 0.01),
        'electricity_cost': (0, 1),
        'block_reward': (0, 50),
        'blocks_per_day': (100, 200),
        # New model metrics (post-calculation, but validate inputs for them)
        'rvmr': (0, 100),
        'mayer_multiple': (0, 5),
        'sentiment_index': (0, 2)
    }
    
    try:
        for key, (min_val, max_val) in ranges.items():
            if key in inputs and not (min_val <= inputs[key] <= max_val):
                st.warning(f"{key.replace('_', ' ').title()} must be between {min_val} and {max_val}.")
                return False
        
        # Logical checks
        if inputs['50_day_ma'] > inputs['200_day_ma'] * 1.5:
            st.warning("50-day MA significantly exceeds 200-day MA.")
            return False
        if inputs['circulating_supply'] > inputs['total_supply']:
            st.warning("Circulating supply cannot exceed total supply.")
            return False
        
        return True
    except Exception as e:
        logging.error(f"Error validating inputs: {str(e)}")
        return False

def export_portfolio(portfolio_df, filename="portfolio.csv"):
    """
    Export portfolio DataFrame to CSV.
    """
    try:
        csv = portfolio_df.to_csv(index=False)
        st.download_button(
            label="Download Portfolio CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
    except Exception as e:
        logging.error(f"Error exporting portfolio: {str(e)}")
        st.warning("Failed to export portfolio.")

def generate_pdf_report(results, portfolio_df, model_comp_fig):
    """
    Generate a PDF report with valuation results, portfolio, and model comparison chart.
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=30, alignment=1)
        story.append(Paragraph("Bitcoin Valuation Report", title_style))
        story.append(Spacer(1, 12))
        
        # Valuation Results
        story.append(Paragraph("Valuation Results", styles['Heading2']))
        results_data = [
            ['Metric', 'Value'],
            ['Model', results.get('model', '-')],
            ['Current Price', f"${results.get('current_price', 0):.2f}"],
            ['Intrinsic Value', f"${results.get('intrinsic_value', 0):.2f}"],
            ['Safe Buy Price', f"${results.get('safe_buy_price', 0):.2f}"],
            ['Undervaluation %', f"{results.get('undervaluation', 0):.2f}%"],
            ['Verdict', results.get('verdict', '-')],
            ['Overall Score', f"{results.get('score', 0)}/100"],
            ['NVT Ratio', f"{results.get('nvt_ratio', 0):.2f}"],
            ['MVRV Z-Score', f"{results.get('mvrv_z_score', 0):.2f}"],
            ['SOPR Signal', results.get('sopr_signal', '-')],
            ['Puell Multiple Signal', results.get('puell_signal', '-')],
            ['Mining Cost vs Price', f"{results.get('mining_cost_vs_price', 0):.2f}%"],
            ['RVMR', f"{results.get('rvmr', 0):.2f}"],
            ['Mayer Multiple', f"{results.get('mayer_multiple', 0):.2f}"],
            ['Sentiment Index (MSC)', f"{results.get('sentiment_index', 0):.2f}"],
            ['Hash Ribbons Signal', results.get('hash_ribbon_signal', '-')]
        ]
        results_table = Table(results_data)
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(results_table)
        story.append(Spacer(1, 12))
        
        # Model Comparison Chart
        story.append(Paragraph("Model Comparison", styles['Heading2']))
        if model_comp_fig:
            try:
                model_comp_fig.write_image('model_comp.png', format='PNG', width=400, height=200)
                story.append(Image('model_comp.png', width=400, height=200))
            except Exception as e:
                logging.error(f"Error adding model comparison chart to PDF: {str(e)}")
        
        # Model Comparison Table
        model_data = [
            ['Model', 'Intrinsic Value'],
            ['S2F', f"${results.get('s2f_value', 0):.2f}"],
            ['Metcalfe', f"${results.get('metcalfe_value', 0):.2f}"],
            ['NVT', f"${results.get('nvt_value', 0):.2f}"],
            ['Pi Cycle', f"${results.get('pi_cycle_value', 0):.2f}"],
            ['Reverse S2F', f"${results.get('reverse_s2f_value', 0):.2f}"],
            ['MSC', f"${results.get('msc_value', 0):.2f}"],
            ['Energy Value', f"${results.get('energy_value', 0):.2f}"],
            ['RVMR', f"${results.get('rvmr_value', 0):.2f}"],
            ['Mayer Multiple', f"${results.get('mayer_multiple_value', 0):.2f}"],
            ['Hash Ribbons', f"${results.get('hash_ribbons_value', 0):.2f}"]
        ]
        model_table = Table(model_data)
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(model_table)
        story.append(Spacer(1, 12))
        
        # Portfolio
        if not portfolio_df.empty:
            story.append(Paragraph("Portfolio Overview", styles['Heading2']))
            portfolio_data = [['Asset', 'Intrinsic Value', 'Undervaluation %', 'Verdict', 'Beta']] + portfolio_df.values.tolist()
            portfolio_table = Table(portfolio_data)
            portfolio_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(portfolio_table)
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("Disclaimer: This tool is for informational purposes only and not financial advice.", styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    except Exception as e:
        logging.error(f"Error generating PDF: {str(e)}")
        return None
