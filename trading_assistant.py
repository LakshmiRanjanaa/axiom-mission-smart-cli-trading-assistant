#!/usr/bin/env python3
"""
Smart CLI Trading Assistant
A machine learning-powered stock analysis tool
"""

import click
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from stock_analyzer import StockAnalyzer
from utils import format_currency, format_percentage, print_banner

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Smart CLI Trading Assistant - ML-powered stock analysis"""
    print_banner()

@cli.command()
@click.argument('symbol')
@click.option('--days', '-d', default=30, help='Number of days to analyze')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed analysis')
def analyze(symbol, days, verbose):
    """Analyze stock trends and get trading insights"""
    
    # Validate inputs
    if days < 7 or days > 365:
        click.echo(click.style("Error: Days must be between 7 and 365", fg='red'))
        return
    
    click.echo(f"\n🔍 Analyzing {symbol.upper()} for the last {days} days...")
    
    try:
        # Initialize analyzer
        analyzer = StockAnalyzer(symbol.upper())
        
        # Fetch data
        with click.progressbar(length=3, label='Fetching data') as bar:
            data = analyzer.fetch_data(days)
            bar.update(1)
            
            # Train ML model
            prediction, confidence = analyzer.train_and_predict(data)
            bar.update(1)
            
            # Generate insights
            insights = analyzer.generate_insights(data, prediction)
            bar.update(1)
        
        # Display results
        display_analysis(symbol.upper(), data, prediction, confidence, insights, verbose)
        
    except Exception as e:
        click.echo(click.style(f"❌ Error: {str(e)}", fg='red'))
        if verbose:
            import traceback
            traceback.print_exc()

def display_analysis(symbol, data, prediction, confidence, insights, verbose):
    """Display the analysis results in a formatted way"""
    
    current_price = data['Close'].iloc[-1]
    price_change = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
    
    click.echo("\n" + "="*50)
    click.echo(f"📊 ANALYSIS RESULTS for {symbol}")
    click.echo("="*50)
    
    # Current status
    click.echo(f"💰 Current Price: {format_currency(current_price)}")
    
    price_color = 'green' if price_change > 0 else 'red'
    click.echo(f"📈 Price Change: {click.style(format_percentage(price_change), fg=price_color)}")
    
    # ML Prediction
    trend_color = 'green' if prediction > 0 else 'red'
    trend_text = 'BULLISH 📈' if prediction > 0 else 'BEARISH 📉'
    
    click.echo(f"\n🤖 ML PREDICTION:")
    click.echo(f"   Trend: {click.style(trend_text, fg=trend_color)}")
    click.echo(f"   Confidence: {format_percentage(confidence)}")
    
    # Insights
    click.echo(f"\n💡 KEY INSIGHTS:")
    for insight in insights:
        click.echo(f"   • {insight}")
    
    if verbose:
        click.echo(f"\n📊 TECHNICAL DATA:")
        click.echo(f"   • Volatility: {format_percentage(data['Close'].std() / data['Close'].mean() * 100)}")
        click.echo(f"   • 5-day avg: {format_currency(data['Close'].tail(5).mean())}")
        click.echo(f"   • Volume avg: {data['Volume'].mean():,.0f}")

if __name__ == '__main__':
    cli()