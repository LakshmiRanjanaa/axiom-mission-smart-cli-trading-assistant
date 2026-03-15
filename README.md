# Smart CLI Trading Assistant

A command-line tool that uses machine learning to analyze stock price patterns and provide trading insights.

## Features
- Fetch real-time stock data using yfinance API
- ML-powered trend analysis using scikit-learn
- Easy-to-use CLI interface with Click framework
- Actionable trading insights and recommendations

## Installation

bash
pip install -r requirements.txt


## Usage

bash
# Analyze a stock with default 30-day period
python trading_assistant.py analyze AAPL

# Analyze with custom time period
python trading_assistant.py analyze TSLA --days 60

# Get help
python trading_assistant.py --help


## Commands

- `analyze SYMBOL`: Analyze stock trends and get trading insights
- `--days N`: Number of days to analyze (default: 30)

## Example Output


Analyzing AAPL for the last 30 days...
✓ Data fetched successfully
✓ ML model trained

Trend Analysis:
- Current Trend: BULLISH
- Confidence: 78.5%
- Recommendation: HOLD

Key Insights:
- Price increased 5.2% over period
- Volatility: MODERATE
- Support Level: $185.50
