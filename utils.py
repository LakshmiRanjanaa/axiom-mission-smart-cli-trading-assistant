"""
Utility functions for formatting and display
"""

def format_currency(value):
    """Format a value as currency"""
    return f"${value:.2f}"

def format_percentage(value):
    """Format a value as percentage"""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════╗
    ║     Smart CLI Trading Assistant      ║
    ║        ML-Powered Stock Analysis     ║
    ╚══════════════════════════════════════╝
    """
    print(banner)

def validate_symbol(symbol):
    """Basic symbol validation"""
    if not symbol or len(symbol) > 10:
        return False
    return symbol.replace('.', '').replace('-', '').isalpha()

def calculate_support_resistance(data, window=10):
    """Calculate basic support and resistance levels"""
    highs = data['High'].rolling(window=window).max()
    lows = data['Low'].rolling(window=window).min()
    
    resistance = highs.tail(5).max()
    support = lows.tail(5).min()
    
    return support, resistance