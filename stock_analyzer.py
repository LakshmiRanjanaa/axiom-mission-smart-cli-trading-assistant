"""
Stock Analysis Module
Handles data fetching, ML model training, and insight generation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class StockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.scaler = StandardScaler()
        self.model = LinearRegression()
    
    def fetch_data(self, days=30):
        """Fetch stock data from yfinance API"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10)  # Extra days for better analysis
            
            # Fetch data
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            # Take only the requested number of days
            data = data.tail(days)
            
            # Add technical indicators
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_10'] = data['Close'].rolling(window=10).mean()
            data['Price_Change'] = data['Close'].pct_change()
            data['Volume_Change'] = data['Volume'].pct_change()
            
            # Fill NaN values
            data = data.fillna(method='bfill')
            
            return data
            
        except Exception as e:
            raise Exception(f"Failed to fetch data for {self.symbol}: {str(e)}")
    
    def prepare_features(self, data):
        """Prepare features for ML model"""
        # Create feature matrix
        features = []
        
        for i in range(5, len(data)):  # Skip first 5 days due to rolling averages
            feature_row = [
                data['Close'].iloc[i-1],  # Previous close
                data['SMA_5'].iloc[i-1],  # 5-day moving average
                data['SMA_10'].iloc[i-1], # 10-day moving average
                data['Volume'].iloc[i-1], # Previous volume
                data['Price_Change'].iloc[i-1], # Previous price change
                data['Volume_Change'].iloc[i-1], # Previous volume change
            ]
            features.append(feature_row)
        
        # Target: next day price change (positive = up, negative = down)
        targets = []
        for i in range(5, len(data)):
            if i < len(data) - 1:
                price_change = (data['Close'].iloc[i+1] - data['Close'].iloc[i]) / data['Close'].iloc[i]
                targets.append(price_change)
        
        return np.array(features[:-1]), np.array(targets)  # Remove last feature (no target)
    
    def train_and_predict(self, data):
        """Train ML model and make prediction"""
        try:
            X, y = self.prepare_features(data)
            
            if len(X) < 5:
                raise ValueError("Not enough data points for analysis")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Make prediction for next day
            last_features = X[-1:]
            last_scaled = self.scaler.transform(last_features)
            prediction = self.model.predict(last_scaled)[0]
            
            # Calculate confidence based on model score
            confidence = abs(self.model.score(X_scaled, y)) * 100
            confidence = min(confidence, 95)  # Cap at 95%
            
            return prediction, confidence
            
        except Exception as e:
            raise Exception(f"ML model training failed: {str(e)}")
    
    def generate_insights(self, data, prediction):
        """Generate actionable insights based on analysis"""
        insights = []
        
        # Price trend insight
        recent_trend = data['Close'].tail(7).pct_change().mean()
        if recent_trend > 0.01:
            insights.append("Strong upward momentum in recent days")
        elif recent_trend < -0.01:
            insights.append("Downward pressure in recent trading")
        else:
            insights.append("Price consolidation phase")
        
        # Volume analysis
        avg_volume = data['Volume'].mean()
        recent_volume = data['Volume'].tail(3).mean()
        
        if recent_volume > avg_volume * 1.2:
            insights.append("Higher than average trading volume")
        elif recent_volume < avg_volume * 0.8:
            insights.append("Lower than average trading volume")
        
        # Volatility insight
        volatility = data['Close'].std() / data['Close'].mean()
        if volatility > 0.03:
            insights.append("High volatility - consider risk management")
        elif volatility < 0.01:
            insights.append("Low volatility - stable price action")
        
        # Trading recommendation
        if prediction > 0.02:
            insights.append("🟢 Recommendation: CONSIDER BUYING")
        elif prediction < -0.02:
            insights.append("🔴 Recommendation: CONSIDER SELLING")
        else:
            insights.append("🟡 Recommendation: HOLD POSITION")
        
        return insights