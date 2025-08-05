#!/usr/bin/env python3
"""
AMD Stock Prediction System
A terminal-based real-time stock analysis and prediction tool using machine learning.
"""

import os
import sys
import time
import signal
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import yfinance as yf
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Please install required packages:")
    print("pip install yfinance scikit-learn pandas numpy requests")
    sys.exit(1)

@dataclass
class StockData:
    """Data structure for stock information"""
    symbol: str
    current_price: float
    previous_close: float
    day_high: float
    day_low: float
    volume: int
    price_change_15m: float
    price_change_30m: float
    price_change_1h: float
    sma_20: float
    rsi_14: float
    timestamp: datetime

@dataclass
class Prediction:
    """Data structure for prediction results"""
    direction: str  # UP, DOWN, STABLE
    confidence: float
    signal: str  # BUY, SELL, WAIT
    price_target: Optional[float] = None

class StockPredictor:
    """Main class for AMD stock prediction system"""
    
    def __init__(self, symbol: str = "AMD", refresh_interval: int = 1800):
        self.symbol = symbol
        self.refresh_interval = refresh_interval  # seconds (default 30 minutes)
        self.running = True
        self.eodhd_api_key = os.getenv("EODHD_API_KEY", "demo")
        self.historical_data = []
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.model_trained = False
        
        # Setup signal handler for graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n🛑 Stopping stock predictor...")
        self.running = False
        
    def fetch_yahoo_data(self) -> Optional[StockData]:
        """Fetch real-time data from Yahoo Finance using yfinance"""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Get current data
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                raise Exception("No data returned from Yahoo Finance")
                
            current_price = hist['Close'].iloc[-1]
            previous_close = info.get('previousClose', hist['Close'].iloc[0])
            day_high = hist['High'].max()
            day_low = hist['Low'].min()
            volume = hist['Volume'].sum()
            
            # Calculate intraday changes
            price_change_15m = self._calculate_price_change(hist, 15)
            price_change_30m = self._calculate_price_change(hist, 30)
            price_change_1h = self._calculate_price_change(hist, 60)
            
            # Get historical data for indicators
            hist_data = ticker.history(period="60d", interval="1d")
            sma_20 = self._calculate_sma(hist_data['Close'], 20)
            rsi_14 = self._calculate_rsi(hist_data['Close'], 14)
            
            return StockData(
                symbol=self.symbol,
                current_price=float(current_price),
                previous_close=float(previous_close),
                day_high=float(day_high),
                day_low=float(day_low),
                volume=int(volume),
                price_change_15m=price_change_15m,
                price_change_30m=price_change_30m,
                price_change_1h=price_change_1h,
                sma_20=sma_20,
                rsi_14=rsi_14,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"❌ Yahoo Finance API error: {e}")
            return None
            
    def fetch_eodhd_data(self) -> Optional[StockData]:
        """Fetch data from EODHD API as backup"""
        try:
            # Real-time price
            url = f"https://eodhistoricaldata.com/api/real-time/{self.symbol}.US"
            params = {"api_token": self.eodhd_api_key, "fmt": "json"}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'code' in data and data['code'] == 401:
                raise Exception("Invalid EODHD API key")
                
            current_price = float(data.get('close', 0))
            previous_close = float(data.get('previousClose', 0))
            day_high = float(data.get('high', 0))
            day_low = float(data.get('low', 0))
            volume = int(data.get('volume', 0))
            
            # Get historical data for technical indicators
            hist_url = f"https://eodhistoricaldata.com/api/eod/{self.symbol}.US"
            hist_params = {
                "api_token": self.eodhd_api_key,
                "period": "d",
                "from": (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            }
            
            hist_response = requests.get(hist_url, params=hist_params, timeout=10)
            hist_response.raise_for_status()
            hist_data = hist_response.json()
            
            if hist_data:
                closes = [float(d['close']) for d in hist_data[-60:]]
                sma_20 = self._calculate_sma(pd.Series(closes), 20)
                rsi_14 = self._calculate_rsi(pd.Series(closes), 14)
            else:
                sma_20 = current_price
                rsi_14 = 50.0
            
            return StockData(
                symbol=self.symbol,
                current_price=current_price,
                previous_close=previous_close,
                day_high=day_high,
                day_low=day_low,
                volume=volume,
                price_change_15m=0.0,  # EODHD doesn't provide intraday intervals easily
                price_change_30m=0.0,
                price_change_1h=0.0,
                sma_20=sma_20,
                rsi_14=rsi_14,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"❌ EODHD API error: {e}")
            return None
            
    def _calculate_price_change(self, hist_data: pd.DataFrame, minutes: int) -> float:
        """Calculate price change over specified minutes"""
        try:
            if len(hist_data) < minutes:
                return 0.0
            current_price = hist_data['Close'].iloc[-1]
            past_price = hist_data['Close'].iloc[-minutes]
            return float(((current_price - past_price) / past_price) * 100)
        except:
            return 0.0
            
    def _calculate_sma(self, prices: pd.Series, period: int) -> float:
        """Calculate Simple Moving Average"""
        try:
            if len(prices) < period:
                return float(prices.mean())
            return float(prices.tail(period).mean())
        except:
            return 0.0
            
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            if len(prices) < period + 1:
                return 50.0
                
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0
            
    def prepare_features(self, stock_data: StockData) -> np.ndarray:
        """Prepare features for machine learning model"""
        features = [
            stock_data.current_price,
            stock_data.previous_close,
            (stock_data.current_price - stock_data.previous_close) / stock_data.previous_close * 100,
            stock_data.sma_20,
            stock_data.rsi_14,
            stock_data.volume / 1000000,  # Volume in millions
            (stock_data.day_high - stock_data.day_low) / stock_data.current_price * 100,  # Daily volatility
        ]
        return np.array(features).reshape(1, -1)
        
    def train_model(self, historical_data: List[StockData]) -> bool:
        """Train the machine learning model with historical data"""
        try:
            if len(historical_data) < 10:
                return False
                
            features = []
            targets = []
            
            for i in range(len(historical_data) - 1):
                current_data = historical_data[i]
                next_data = historical_data[i + 1]
                
                # Prepare features
                feature_vector = [
                    current_data.current_price,
                    current_data.previous_close,
                    (current_data.current_price - current_data.previous_close) / current_data.previous_close * 100,
                    current_data.sma_20,
                    current_data.rsi_14,
                    current_data.volume / 1000000,
                    (current_data.day_high - current_data.day_low) / current_data.current_price * 100,
                ]
                
                # Calculate target (future price change)
                price_change = (next_data.current_price - current_data.current_price) / current_data.current_price * 100
                
                features.append(feature_vector)
                targets.append(price_change)
                
            if len(features) < 5:
                return False
                
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.model_trained = True
            
            return True
            
        except Exception as e:
            print(f"❌ Model training error: {e}")
            return False
            
    def predict_price_movement(self, stock_data: StockData) -> Prediction:
        """Predict price movement using ML model and technical analysis"""
        try:
            # Basic technical analysis fallback
            rsi_signal = "WAIT"
            if stock_data.rsi_14 > 70:
                rsi_signal = "SELL"
            elif stock_data.rsi_14 < 30:
                rsi_signal = "BUY"
                
            sma_trend = "STABLE"
            if stock_data.current_price > stock_data.sma_20 * 1.02:
                sma_trend = "UP"
            elif stock_data.current_price < stock_data.sma_20 * 0.98:
                sma_trend = "DOWN"
                
            # If model is trained, use ML prediction
            if self.model_trained:
                features = self.prepare_features(stock_data)
                features_scaled = self.scaler.transform(features)
                predicted_change = self.model.predict(features_scaled)[0]
                
                # Calculate confidence based on model certainty and technical indicators
                confidence = min(abs(predicted_change) * 10 + 50, 95)
                
                if predicted_change > 1.0:
                    direction = "UP"
                    signal = "BUY" if rsi_signal != "SELL" else "WAIT"
                elif predicted_change < -1.0:
                    direction = "DOWN"
                    signal = "SELL" if rsi_signal != "BUY" else "WAIT"
                else:
                    direction = "STABLE"
                    signal = "WAIT"
                    
                price_target = stock_data.current_price * (1 + predicted_change / 100)
                
            else:
                # Fallback to technical analysis only
                if sma_trend == "UP" and stock_data.rsi_14 < 70:
                    direction = "UP"
                    signal = "BUY"
                    confidence = 60.0
                elif sma_trend == "DOWN" and stock_data.rsi_14 > 30:
                    direction = "DOWN"
                    signal = "SELL"
                    confidence = 60.0
                else:
                    direction = "STABLE"
                    signal = "WAIT"
                    confidence = 50.0
                    
                price_target = None
                
            return Prediction(
                direction=direction,
                confidence=confidence,
                signal=signal,
                price_target=price_target
            )
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return Prediction(direction="STABLE", confidence=50.0, signal="WAIT")
            
    def display_data(self, stock_data: StockData, prediction: Prediction):
        """Display formatted stock data and prediction in terminal"""
        # Clear screen and show header
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print(f"📊 AMD STOCK PREDICTION SYSTEM - {stock_data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Current Price Section
        price_change = stock_data.current_price - stock_data.previous_close
        price_change_pct = (price_change / stock_data.previous_close) * 100
        
        change_color = "🟢" if price_change >= 0 else "🔴"
        change_symbol = "+" if price_change >= 0 else ""
        
        print(f"\n💰 CURRENT PRICE DATA:")
        print(f"   Current Price:     ${stock_data.current_price:.2f}")
        print(f"   Previous Close:    ${stock_data.previous_close:.2f}")
        print(f"   Daily Change:      {change_color} {change_symbol}{price_change:.2f} ({change_symbol}{price_change_pct:.2f}%)")
        print(f"   Day High:          ${stock_data.day_high:.2f}")
        print(f"   Day Low:           ${stock_data.day_low:.2f}")
        print(f"   Volume:            {stock_data.volume:,}")
        
        # Intraday Changes
        print(f"\n⏱️  INTRADAY CHANGES:")
        print(f"   15-minute:         {stock_data.price_change_15m:+.2f}%")
        print(f"   30-minute:         {stock_data.price_change_30m:+.2f}%")
        print(f"   1-hour:            {stock_data.price_change_1h:+.2f}%")
        
        # Technical Indicators
        print(f"\n📈 TECHNICAL INDICATORS:")
        print(f"   SMA (20):          ${stock_data.sma_20:.2f}")
        print(f"   RSI (14):          {stock_data.rsi_14:.1f}")
        
        # RSI interpretation
        if stock_data.rsi_14 > 70:
            rsi_status = "🔴 Overbought"
        elif stock_data.rsi_14 < 30:
            rsi_status = "🟢 Oversold"
        else:
            rsi_status = "🟡 Neutral"
        print(f"   RSI Status:        {rsi_status}")
        
        # Prediction Section
        direction_emoji = {"UP": "🚀", "DOWN": "📉", "STABLE": "➡️"}
        signal_emoji = {"BUY": "🟢", "SELL": "🔴", "WAIT": "🟡"}
        
        print(f"\n🤖 AI PREDICTION:")
        print(f"   Direction:         {direction_emoji.get(prediction.direction, '❓')} {prediction.direction}")
        print(f"   Confidence:        {prediction.confidence:.1f}%")
        print(f"   Trading Signal:    {signal_emoji.get(prediction.signal, '❓')} {prediction.signal}")
        
        if prediction.price_target:
            print(f"   Price Target:      ${prediction.price_target:.2f}")
            
        # Model Status
        model_status = "✅ TRAINED" if self.model_trained else "⚠️  LEARNING"
        print(f"   Model Status:      {model_status}")
        
        print(f"\n🔄 Next update in {self.refresh_interval//60} minutes...")
        print(f"📝 Historical data points: {len(self.historical_data)}")
        print("\n💡 Press Ctrl+C to stop")
        print("=" * 80)
        
    def run(self):
        """Main execution loop"""
        print("🚀 Starting AMD Stock Prediction System...")
        print(f"📡 Refresh interval: {self.refresh_interval//60} minutes")
        print("🔧 Loading initial data...\n")
        
        while self.running:
            try:
                # Fetch stock data (try Yahoo Finance first, then EODHD)
                stock_data = self.fetch_yahoo_data()
                if not stock_data:
                    print("🔄 Trying EODHD API...")
                    stock_data = self.fetch_eodhd_data()
                    
                if not stock_data:
                    print("❌ Failed to fetch data from all sources. Retrying in 60 seconds...")
                    time.sleep(60)
                    continue
                    
                # Add to historical data
                self.historical_data.append(stock_data)
                
                # Keep only last 100 data points for efficiency
                if len(self.historical_data) > 100:
                    self.historical_data = self.historical_data[-100:]
                    
                # Train model if we have enough data
                if len(self.historical_data) >= 10 and not self.model_trained:
                    print("🧠 Training machine learning model...")
                    self.train_model(self.historical_data)
                elif len(self.historical_data) >= 20:
                    # Retrain periodically with new data
                    self.train_model(self.historical_data)
                    
                # Make prediction
                prediction = self.predict_price_movement(stock_data)
                
                # Display results
                self.display_data(stock_data, prediction)
                
                # Wait for next update
                if self.running:
                    time.sleep(self.refresh_interval)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                print("🔄 Retrying in 60 seconds...")
                time.sleep(60)
                
        print("\n✅ Stock predictor stopped gracefully.")

def main():
    """Main entry point"""
    # Configuration
    SYMBOL = "AMD"
    REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "1800"))  # 30 minutes default
    
    print("🎯 AMD Stock Prediction System")
    print("===============================")
    print(f"📈 Target Stock: {SYMBOL}")
    print(f"⏰ Refresh Interval: {REFRESH_INTERVAL//60} minutes")
    
    # Check for API keys
    eodhd_key = os.getenv("EODHD_API_KEY", "demo")
    if eodhd_key == "demo":
        print("⚠️  Using demo EODHD API key (limited functionality)")
        print("💡 Set EODHD_API_KEY environment variable for full access")
    else:
        print("✅ EODHD API key configured")
        
    print("\n🚀 Initializing system...")
    
    try:
        predictor = StockPredictor(symbol=SYMBOL, refresh_interval=REFRESH_INTERVAL)
        predictor.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
