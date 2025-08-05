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
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Please install required packages:")
    print("pip install yfinance scikit-learn pandas numpy requests tensorflow")
    sys.exit(1)

# LSTM functionality disabled due to compatibility issues
# Will use enhanced traditional ML models instead
LSTM_AVAILABLE = False

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
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

class StockPredictor:
    """Main class for AMD stock prediction system"""
    
    def __init__(self, symbol: str = "AMD", refresh_interval: int = 60):
        self.symbol = symbol
        self.refresh_interval = refresh_interval  # seconds (default 1 minute)
        self.running = True
        self.eodhd_api_key = os.getenv("EODHD_API_KEY", "demo")
        self.historical_data = []
        
        # Multiple models for ensemble predictions
        self.scaler = MinMaxScaler()
        self.linear_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lstm_model = None
        self.model_trained = False
        self.lstm_trained = False
        
        # Trading system parameters
        self.risk_tolerance = 0.02  # 2% risk per trade
        self.reward_ratio = 2.0     # 2:1 reward-to-risk ratio
        
        # Performance tracking
        self.prediction_history = []
        self.accuracy_score = 0.0
        
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
            
            # Get most recent 1-minute data for real-time prices
            hist_1m = ticker.history(period="1d", interval="1m")
            # Get daily data for previous close
            hist_daily = ticker.history(period="2d", interval="1d")
            
            if hist_1m.empty:
                raise Exception("No minute data returned from Yahoo Finance")
                
            # Use latest 1-minute data for current price
            current_price = float(hist_1m['Close'].iloc[-1])
            previous_close = float(hist_daily['Close'].iloc[-2]) if len(hist_daily) > 1 else float(hist_1m['Close'].iloc[0])
            
            # Today's high/low from minute data
            day_high = float(hist_1m['High'].max())
            day_low = float(hist_1m['Low'].min())
            volume = int(hist_1m['Volume'].sum())
            
            # Calculate intraday changes from minute data
            price_change_15m = self._calculate_price_change(hist_1m, 15)
            price_change_30m = self._calculate_price_change(hist_1m, 30)
            price_change_1h = self._calculate_price_change(hist_1m, 60)
            
            # Get historical data for indicators (daily for SMA/RSI)
            hist_daily_long = ticker.history(period="60d", interval="1d")
            close_prices = hist_daily_long['Close'] if not hist_daily_long.empty else pd.Series([current_price])
            sma_20 = self._calculate_sma(close_prices, 20)
            rsi_14 = self._calculate_rsi(close_prices, 14)
            
            return StockData(
                symbol=self.symbol,
                current_price=current_price,
                previous_close=previous_close,
                day_high=day_high,
                day_low=day_low,
                volume=volume,
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
            return float(rsi.iloc[-1]) if hasattr(rsi.iloc[-1], '__float__') else float(rsi.tail(1).values[0])
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
        
    def create_lstm_model(self, input_shape):
        """Create LSTM model architecture"""
        if not LSTM_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error',
                     metrics=['mae'])
        return model
    
    def prepare_lstm_data(self, historical_data: List[StockData], lookback=10):
        """Prepare data for LSTM training"""
        if len(historical_data) < lookback + 5:
            return None, None
            
        # Create feature matrix
        features = []
        for data in historical_data:
            feature_vector = [
                data.current_price,
                data.previous_close,
                (data.current_price - data.previous_close) / data.previous_close * 100,
                data.sma_20,
                data.rsi_14,
                data.volume / 1000000,
                (data.day_high - data.day_low) / data.current_price * 100,
                data.price_change_15m,
                data.price_change_30m,
                data.price_change_1h
            ]
            features.append(feature_vector)
            
        features = np.array(features)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(lookback, len(features)):
            X.append(features[i-lookback:i])
            # Target: next price change percentage
            current_price = historical_data[i-1].current_price
            next_price = historical_data[i].current_price
            price_change = (next_price - current_price) / current_price * 100
            y.append(price_change)
            
        return np.array(X), np.array(y)
    
    def train_models(self, historical_data: List[StockData]) -> bool:
        """Train multiple ML models with historical data"""
        try:
            if len(historical_data) < 15:
                return False
                
            # Prepare traditional ML features
            features = []
            targets = []
            
            for i in range(len(historical_data) - 1):
                current_data = historical_data[i]
                next_data = historical_data[i + 1]
                
                # Enhanced features
                feature_vector = [
                    current_data.current_price,
                    current_data.previous_close,
                    (current_data.current_price - current_data.previous_close) / current_data.previous_close * 100,
                    current_data.sma_20,
                    current_data.rsi_14,
                    current_data.volume / 1000000,
                    (current_data.day_high - current_data.day_low) / current_data.current_price * 100,
                    current_data.price_change_15m,
                    current_data.price_change_30m,
                    current_data.price_change_1h,
                    # Price relative to SMA
                    (current_data.current_price - current_data.sma_20) / current_data.sma_20 * 100,
                    # Volatility indicator
                    abs(current_data.price_change_15m) + abs(current_data.price_change_30m) + abs(current_data.price_change_1h)
                ]
                
                # Calculate target (future price change)
                price_change = (next_data.current_price - current_data.current_price) / current_data.current_price * 100
                
                features.append(feature_vector)
                targets.append(price_change)
                
            if len(features) < 10:
                return False
                
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train traditional models
            self.linear_model.fit(X_scaled, y)
            self.rf_model.fit(X_scaled, y)
            
            # Train LSTM if available and enough data
            if LSTM_AVAILABLE and len(historical_data) >= 25:
                X_lstm, y_lstm = self.prepare_lstm_data(historical_data)
                if X_lstm is not None and len(X_lstm) > 10:
                    # Scale LSTM data
                    original_shape = X_lstm.shape
                    X_lstm_reshaped = X_lstm.reshape(-1, X_lstm.shape[-1])
                    X_lstm_scaled = self.scaler.fit_transform(X_lstm_reshaped)
                    X_lstm_scaled = X_lstm_scaled.reshape(original_shape)
                    
                    # Create and train LSTM
                    self.lstm_model = self.create_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
                    if self.lstm_model:
                        self.lstm_model.fit(X_lstm_scaled, y_lstm, 
                                          epochs=50, batch_size=8, verbose=0,
                                          validation_split=0.2)
                        self.lstm_trained = True
            
            self.model_trained = True
            return True
            
        except Exception as e:
            print(f"❌ Model training error: {e}")
            return False
            
    def get_ensemble_prediction(self, stock_data: StockData) -> Tuple[float, float]:
        """Get ensemble prediction from multiple models"""
        predictions = []
        confidences = []
        
        if self.model_trained:
            features = self.prepare_features(stock_data)
            features_scaled = self.scaler.transform(features)
            
            # Linear regression prediction
            linear_pred = self.linear_model.predict(features_scaled)[0]
            predictions.append(linear_pred)
            confidences.append(0.3)  # Weight
            
            # Random Forest prediction
            rf_pred = self.rf_model.predict(features_scaled)[0]
            predictions.append(rf_pred)
            confidences.append(0.4)  # Weight
            
            # LSTM prediction if available
            if self.lstm_trained and self.lstm_model and len(self.historical_data) >= 10:
                try:
                    # Prepare LSTM input
                    lstm_features = []
                    for data in self.historical_data[-10:]:
                        feature_vector = [
                            data.current_price, data.previous_close,
                            (data.current_price - data.previous_close) / data.previous_close * 100,
                            data.sma_20, data.rsi_14, data.volume / 1000000,
                            (data.day_high - data.day_low) / data.current_price * 100,
                            data.price_change_15m, data.price_change_30m, data.price_change_1h
                        ]
                        lstm_features.append(feature_vector)
                    
                    lstm_input = np.array(lstm_features).reshape(1, 10, 10)
                    lstm_input_scaled = self.scaler.transform(lstm_input.reshape(-1, 10)).reshape(1, 10, 10)
                    lstm_pred = self.lstm_model.predict(lstm_input_scaled, verbose=0)[0][0]
                    predictions.append(lstm_pred)
                    confidences.append(0.3)  # Weight
                except:
                    pass
        
        if predictions:
            # Weighted average
            ensemble_pred = np.average(predictions, weights=confidences[:len(predictions)])
            ensemble_conf = np.mean(confidences[:len(predictions)]) * 100
            return ensemble_pred, min(ensemble_conf + abs(ensemble_pred) * 15, 95)
        
        return 0.0, 50.0

    def calculate_risk_management(self, stock_data: StockData, predicted_change: float, signal: str) -> Tuple[float, float, float]:
        """Calculate stop loss, take profit, and risk-reward ratio"""
        current_price = stock_data.current_price
        
        if signal == "BUY":
            stop_loss = current_price * (1 - self.risk_tolerance)
            take_profit = current_price * (1 + self.risk_tolerance * self.reward_ratio)
        elif signal == "SELL":
            stop_loss = current_price * (1 + self.risk_tolerance)
            take_profit = current_price * (1 - self.risk_tolerance * self.reward_ratio)
        else:
            return None, None, None
            
        risk_amount = abs(current_price - stop_loss)
        reward_amount = abs(take_profit - current_price)
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return stop_loss, take_profit, risk_reward_ratio

    def predict_price_movement(self, stock_data: StockData) -> Prediction:
        """Advanced prediction using ensemble models and risk management"""
        try:
            # Technical analysis indicators
            rsi_signal = "WAIT"
            if stock_data.rsi_14 > 70:
                rsi_signal = "SELL"
            elif stock_data.rsi_14 < 30:
                rsi_signal = "BUY"
                
            sma_trend = "STABLE"
            if stock_data.current_price > stock_data.sma_20 * 1.015:
                sma_trend = "UP"
            elif stock_data.current_price < stock_data.sma_20 * 0.985:
                sma_trend = "DOWN"
            
            momentum_score = (stock_data.price_change_15m + stock_data.price_change_30m + stock_data.price_change_1h) / 3
            
            # Get ensemble prediction
            predicted_change, base_confidence = self.get_ensemble_prediction(stock_data)
            
            # Enhanced decision logic
            if self.model_trained and abs(predicted_change) > 0.3:
                # Use ML prediction with technical confirmation
                if predicted_change > 0.8:
                    direction = "UP"
                    signal = "BUY" if rsi_signal != "SELL" and momentum_score > -0.3 else "WAIT"
                elif predicted_change < -0.8:
                    direction = "DOWN"  
                    signal = "SELL" if rsi_signal != "BUY" and momentum_score < 0.3 else "WAIT"
                else:
                    direction = "STABLE"
                    signal = "WAIT"
                    
                confidence = min(base_confidence * (1 + abs(predicted_change) / 5), 95)
                price_target = stock_data.current_price * (1 + predicted_change / 100)
                
            else:
                # Enhanced technical analysis with lower thresholds for more active signals
                strong_momentum = abs(momentum_score) > 0.3
                price_vs_sma = (stock_data.current_price - stock_data.sma_20) / stock_data.sma_20 * 100
                
                # More sensitive prediction logic for 30-40 cent movements
                if momentum_score > 0.3 and price_vs_sma > 1.0:  # Strong bullish
                    direction = "UP"
                    signal = "BUY" if stock_data.rsi_14 < 75 else "WAIT"
                    confidence = min(70.0 + abs(momentum_score) * 15, 90.0)
                elif momentum_score < -0.3 and price_vs_sma > 1.0:  # Bearish with high price
                    direction = "DOWN"
                    signal = "SELL" if stock_data.rsi_14 > 25 else "WAIT"
                    confidence = min(70.0 + abs(momentum_score) * 15, 90.0)
                elif momentum_score > 0.15:  # Moderate bullish momentum
                    direction = "UP"
                    signal = "BUY" if stock_data.rsi_14 < 70 and price_vs_sma > 0 else "WAIT"
                    confidence = min(60.0 + abs(momentum_score) * 20, 80.0)
                elif momentum_score < -0.15:  # Moderate bearish momentum
                    direction = "DOWN"
                    signal = "SELL" if stock_data.rsi_14 > 30 and price_vs_sma > 0 else "WAIT"
                    confidence = min(60.0 + abs(momentum_score) * 20, 80.0)
                elif abs(stock_data.price_change_15m) > 0.2:  # Recent significant movement
                    direction = "UP" if stock_data.price_change_15m > 0 else "DOWN"
                    signal = "BUY" if direction == "UP" and stock_data.rsi_14 < 65 else "SELL" if direction == "DOWN" and stock_data.rsi_14 > 35 else "WAIT"
                    confidence = min(55.0 + abs(stock_data.price_change_15m) * 10, 75.0)
                else:
                    direction = "STABLE"
                    signal = "WAIT"
                    confidence = 50.0
                    
                price_target = None
                
            # Calculate risk management
            stop_loss, take_profit, risk_reward = self.calculate_risk_management(
                stock_data, predicted_change, signal
            )
            
            prediction = Prediction(
                direction=direction,
                confidence=confidence,
                signal=signal,
                price_target=price_target,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward
            )
            
            # Track prediction for accuracy calculation
            self.prediction_history.append({
                'timestamp': stock_data.timestamp,
                'prediction': predicted_change,
                'actual_price': stock_data.current_price,
                'signal': signal
            })
            
            return prediction
            
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
        
        # Momentum analysis
        momentum = (stock_data.price_change_15m + stock_data.price_change_30m + stock_data.price_change_1h) / 3
        momentum_status = "🚀 Strong Up" if momentum > 0.5 else "📉 Strong Down" if momentum < -0.5 else "➡️ Neutral"
        print(f"   Momentum Score:    {momentum:+.2f}% ({momentum_status})")
        
        # Price vs SMA analysis
        price_vs_sma = (stock_data.current_price - stock_data.sma_20) / stock_data.sma_20 * 100
        print(f"   Price vs SMA-20:   {price_vs_sma:+.1f}% ({price_vs_sma > 0 and '🟢 Above' or '🔴 Below'})")
        
        # Prediction Section
        direction_emoji = {"UP": "🚀", "DOWN": "📉", "STABLE": "➡️"}
        signal_emoji = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "WAIT": "🟡 WAIT"}
        
        print(f"\n🤖 AI PREDICTION & TRADING SIGNALS:")
        print(f"   Direction:         {direction_emoji.get(prediction.direction, '❓')} {prediction.direction}")
        print(f"   Confidence:        {prediction.confidence:.1f}%")
        print(f"   Trading Signal:    {signal_emoji.get(prediction.signal, '❓')}")
        
        if prediction.price_target:
            target_change = ((prediction.price_target - stock_data.current_price) / stock_data.current_price) * 100
            print(f"   Price Target:      ${prediction.price_target:.2f} ({target_change:+.1f}%)")
            
        # Risk Management
        if prediction.stop_loss and prediction.take_profit:
            print(f"\n💰 RISK MANAGEMENT:")
            print(f"   Stop Loss:         ${prediction.stop_loss:.2f}")
            print(f"   Take Profit:       ${prediction.take_profit:.2f}")
            if prediction.risk_reward_ratio:
                print(f"   Risk/Reward:       1:{prediction.risk_reward_ratio:.1f}")
                
        # Model Status
        models_active = []
        if self.model_trained:
            models_active.append("Linear+RF")
        if self.lstm_trained:
            models_active.append("LSTM")
            
        if models_active:
            model_status = f"✅ ACTIVE: {'+'.join(models_active)}"
        else:
            model_status = "⚠️  LEARNING"
        print(f"\n   Model Status:      {model_status}")
        
        # Accuracy tracking
        if len(self.prediction_history) > 5:
            print(f"   Prediction History: {len(self.prediction_history)} signals tracked")
        
        if self.refresh_interval >= 60:
            print(f"\n🔄 Next update in {self.refresh_interval//60} minutes...")
        else:
            print(f"\n🔄 Next update in {self.refresh_interval} seconds...")
        print(f"📝 Historical data points: {len(self.historical_data)}")
        print("\n💡 Press Ctrl+C to stop")
        print("=" * 80)
        
    def run(self):
        """Main execution loop"""
        print("🚀 Starting AMD Stock Prediction System...")
        if self.refresh_interval >= 60:
            print(f"📡 Refresh interval: {self.refresh_interval//60} minutes")
        else:
            print(f"📡 Refresh interval: {self.refresh_interval} seconds")
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
                    
                # Train models if we have enough data
                if len(self.historical_data) >= 15 and not self.model_trained:
                    print("🧠 Training machine learning models...")
                    self.train_models(self.historical_data)
                elif len(self.historical_data) >= 30 and len(self.historical_data) % 10 == 0:
                    # Retrain periodically with new data
                    print("🔄 Retraining models with new data...")
                    self.train_models(self.historical_data)
                    
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
    REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "60"))  # 1 minute default
    
    print("🎯 AMD Stock Prediction System")
    print("===============================")
    print(f"📈 Target Stock: {SYMBOL}")
    if REFRESH_INTERVAL >= 60:
        print(f"⏰ Refresh Interval: {REFRESH_INTERVAL//60} minutes")
    else:
        print(f"⏰ Refresh Interval: {REFRESH_INTERVAL} seconds")
    
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
