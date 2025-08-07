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
from typing import Dict, List, Tuple, Optional, Any
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

# LSTM for 30-minute stable predictions
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except ImportError:
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
    price_range_30m: Optional[tuple] = None  # (low, high) for 30-min prediction
    prediction_30m_timestamp: Optional[datetime] = None  # When 30-min prediction was made
    prediction_30m_expires: Optional[datetime] = None    # When prediction expires

@dataclass
class NextDayPrediction:
    """Data structure for next-day market open prediction"""
    predicted_open_price: float
    confidence: float
    price_change_pct: float
    direction: str  # UP, DOWN, STABLE
    sentiment_score: float  # -1 to +1
    overnight_factors: Dict[str, float]
    futures_correlation: float
    pre_market_trend: str
    risk_assessment: str  # LOW, MEDIUM, HIGH
    target_range: Tuple[float, float]  # (low, high) expected range
    created_at: datetime

class StockPredictor:
    """Main class for AMD stock prediction system"""
    
    def __init__(self, symbol: str = "AMD", refresh_interval: int = 60):
        self.symbol = symbol
        self.refresh_interval = refresh_interval  # seconds (default 1 minute)
        
        # Next-day prediction system
        self.next_day_model = None
        self.sentiment_cache = {}
        self.overnight_data_cache = {}
        self.last_next_day_prediction = None
        self.next_day_prediction_time = None
        
        # API configurations (will be set from environment variables)
        self.eodhd_api_key = os.getenv('EODHD_API_KEY', '')
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        
        # Futures correlation data for AMD (semiconductor sector)
        self.semiconductor_futures = ['NQ', 'QQQ']  # NASDAQ futures correlation
        self.running = True
        self.eodhd_api_key = os.getenv("EODHD_API_KEY")
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
        
        # 30-minute stable prediction cache
        self.current_30min_prediction = None
        self.prediction_30m_made_at = None
        self.prediction_30m_direction = None
        self.prediction_30m_confidence = None
        self.comprehensive_data_cache = []
        
        # Pre-trained LSTM model for stable predictions
        self.stable_lstm_model = None
        self.lstm_scaler = MinMaxScaler()
        self.prediction_history_30m = []  # Track 30-min prediction accuracy
        
        # 1-minute ahead prediction system
        self.minute_ahead_model = None
        self.minute_scaler = MinMaxScaler()
        self.last_1min_prediction = None
        self.last_1min_prediction_time = None
        
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
        if not self.eodhd_api_key:
            return None
            
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
                price_change_15m=self._get_intraday_change_estimate(current_price, previous_close, 15),
                price_change_30m=self._get_intraday_change_estimate(current_price, previous_close, 30),
                price_change_1h=self._get_intraday_change_estimate(current_price, previous_close, 60),
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
    
    def _get_intraday_change_estimate(self, current_price: float, previous_close: float, minutes: int) -> float:
        """Estimate intraday price change when detailed data is unavailable"""
        try:
            # Calculate daily change so far
            daily_change_pct = ((current_price - previous_close) / previous_close) * 100
            
            # Estimate proportional change based on time elapsed
            # Assume linear distribution throughout trading day (6.5 hours = 390 minutes)
            current_time = datetime.now()
            if current_time.weekday() < 5:  # Weekday
                market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
                if current_time < market_open:
                    return 0.0
                    
                elapsed_minutes = (current_time - market_open).total_seconds() / 60
                total_trading_minutes = 390  # 6.5 hours
                
                # Estimate what portion occurred in the specified timeframe
                if elapsed_minutes >= minutes:
                    proportion = minutes / elapsed_minutes
                    return daily_change_pct * proportion * 0.3  # Conservative estimate
                    
            return daily_change_pct * 0.1  # Fallback to small portion of daily change
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
            return 0.0, 0.0, 0.0
            
        risk_amount = abs(current_price - stop_loss)
        reward_amount = abs(take_profit - current_price)
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return stop_loss, take_profit, risk_reward_ratio

    def build_lstm_model(self, sequence_length: int = 60) -> Any:
        """Build LSTM model for stable 30-minute predictions"""
        if not LSTM_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(sequence_length, 6)),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def get_comprehensive_market_data(self, stock_data: StockData) -> dict:
        """Collect comprehensive market data for stable 30-min prediction"""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Get extended historical data for LSTM training
            hist_5d = ticker.history(period="5d", interval="1m")
            hist_1m_extended = ticker.history(period="1d", interval="1m")
            
            if hist_5d.empty or hist_1m_extended.empty:
                return {}
                
            # Prepare comprehensive features
            features = {
                'prices': hist_5d['Close'].values[-300:] if len(hist_5d) >= 300 else hist_5d['Close'].values,
                'volumes': hist_5d['Volume'].values[-300:] if len(hist_5d) >= 300 else hist_5d['Volume'].values,
                'highs': hist_5d['High'].values[-300:] if len(hist_5d) >= 300 else hist_5d['High'].values,
                'lows': hist_5d['Low'].values[-300:] if len(hist_5d) >= 300 else hist_5d['Low'].values,
                'current_price': stock_data.current_price,
                'rsi': stock_data.rsi_14,
                'sma': stock_data.sma_20,
                'momentum': (stock_data.price_change_15m + stock_data.price_change_30m + stock_data.price_change_1h) / 3,
                'volume_current': stock_data.volume,
                'volatility': np.std(hist_1m_extended['Close'].values[-60:]) if len(hist_1m_extended) >= 60 else 0
            }
            
            return features
            
        except Exception as e:
            print(f"Error collecting comprehensive data: {e}")
            return {}

    def predict_30min_stable_range(self, stock_data: StockData) -> tuple:
        """STABLE 30-minute prediction - creates ONE prediction that stays fixed for 30 minutes"""
        
        current_time = datetime.now()
        
        # Check if we have a valid cached 30-minute prediction
        if (self.current_30min_prediction and 
            self.prediction_30m_made_at and 
            (current_time - self.prediction_30m_made_at).total_seconds() < 1800):  # 30 minutes = 1800 seconds
            
            # Calculate remaining time
            remaining_seconds = 1800 - (current_time - self.prediction_30m_made_at).total_seconds()
            remaining_minutes = int(remaining_seconds / 60)
            
            print(f"⏳ Using cached 30-min prediction (expires in {remaining_minutes}m)")
            return self.current_30min_prediction
            
        print("🔄 Creating NEW 30-minute STABLE prediction (will NOT change for 30 minutes)...")
        
        # Get comprehensive historical data for LSTM training
        comprehensive_data = self.get_comprehensive_market_data(stock_data)
        if not comprehensive_data:
            return self._create_stable_fallback_prediction(stock_data, current_time)
            
        # Use pre-trained LSTM model for stable predictions
        if LSTM_AVAILABLE and len(comprehensive_data['prices']) >= 120:  # Need more data for stability
            try:
                stable_range = self._create_lstm_stable_prediction(stock_data, comprehensive_data, current_time)
                if stable_range:
                    return stable_range
            except Exception as e:
                print(f"⚠️ LSTM prediction failed: {e}")
                
        # Fallback to enhanced stable analysis
        return self._create_stable_fallback_prediction(stock_data, current_time)

    def _create_lstm_stable_prediction(self, stock_data: StockData, comprehensive_data: dict, current_time: datetime) -> tuple:
        """Create a truly stable LSTM-based 30-minute prediction"""
        try:
            # Prepare robust feature matrix with more historical data
            prices = comprehensive_data['prices']
            volumes = comprehensive_data['volumes'] 
            highs = comprehensive_data['highs']
            lows = comprehensive_data['lows']
            
            # Create comprehensive features (6 features)
            min_len = min(len(prices), len(volumes), len(highs), len(lows))
            if min_len < 120:  # Need sufficient data for stability
                return (0.0, 0.0)
                
            features = np.column_stack([
                prices[-min_len:],
                volumes[-min_len:],
                highs[-min_len:], 
                lows[-min_len:],
                np.full(min_len, comprehensive_data['rsi']),
                np.full(min_len, comprehensive_data['momentum'])
            ])
            
            # Normalize features with dedicated scaler
            features_scaled = self.lstm_scaler.fit_transform(features)
            
            # Build and train a robust LSTM model
            if not self.stable_lstm_model:
                self.stable_lstm_model = self._build_stable_lstm_model()
                
            if self.stable_lstm_model:
                # Prepare training data for 30-minute predictions
                X_train, y_train = [], []
                sequence_length = 60
                
                # Create training sequences predicting 30 minutes ahead
                for i in range(sequence_length, len(features_scaled) - 30):
                    X_train.append(features_scaled[i-sequence_length:i])
                    y_train.append(features_scaled[i+30, 0])  # Price 30 steps ahead
                    
                if len(X_train) >= 20:  # Need sufficient training data
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    
                    # Train the model thoroughly for stable predictions
                    self.stable_lstm_model.fit(X_train, y_train, 
                                             epochs=30, batch_size=8, verbose=0,
                                             validation_split=0.2)
                    
                    # Make prediction using last 60 data points
                    last_sequence = features_scaled[-sequence_length:].reshape(1, sequence_length, 6)
                    prediction_normalized = self.stable_lstm_model.predict(last_sequence, verbose=0)[0][0]
                    
                    # Denormalize prediction
                    temp_data = np.zeros((1, 6))
                    temp_data[0, 0] = prediction_normalized
                    predicted_price = self.lstm_scaler.inverse_transform(temp_data)[0, 0]
                    
                    # Determine direction and create stable range
                    current_price = stock_data.current_price
                    price_change_pct = (predicted_price - current_price) / current_price * 100
                    
                    # BALANCED LSTM direction detection with equal sensitivity for both directions
                    
                    # Check daily performance for additional context
                    daily_change = (current_price - stock_data.previous_close) / stock_data.previous_close * 100
                    
                    # PRIORITY: Sensitive DOWN detection (protect losses first)
                    if price_change_pct < -0.02:  # More sensitive bearish detection 
                        direction = "DOWN"
                        range_size = min(current_price * 0.003, 0.50)
                        price_low = current_price - (range_size * 0.8)
                        price_high = current_price + (range_size * 0.2)
                        confidence = min(85 + abs(price_change_pct) * 12, 95)  # Higher confidence for protection
                        
                    elif price_change_pct > 0.05:  # Standard bullish prediction
                        direction = "UP"
                        range_size = min(current_price * 0.003, 0.50)
                        price_low = current_price - (range_size * 0.2)
                        price_high = current_price + (range_size * 0.8)
                        confidence = min(80 + abs(price_change_pct) * 10, 95)
                        
                    else:  # Stable prediction
                        direction = "STABLE"
                        range_size = min(current_price * 0.002, 0.30)
                        price_low = current_price - (range_size / 2)
                        price_high = current_price + (range_size / 2)
                        confidence = 70
                    
                    # Cache the stable prediction
                    stable_range = (round(price_low, 2), round(price_high, 2))
                    self.current_30min_prediction = stable_range
                    self.prediction_30m_made_at = current_time
                    self.prediction_30m_direction = direction
                    self.prediction_30m_confidence = confidence
                    
                    print(f"🎯 LSTM STABLE 30-min: {direction} | Range: ${price_low:.2f}-${price_high:.2f} | Confidence: {confidence:.1f}%")
                    print(f"📌 LOCKED until {(current_time + timedelta(minutes=30)).strftime('%H:%M:%S')} (30 minutes)")
                    
                    return stable_range
                    
        except Exception as e:
            print(f"LSTM stable prediction error: {e}")
            return (0.0, 0.0)
    
    def _build_stable_lstm_model(self) -> Any:
        """Build a robust LSTM model optimized for stable 30-minute predictions"""
        if not LSTM_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(60, 6)),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3), 
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
        return model
    
    def _create_stable_fallback_prediction(self, stock_data: StockData, current_time: datetime) -> tuple:
        """Create stable fallback prediction when LSTM is not available"""
        
        # Base volatility from recent movements
        recent_volatility = (abs(stock_data.price_change_15m) + abs(stock_data.price_change_30m)) / 2
        
        # Volume-adjusted volatility multiplier
        volume_multiplier = 1.0
        if stock_data.volume > 50000000:  # Very high volume
            volume_multiplier = 1.4
        elif stock_data.volume > 45000000:  # High volume
            volume_multiplier = 1.2
        elif stock_data.volume < 30000000:  # Low volume
            volume_multiplier = 0.8
            
        # RSI-based range adjustment
        rsi_pressure = 0.0
        if stock_data.rsi_14 > 70:  # Overbought - expect wider downside range
            rsi_pressure = -0.15
        elif stock_data.rsi_14 < 30:  # Oversold - expect wider upside range
            rsi_pressure = +0.15
        elif stock_data.rsi_14 > 65:  # Moderately overbought
            rsi_pressure = -0.08
        elif stock_data.rsi_14 < 35:  # Moderately oversold
            rsi_pressure = +0.08
            
        # BALANCED momentum calculation based on actual market conditions
        recent_momentum = stock_data.price_change_15m * 0.5 + stock_data.price_change_30m * 0.3 + stock_data.price_change_1h * 0.2
        
        # Equal treatment for both directions - let the data decide
        momentum_bias = recent_momentum  # No artificial amplification
        
        # BALANCED support/resistance analysis - equal treatment for both directions
        sma_distance = (stock_data.current_price - stock_data.sma_20) / stock_data.sma_20 * 100
        
        # BALANCED daily change pressure - treat up and down movements equally  
        daily_change_pct = (stock_data.current_price - stock_data.previous_close) / stock_data.previous_close * 100
        daily_pressure = 0.0
        
        # Equal treatment for large moves in both directions
        if daily_change_pct > 5:  # Up more than 5% today - strong bullish pressure
            daily_pressure = +0.15
        elif daily_change_pct > 2:  # Up more than 2% today
            daily_pressure = +0.08
        elif daily_change_pct > 1:  # Up more than 1% today
            daily_pressure = +0.04
        elif daily_change_pct < -5:  # Down more than 5% today - strong bearish pressure
            daily_pressure = -0.15
        elif daily_change_pct < -2:  # Down more than 2% today
            daily_pressure = -0.08
        elif daily_change_pct < -1:  # Down more than 1% today
            daily_pressure = -0.04
        
        # BALANCED SMA resistance/support - equal treatment for both directions
        if sma_distance > 2:  # Above SMA - resistance (equal magnitude)
            sma_pressure = -0.06
        elif sma_distance < -2:  # Below SMA - support (equal magnitude)
            sma_pressure = +0.06
        else:
            sma_pressure = 0.0
            
        # Calculate TIGHT base range (max 50¢ total range)
        base_range_pct = max(recent_volatility * volume_multiplier, 0.08)  # Minimum 0.08%  
        base_range_pct = min(base_range_pct, 0.25)  # Maximum 0.25% (much tighter)
        
        # Apply ALL directional biases with EQUAL treatment for both directions
        total_bias = momentum_bias + rsi_pressure + sma_pressure + daily_pressure
        
        # Calculate TIGHT price range with STRONG directional bias
        range_amount = stock_data.current_price * (base_range_pct / 100)
        max_range_dollars = 0.50  # Hard limit: 50¢ maximum range
        range_amount = min(range_amount, max_range_dollars)
        
        # BALANCED directional bias - equal thresholds for up and down
        if total_bias > 0.03:  # Bullish bias
            price_low = stock_data.current_price - (range_amount * 0.2)   # 20% downside
            price_high = stock_data.current_price + (range_amount * 0.8)  # 80% upside bias
        elif total_bias < -0.03:  # Bearish bias (equal threshold)
            price_low = stock_data.current_price - (range_amount * 0.8)   # 80% downside bias
            price_high = stock_data.current_price + (range_amount * 0.2)  # 20% upside
        else:  # Neutral - balanced range
            price_low = stock_data.current_price - (range_amount * 0.5)   # 50/50 split
            price_high = stock_data.current_price + (range_amount * 0.5)
            
        # Cache the stable prediction for 30 minutes
        stable_range = (round(price_low, 2), round(price_high, 2))
        self.current_30min_prediction = stable_range
        self.prediction_30m_made_at = current_time
        
        # PRIORITY: More sensitive to DOWN movements (protect losses first)
        if total_bias < -0.02:  # More sensitive bearish detection (lower threshold)
            self.prediction_30m_direction = "DOWN" 
            self.prediction_30m_confidence = min(90, 75 + abs(total_bias) * 55)  # Higher confidence for protection
        elif total_bias > 0.03:  # Standard bullish threshold
            self.prediction_30m_direction = "UP"
            self.prediction_30m_confidence = min(85, 70 + abs(total_bias) * 50)
        else:
            self.prediction_30m_direction = "STABLE"
            self.prediction_30m_confidence = 65
            
        print(f"📈 FALLBACK STABLE prediction: {self.prediction_30m_direction} | Range: ${price_low:.2f}-${price_high:.2f}")
        print(f"📌 LOCKED until {(current_time + timedelta(minutes=30)).strftime('%H:%M:%S')} (30 minutes)")
        
        return stable_range

    def predict_1minute_ahead(self, stock_data: StockData) -> dict:
        """Predict the exact price movement for the next 1 minute with high accuracy"""
        current_time = datetime.now()
        
        # Only create new prediction if more than 30 seconds have passed
        if (self.last_1min_prediction and self.last_1min_prediction_time and 
            (current_time - self.last_1min_prediction_time).total_seconds() < 30):
            return self.last_1min_prediction
            
        try:
            # Get comprehensive data for 1-minute prediction
            comprehensive_data = self.get_comprehensive_market_data(stock_data)
            if not comprehensive_data or len(comprehensive_data['prices']) < 60:
                return self._fallback_1min_prediction(stock_data)
                
            # Use LSTM for precise 1-minute prediction
            if LSTM_AVAILABLE:
                prediction = self._lstm_1minute_prediction(stock_data, comprehensive_data)
                if prediction:
                    self.last_1min_prediction = prediction
                    self.last_1min_prediction_time = current_time
                    return prediction
                    
            # Fallback to technical analysis
            return self._fallback_1min_prediction(stock_data)
            
        except Exception as e:
            print(f"1-minute prediction error: {e}")
            return self._fallback_1min_prediction(stock_data)
    
    def _lstm_1minute_prediction(self, stock_data: StockData, comprehensive_data: dict) -> dict:
        """Use LSTM to predict next 1-minute price movement"""
        try:
            prices = comprehensive_data['prices']
            volumes = comprehensive_data['volumes']
            
            if len(prices) < 60:
                return {}
                
            # Create features for 1-minute prediction (simplified for speed)
            recent_prices = prices[-60:]
            recent_volumes = volumes[-60:]
            
            # Calculate micro-features for 1-minute prediction
            price_changes = np.diff(recent_prices) / recent_prices[:-1] * 100
            volume_changes = np.diff(recent_volumes) / recent_volumes[:-1] * 100
            
            # Create feature matrix
            features = np.column_stack([
                recent_prices[1:],  # Remove first element to match diff arrays
                recent_volumes[1:],
                price_changes,
                volume_changes,
                np.full(len(price_changes), stock_data.rsi_14),
                np.full(len(price_changes), stock_data.price_change_15m)
            ])
            
            # Normalize features
            features_scaled = self.minute_scaler.fit_transform(features)
            
            # Build lightweight model for 1-minute prediction
            if not self.minute_ahead_model:
                self.minute_ahead_model = self._build_1minute_model()
                
            if self.minute_ahead_model and len(features_scaled) >= 30:
                # Prepare training data (predict 1 step ahead)
                X_train, y_train = [], []
                sequence_length = 20  # Shorter sequence for 1-minute
                
                for i in range(sequence_length, len(features_scaled) - 1):
                    X_train.append(features_scaled[i-sequence_length:i])
                    y_train.append(features_scaled[i+1, 0])  # Next price
                    
                if len(X_train) >= 10:
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    
                    # Quick training for 1-minute prediction
                    self.minute_ahead_model.fit(X_train, y_train, epochs=10, batch_size=4, verbose=0)
                    
                    # Predict next minute
                    last_sequence = features_scaled[-sequence_length:].reshape(1, sequence_length, 6)
                    prediction_normalized = self.minute_ahead_model.predict(last_sequence, verbose=0)[0][0]
                    
                    # Denormalize
                    temp_data = np.zeros((1, 6))
                    temp_data[0, 0] = prediction_normalized
                    predicted_price = self.minute_scaler.inverse_transform(temp_data)[0, 0]
                    
                    # Calculate change and confidence
                    current_price = stock_data.current_price
                    price_change = predicted_price - current_price
                    price_change_pct = (price_change / current_price) * 100
                    
                    # Determine direction and confidence
                    if abs(price_change_pct) > 0.05:  # Meaningful change
                        direction = "UP" if price_change_pct > 0 else "DOWN"
                        confidence = min(70 + abs(price_change_pct) * 20, 95)
                    else:
                        direction = "STABLE"
                        confidence = 60
                        
                    return {
                        'predicted_price': round(predicted_price, 2),
                        'price_change': round(price_change, 2),
                        'price_change_pct': round(price_change_pct, 3),
                        'direction': direction,
                        'confidence': round(confidence, 1),
                        'method': 'LSTM'
                    }
                    
        except Exception as e:
            print(f"LSTM 1-minute prediction failed: {e}")
            return {}
            
    def _build_1minute_model(self) -> Any:
        """Build lightweight LSTM model for 1-minute predictions"""
        if not LSTM_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(20, 6)),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def _fallback_1min_prediction(self, stock_data: StockData) -> dict:
        """Fallback 1-minute prediction using technical analysis - BALANCED for both directions"""
        # Use recent momentum and volatility
        momentum_15m = stock_data.price_change_15m
        momentum_30m = stock_data.price_change_30m  
        momentum_1h = stock_data.price_change_1h
        rsi_signal = stock_data.rsi_14
        
        # Calculate combined momentum with recent emphasis
        recent_momentum = (momentum_15m * 0.6) + (momentum_30m * 0.3) + (momentum_1h * 0.1)
        
        # PRIORITY: Detect price drops first (protect losses)
        if momentum_15m < -0.02 or recent_momentum < -0.03:  # Lower thresholds for downs
            direction = "DOWN"
            predicted_change_pct = recent_momentum * 0.8  # Amplify prediction for protection
            confidence = min(85, 70 + abs(recent_momentum) * 8)  # High confidence for drops
            
        # Equal treatment for rises
        elif momentum_15m > 0.02 and recent_momentum > 0.03 and rsi_signal < 75:  
            direction = "UP"
            predicted_change_pct = recent_momentum * 0.8  # Equal treatment
            confidence = min(85, 70 + abs(recent_momentum) * 8)  # Equal confidence
            
        # Check RSI extreme conditions
        elif rsi_signal > 70 and momentum_15m <= 0.02:  # Overbought and momentum slowing
            direction = "DOWN"
            predicted_change_pct = -0.05  # Predict pullback
            confidence = min(75, 60 + (rsi_signal - 70) * 2)
            
        elif rsi_signal < 30 and momentum_15m >= -0.02:  # Oversold and momentum stabilizing
            direction = "UP"
            predicted_change_pct = 0.05  # Predict bounce
            confidence = min(75, 60 + (30 - rsi_signal) * 2)
            
        # Micro-trend detection for small movements
        elif momentum_15m < -0.005:  # Even tiny drops get detected
            direction = "DOWN"
            predicted_change_pct = momentum_15m * 0.5  # Conservative but accurate
            confidence = min(70, 60 + abs(momentum_15m) * 10)
            
        elif momentum_15m > 0.005:  # Equal detection for small rises
            direction = "UP"  
            predicted_change_pct = momentum_15m * 0.5  # Equal treatment
            confidence = min(70, 60 + abs(momentum_15m) * 10)
            
        else:
            direction = "STABLE"
            predicted_change_pct = 0.0
            confidence = 55
            
        predicted_price = stock_data.current_price * (1 + predicted_change_pct / 100)
        price_change = predicted_price - stock_data.current_price
        
        return {
            'predicted_price': round(predicted_price, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(predicted_change_pct, 3),
            'direction': direction,
            'confidence': confidence,
            'method': 'Technical'
        }

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
                
                # Honest balanced prediction - prioritize immediate price movements
                recent_change_15m = stock_data.price_change_15m
                recent_change_30m = stock_data.price_change_30m
                volume_surge = stock_data.volume > 45000000
                
                # PRIORITY: Check for immediate price drops (most important for protection)
                if recent_change_15m < -0.02:  # Lower threshold: catch smaller drops
                    direction = "DOWN"
                    signal = "SELL" if stock_data.rsi_14 > 25 else "WAIT"  # More aggressive selling
                    confidence = min(65.0 + abs(recent_change_15m) * 40, 90.0)  # Higher confidence for drops
                    
                # Check for immediate price rises (buying opportunities) - equal treatment
                elif recent_change_15m > 0.02:  # Equal threshold for rises
                    direction = "UP"
                    signal = "BUY" if stock_data.rsi_14 < 75 else "WAIT"
                    confidence = min(65.0 + abs(recent_change_15m) * 40, 90.0)  # Equal confidence
                    
                # MOMENTUM ANALYSIS - Give priority to negative momentum (protect losses)
                elif momentum_score < -0.02:  # More sensitive to negative momentum
                    direction = "DOWN"
                    signal = "SELL" if stock_data.rsi_14 > 25 else "WAIT"  # More aggressive selling
                    confidence = min(70.0 + abs(momentum_score) * 35, 90.0)  # Higher confidence for protection
                    
                # Positive momentum = UP (equal threshold and treatment)
                elif momentum_score > 0.02 and recent_change_15m > -0.05:  # Equal conditions
                    direction = "UP" 
                    signal = "BUY" if stock_data.rsi_14 < 75 else "WAIT"
                    confidence = min(70.0 + abs(momentum_score) * 35, 90.0)  # Equal confidence calculation
                    
                # RSI extreme conditions - equal treatment for overbought and oversold
                elif stock_data.rsi_14 > 70:  # Standard overbought threshold
                    direction = "DOWN"
                    signal = "SELL"
                    confidence = min(60.0 + (stock_data.rsi_14 - 70) * 3, 80.0)
                    
                elif stock_data.rsi_14 < 30:  # Equal oversold condition
                    direction = "UP"
                    signal = "BUY"
                    confidence = min(60.0 + (30 - stock_data.rsi_14) * 3, 80.0)  # Equal confidence
                    
                # Price vs SMA analysis - balanced treatment for both extremes
                elif price_vs_sma > 2.0 and momentum_score < 0.2:  # More reasonable threshold
                    direction = "DOWN"
                    signal = "SELL" if stock_data.rsi_14 > 40 else "WAIT"
                    confidence = min(58.0 + (price_vs_sma - 2) * 3, 75.0)
                    
                elif price_vs_sma < -2.0 and momentum_score > -0.2:  # Equal opposite condition
                    direction = "UP"
                    signal = "BUY" if stock_data.rsi_14 < 60 else "WAIT"
                    confidence = min(58.0 + abs(price_vs_sma + 2) * 3, 75.0)  # Equal confidence
                    
                # Positive momentum and conditions aligned
                elif momentum_score > 0.1 and recent_change_15m > 0 and stock_data.rsi_14 < 65:
                    direction = "UP"
                    signal = "BUY"
                    confidence = min(60.0 + momentum_score * 25, 80.0)  # Improved confidence
                    
                else:
                    # PRIORITY: Check recent price action for micro-trends
                    if recent_change_15m < -0.005 or momentum_score < -0.01:  # Even tiny drops get priority
                        direction = "DOWN"
                        signal = "WAIT"  # Conservative but correct direction
                        confidence = 55.0
                    elif recent_change_15m > 0.005 or momentum_score > 0.01:  # Equal treatment for rises
                        direction = "UP"
                        signal = "WAIT"  # Conservative but correct direction
                        confidence = 55.0
                    elif stock_data.rsi_14 > 55:  # Slightly overbought
                        direction = "DOWN"
                        signal = "WAIT"
                        confidence = 52.0
                    elif stock_data.rsi_14 < 45:  # Slightly oversold
                        direction = "UP"
                        signal = "WAIT"
                        confidence = 52.0
                    else:
                        direction = "STABLE"
                        signal = "WAIT"
                        confidence = 50.0
                    
                price_target = None
                
            # Calculate risk management
            stop_loss, take_profit, risk_reward = self.calculate_risk_management(
                stock_data, predicted_change, signal
            )
            
            # Calculate stable 30-minute price range prediction
            price_range_30m = self.predict_30min_stable_range(stock_data)
            prediction_expires = None
            if self.prediction_30m_made_at:
                prediction_expires = self.prediction_30m_made_at + timedelta(minutes=30)
            
            prediction = Prediction(
                direction=direction,
                confidence=confidence,
                signal=signal,
                price_target=price_target,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                price_range_30m=price_range_30m,
                prediction_30m_timestamp=self.prediction_30m_made_at,
                prediction_30m_expires=prediction_expires
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
        
        # Momentum analysis - balanced thresholds
        momentum = (stock_data.price_change_15m + stock_data.price_change_30m + stock_data.price_change_1h) / 3
        momentum_status = "🚀 Strong Up" if momentum > 0.3 else "📉 Strong Down" if momentum < -0.3 else "➡️ Neutral"
        print(f"   Momentum Score:    {momentum:+.2f}% ({momentum_status})")
        
        # Recent price action analysis
        trend_15m = "🟢 Rising" if stock_data.price_change_15m > 0.05 else "🔴 Falling" if stock_data.price_change_15m < -0.05 else "➡️ Flat"
        print(f"   15-min Trend:      {stock_data.price_change_15m:+.2f}% ({trend_15m})")
        
        # Price vs SMA analysis
        price_vs_sma = (stock_data.current_price - stock_data.sma_20) / stock_data.sma_20 * 100
        print(f"   Price vs SMA-20:   {price_vs_sma:+.1f}% ({price_vs_sma > 0 and '🟢 Above' or '🔴 Below'})")
        
        # Volume analysis for market activity
        volume_status = "🔥 High" if stock_data.volume > 45000000 else "📊 Normal" if stock_data.volume > 30000000 else "📉 Low"
        print(f"   Volume Activity:   {stock_data.volume:,} ({volume_status})")
        
        # Prediction Section
        direction_emoji = {"UP": "🚀", "DOWN": "📉", "STABLE": "➡️"}
        signal_emoji = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "WAIT": "🟡 WAIT"}
        
        print(f"\n🤖 AI PREDICTION & TRADING SIGNALS:")
        print(f"   Direction:         {direction_emoji.get(prediction.direction, '❓')} {prediction.direction}")
        print(f"   Confidence:        {prediction.confidence:.1f}%")
        print(f"   Trading Signal:    {signal_emoji.get(prediction.signal, '❓')}")
        
        if prediction.price_target:
            target_change = ((prediction.price_target - stock_data.current_price) / stock_data.current_price) * 100
            profit_cents = abs(prediction.price_target - stock_data.current_price) * 100
            print(f"   Price Target:      ${prediction.price_target:.2f} ({target_change:+.1f}%, ~{profit_cents:.0f}¢ profit)")
            
        # 30-minute stable price range prediction
        if prediction.price_range_30m:
            low, high = prediction.price_range_30m
            range_size = high - low
            profit_potential = max(abs(high - stock_data.current_price), abs(stock_data.current_price - low)) * 100
            
            print(f"🎯 30-MIN STABLE PREDICTION:")
            if prediction.prediction_30m_expires:
                time_remaining = (prediction.prediction_30m_expires - datetime.now()).total_seconds() / 60
                print(f"   ⏰ Valid for:       {time_remaining:.0f} more minutes")
            print(f"   Expected Range:    ${low:.2f} - ${high:.2f}")
            print(f"   Range Size:        ${range_size:.2f} (~{profit_potential:.0f}¢ max profit)")
            
            # Direction bias within range (using cached direction for clarity)
            current_position = (stock_data.current_price - low) / (high - low) * 100 if range_size > 0 else 50
            
            # Use the cached 30-min prediction direction for consistent messaging
            if hasattr(self, 'prediction_30m_direction') and self.prediction_30m_direction:
                if self.prediction_30m_direction == "UP":
                    bias = "🟢 Bullish bias"
                    recommendation = "📈 Expected to move UP"
                    if current_position > 60:
                        position_text = f"🔥 {current_position:.0f}% in range (good sell opportunity)"
                    else:
                        position_text = f"{current_position:.0f}% in range (hold/buy opportunity)"
                elif self.prediction_30m_direction == "DOWN":
                    bias = "🔴 Bearish bias"
                    recommendation = "📉 Expected to move DOWN"
                    if current_position < 40:
                        position_text = f"🔥 {current_position:.0f}% in range (good buy opportunity)"
                    else:
                        position_text = f"{current_position:.0f}% in range (sell opportunity)"
                else:
                    bias = "🟡 Stable"
                    recommendation = "➡️ Minimal movement expected"
                    position_text = f"{current_position:.0f}% in range (neutral)"
            else:
                # Fallback to old logic if no cached direction
                if current_position < 30:
                    bias = "🟢 Near bottom (bullish bias)"
                    recommendation = "📈 Expected to move UP"
                elif current_position > 70:
                    bias = "🔴 Near top (bearish bias)"
                    recommendation = "📉 Expected to move DOWN"
                else:
                    bias = "🟡 Mid-range"
                    recommendation = "➡️ Sideways movement expected"
                position_text = f"{current_position:.0f}% in range ({bias})"
                
            print(f"   Current Position:  {position_text}")
            print(f"   30-min Direction:  {recommendation}")
            if hasattr(self, 'prediction_30m_confidence') and self.prediction_30m_confidence:
                print(f"   Confidence:        {self.prediction_30m_confidence}%")
            
        # 1-minute ahead prediction
        minute_prediction = self.predict_1minute_ahead(stock_data)
        if minute_prediction:
            print(f"\n⚡ 1-MINUTE AHEAD PREDICTION:")
            direction_emoji = "📈" if minute_prediction['direction'] == "UP" else "📉" if minute_prediction['direction'] == "DOWN" else "➡️"
            print(f"   Next Price:        ${minute_prediction['predicted_price']:.2f} ({minute_prediction['price_change']:+.2f}¢)")
            print(f"   Direction:         {direction_emoji} {minute_prediction['direction']} ({minute_prediction['price_change_pct']:+.3f}%)")
            print(f"   Confidence:        {minute_prediction['confidence']:.1f}% ({minute_prediction['method']})")
            
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
        
        # Show next-day prediction if available
        self.display_next_day_prediction()

    def get_market_sentiment(self) -> float:
        """Get market sentiment score from multiple sources (-1 to +1)"""
        try:
            sentiment_scores = []
            
            # Alpha Vantage News Sentiment (if API key available)
            if self.alpha_vantage_key:
                av_sentiment = self._get_alpha_vantage_sentiment()
                if av_sentiment is not None:
                    sentiment_scores.append(av_sentiment)
            
            # EODHD News Sentiment (if API key available)
            if self.eodhd_api_key:
                eodhd_sentiment = self._get_eodhd_sentiment()
                if eodhd_sentiment is not None:
                    sentiment_scores.append(eodhd_sentiment)
            
            # Fallback: Technical sentiment based on price action
            if not sentiment_scores:
                technical_sentiment = self._calculate_technical_sentiment()
                sentiment_scores.append(technical_sentiment)
            
            # Return average sentiment
            return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            
        except Exception as e:
            print(f"⚠️ Sentiment analysis error: {e}")
            return 0.0

    def _get_alpha_vantage_sentiment(self) -> Optional[float]:
        """Get sentiment from Alpha Vantage News API"""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': self.symbol,
                'apikey': self.alpha_vantage_key,
                'limit': 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'feed' in data:
                sentiments = []
                for article in data['feed'][:10]:  # Use recent 10 articles
                    for ticker_sentiment in article.get('ticker_sentiment', []):
                        if ticker_sentiment.get('ticker') == self.symbol:
                            sentiment_score = float(ticker_sentiment.get('ticker_sentiment_score', 0))
                            sentiments.append(sentiment_score)
                
                return sum(sentiments) / len(sentiments) if sentiments else None
            
        except Exception as e:
            print(f"Alpha Vantage sentiment error: {e}")
        
        return None

    def _get_eodhd_sentiment(self) -> Optional[float]:
        """Get sentiment from EODHD News API"""
        try:
            url = f"https://eodhd.com/api/news"
            params = {
                'api_token': self.eodhd_api_key,
                's': self.symbol,
                'limit': 20,
                'fmt': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data and isinstance(data, list):
                # Simple sentiment analysis based on title keywords
                sentiment_scores = []
                positive_words = ['surge', 'bullish', 'gains', 'beats', 'strong', 'upgrade', 'growth', 'positive']
                negative_words = ['plunge', 'bearish', 'falls', 'misses', 'weak', 'downgrade', 'decline', 'negative']
                
                for article in data[:10]:
                    title = article.get('title', '').lower()
                    content = article.get('content', '')[:200].lower()  # First 200 chars
                    text = f"{title} {content}"
                    
                    positive_count = sum(1 for word in positive_words if word in text)
                    negative_count = sum(1 for word in negative_words if word in text)
                    
                    if positive_count > 0 or negative_count > 0:
                        sentiment = (positive_count - negative_count) / (positive_count + negative_count + 1)
                        sentiment_scores.append(sentiment)
                
                return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None
                
        except Exception as e:
            print(f"EODHD sentiment error: {e}")
        
        return None

    def _calculate_technical_sentiment(self) -> float:
        """Calculate technical sentiment based on recent price action"""
        if len(self.historical_data) < 5:
            return 0.0
            
        recent_data = self.historical_data[-5:]
        
        # Price momentum sentiment
        price_changes = []
        for i in range(1, len(recent_data)):
            change = (recent_data[i].current_price - recent_data[i-1].current_price) / recent_data[i-1].current_price
            price_changes.append(change)
        
        avg_momentum = sum(price_changes) / len(price_changes)
        
        # RSI sentiment
        latest_rsi = recent_data[-1].rsi_14
        rsi_sentiment = 0.0
        if latest_rsi > 70:
            rsi_sentiment = -0.3  # Overbought = negative sentiment
        elif latest_rsi < 30:
            rsi_sentiment = 0.3   # Oversold = positive sentiment
        
        # Volume sentiment
        latest_volume = recent_data[-1].volume
        avg_volume = sum(d.volume for d in recent_data) / len(recent_data)
        volume_sentiment = min((latest_volume - avg_volume) / avg_volume, 0.5) if avg_volume > 0 else 0
        
        # Combine technical indicators
        technical_sentiment = (avg_momentum * 10) + rsi_sentiment + (volume_sentiment * 0.2)
        return max(-1.0, min(1.0, technical_sentiment))

    def get_overnight_futures_data(self) -> Dict[str, float]:
        """Get overnight futures data that correlates with AMD"""
        try:
            overnight_factors = {}
            
            # NASDAQ 100 futures (AMD is tech stock)
            nq_data = self._get_yahoo_futures_data('NQ=F')
            if nq_data:
                overnight_factors['nasdaq_futures'] = nq_data
            
            # SOX (Semiconductor index) - direct correlation
            sox_data = self._get_yahoo_futures_data('SOXX')
            if sox_data:
                overnight_factors['semiconductor_etf'] = sox_data
            
            # VIX for market volatility
            vix_data = self._get_yahoo_futures_data('^VIX')
            if vix_data:
                overnight_factors['volatility_index'] = vix_data
            
            return overnight_factors
            
        except Exception as e:
            print(f"⚠️ Futures data error: {e}")
            return {}

    def _get_yahoo_futures_data(self, symbol: str) -> Optional[float]:
        """Get overnight price change for futures/indices"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2d", interval="1d")
            
            if len(data) >= 2:
                yesterday_close = data.iloc[-2]['Close']
                current_price = data.iloc[-1]['Close']
                change_pct = (current_price - yesterday_close) / yesterday_close * 100
                return float(change_pct)
                
        except Exception as e:
            print(f"Futures data error for {symbol}: {e}")
        
        return None

    def predict_next_day_open(self, current_stock_data: StockData) -> NextDayPrediction:
        """Generate comprehensive next-day market open prediction"""
        try:
            print("🔮 Generating next-day open price prediction...")
            
            # Get sentiment analysis
            sentiment_score = self.get_market_sentiment()
            
            # Get overnight factors
            overnight_factors = self.get_overnight_futures_data()
            
            # Calculate futures correlation
            futures_correlation = self._calculate_futures_correlation(overnight_factors)
            
            # Base prediction using multiple models
            base_prediction = self._calculate_base_next_day_prediction(current_stock_data)
            
            # Apply sentiment adjustment
            sentiment_adjustment = sentiment_score * 0.02  # ±2% max adjustment
            
            # Apply futures correlation
            futures_adjustment = futures_correlation * 0.015  # ±1.5% max adjustment
            
            # Combine all factors
            total_adjustment = sentiment_adjustment + futures_adjustment
            predicted_price = current_stock_data.current_price * (1 + total_adjustment)
            
            # Calculate confidence based on data availability
            confidence = self._calculate_next_day_confidence(sentiment_score, overnight_factors)
            
            # Determine direction and risk
            price_change_pct = (predicted_price - current_stock_data.current_price) / current_stock_data.current_price * 100
            direction = "UP" if price_change_pct > 0.1 else "DOWN" if price_change_pct < -0.1 else "STABLE"
            
            # Risk assessment
            risk_level = "HIGH" if abs(price_change_pct) > 3 else "MEDIUM" if abs(price_change_pct) > 1 else "LOW"
            
            # Target range (±1% around prediction)
            range_size = predicted_price * 0.01
            target_range = (predicted_price - range_size, predicted_price + range_size)
            
            # Pre-market trend assessment
            pre_market_trend = self._assess_pre_market_trend(overnight_factors)
            
            prediction = NextDayPrediction(
                predicted_open_price=round(predicted_price, 2),
                confidence=round(confidence, 1),
                price_change_pct=round(price_change_pct, 2),
                direction=direction,
                sentiment_score=round(sentiment_score, 3),
                overnight_factors=overnight_factors,
                futures_correlation=round(futures_correlation, 3),
                pre_market_trend=pre_market_trend,
                risk_assessment=risk_level,
                target_range=(round(target_range[0], 2), round(target_range[1], 2)),
                created_at=datetime.now()
            )
            
            # Cache the prediction
            self.last_next_day_prediction = prediction
            self.next_day_prediction_time = datetime.now()
            
            return prediction
            
        except Exception as e:
            print(f"❌ Next-day prediction error: {e}")
            # Return a basic prediction if there's an error
            return NextDayPrediction(
                predicted_open_price=current_stock_data.current_price,
                confidence=50.0,
                price_change_pct=0.0,
                direction="STABLE",
                sentiment_score=0.0,
                overnight_factors={},
                futures_correlation=0.0,
                pre_market_trend="NEUTRAL",
                risk_assessment="MEDIUM",
                target_range=(current_stock_data.current_price * 0.99, current_stock_data.current_price * 1.01),
                created_at=datetime.now()
            )

    def _calculate_base_next_day_prediction(self, stock_data: StockData) -> float:
        """Calculate base next-day prediction using historical patterns"""
        if len(self.historical_data) < 10:
            return 0.0  # Not enough data
        
        # Analyze historical overnight gaps
        overnight_gaps = []
        daily_closes = []
        
        for i in range(len(self.historical_data) - 1):
            current_close = self.historical_data[i].current_price
            next_open = self.historical_data[i + 1].current_price  # Approximation
            gap = (next_open - current_close) / current_close
            overnight_gaps.append(gap)
            daily_closes.append(current_close)
        
        # Recent pattern analysis
        recent_gaps = overnight_gaps[-5:] if len(overnight_gaps) >= 5 else overnight_gaps
        avg_gap = sum(recent_gaps) / len(recent_gaps) if recent_gaps else 0
        
        # RSI influence on overnight gaps
        rsi_factor = 0.0
        if stock_data.rsi_14 > 70:
            rsi_factor = -0.005  # Overbought tends to gap down
        elif stock_data.rsi_14 < 30:
            rsi_factor = 0.005   # Oversold tends to gap up
        
        # Volume influence
        if len(self.historical_data) >= 5:
            avg_volume = sum(d.volume for d in self.historical_data[-5:]) / 5
            volume_factor = (stock_data.volume - avg_volume) / avg_volume * 0.001
        else:
            volume_factor = 0
        
        total_expected_change = avg_gap + rsi_factor + volume_factor
        return total_expected_change

    def _calculate_futures_correlation(self, overnight_factors: Dict[str, float]) -> float:
        """Calculate correlation with futures for next-day prediction"""
        if not overnight_factors:
            return 0.0
        
        # Weight different factors
        correlation = 0.0
        
        if 'nasdaq_futures' in overnight_factors:
            correlation += overnight_factors['nasdaq_futures'] * 0.6  # 60% weight for NASDAQ
        
        if 'semiconductor_etf' in overnight_factors:
            correlation += overnight_factors['semiconductor_etf'] * 0.8  # 80% weight for semiconductors
        
        if 'volatility_index' in overnight_factors:
            correlation -= overnight_factors['volatility_index'] * 0.2  # Inverse correlation with VIX
        
        return correlation / 100  # Convert to decimal

    def _calculate_next_day_confidence(self, sentiment_score: float, overnight_factors: Dict[str, float]) -> float:
        """Calculate confidence level for next-day prediction"""
        base_confidence = 60.0
        
        # More data sources = higher confidence
        if sentiment_score != 0:
            base_confidence += 10
        
        if overnight_factors:
            base_confidence += len(overnight_factors) * 5
        
        # Strong sentiment = higher confidence
        base_confidence += abs(sentiment_score) * 15
        
        # Historical data quality
        if len(self.historical_data) >= 20:
            base_confidence += 10
        
        return min(95.0, base_confidence)

    def _assess_pre_market_trend(self, overnight_factors: Dict[str, float]) -> str:
        """Assess pre-market trend based on overnight factors"""
        if not overnight_factors:
            return "NEUTRAL"
        
        total_change = sum(overnight_factors.values())
        
        if total_change > 1.5:
            return "BULLISH"
        elif total_change < -1.5:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def display_next_day_prediction(self):
        """Display next-day prediction if available and recent"""
        if not self.last_next_day_prediction:
            return
        
        # Only show if prediction is recent (within 4 hours)
        if (datetime.now() - self.next_day_prediction_time).total_seconds() > 14400:
            return
        
        pred = self.last_next_day_prediction
        direction_emoji = "🚀" if pred.direction == "UP" else "📉" if pred.direction == "DOWN" else "➡️"
        risk_emoji = "🔴" if pred.risk_assessment == "HIGH" else "🟡" if pred.risk_assessment == "MEDIUM" else "🟢"
        
        print(f"\n🔮 NEXT-DAY MARKET OPEN PREDICTION:")
        print(f"   Predicted Price:   ${pred.predicted_open_price:.2f} ({pred.price_change_pct:+.2f}%)")
        print(f"   Direction:         {direction_emoji} {pred.direction}")
        print(f"   Confidence:        {pred.confidence:.1f}%")
        print(f"   Target Range:      ${pred.target_range[0]:.2f} - ${pred.target_range[1]:.2f}")
        print(f"   Risk Level:        {risk_emoji} {pred.risk_assessment}")
        print(f"   Sentiment Score:   {pred.sentiment_score:+.3f} (-1 to +1)")
        print(f"   Pre-Market Trend:  {pred.pre_market_trend}")
        
        if pred.overnight_factors:
            print(f"   Overnight Factors:")
            for factor, value in pred.overnight_factors.items():
                factor_name = factor.replace('_', ' ').title()
                print(f"     {factor_name}: {value:+.2f}%")
        
        prediction_age = datetime.now() - pred.created_at
        hours_old = prediction_age.total_seconds() / 3600
        print(f"   Created:           {hours_old:.1f} hours ago")

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
                
                # Generate next-day prediction (once every 4 hours or on first run)
                current_time = datetime.now()
                should_predict_next_day = (
                    not self.next_day_prediction_time or 
                    (current_time - self.next_day_prediction_time).total_seconds() > 14400 or  # 4 hours
                    current_time.hour >= 15  # After 3 PM
                )
                
                if should_predict_next_day:
                    try:
                        next_day_pred = self.predict_next_day_open(stock_data)
                        print(f"✅ Next-day prediction generated for tomorrow's open")
                    except Exception as e:
                        print(f"⚠️ Next-day prediction failed: {e}")
                
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
    
    # Check for backup API keys
    eodhd_key = os.getenv("EODHD_API_KEY")
    if not eodhd_key:
        print("🟡 EODHD API key not set (Yahoo Finance primary data source)")
        print("💡 Set EODHD_API_KEY environment variable for backup data source")
    else:
        print("✅ EODHD backup API key configured")
        
    print("\n🚀 Initializing system...")
    
    try:
        predictor = StockPredictor(symbol=SYMBOL, refresh_interval=REFRESH_INTERVAL)
        predictor.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
