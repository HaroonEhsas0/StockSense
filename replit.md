# AMD Stock Prediction System

## Overview

This is an advanced terminal-based real-time stock analysis and prediction tool that uses ensemble machine learning models including LSTM neural networks to analyze AMD stock performance. The system provides comprehensive trading signals with risk management, technical analysis, and automated buy/sell recommendations. Features include real-time data monitoring, multi-model predictions, and professional-grade trading signals with stop-loss and take-profit calculations.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

### Migration to Replit Environment (2025-08-06)
- Successfully migrated from Replit Agent to standard Replit environment
- Fixed TensorFlow type hint issue with Sequential class by using Any type
- All dependencies installed and working correctly
- Application runs successfully with real-time stock data fetching
- **CONFIRMED: Using 100% authentic APIs** - Yahoo Finance provides real-time data without API keys
- EODHD API available as backup (requires API key for enhanced features)
- Real-time AMD stock tracking operational with live price updates
- Security practices maintained with proper client/server separation

## System Architecture

### Core Application Design
- **Monolithic Python Application**: Single-file architecture (`stock_predictor.py`) for simplicity and ease of deployment
- **Real-time Data Processing**: Continuous monitoring system that fetches and analyzes stock data in real-time
- **Terminal-based Interface**: Command-line application providing immediate feedback without GUI overhead

### Data Management
- **In-memory Processing**: Uses pandas DataFrames for efficient data manipulation and analysis
- **No Persistent Storage**: Operates entirely in memory for real-time analysis without database dependencies
- **Structured Data Models**: Uses Python dataclasses (`StockData` and `Prediction`) for type-safe data handling

### Machine Learning Pipeline
- **Ensemble Model Architecture**: Multiple ML models for improved prediction accuracy
  - Linear Regression for baseline predictions
  - Random Forest (100 estimators) for non-linear pattern recognition
  - LSTM Neural Network for time-series sequence modeling
- **Advanced Feature Engineering**: Enhanced feature set including technical indicators, momentum scores, and volatility measures
- **Data Preprocessing**: MinMaxScaler normalization optimized for neural network training

### Trading System Components
- **Risk Management**: Automated stop-loss and take-profit calculations with 2:1 risk-reward ratios
- **Multiple Timeframe Analysis**: Enhanced momentum scoring across 15-minute, 30-minute, and 1-hour intervals
- **Technical Indicators**: SMA-20, RSI-14 with intelligent overbought/oversold detection
- **Signal Generation**: Professional trading signals (BUY/SELL/WAIT) with confidence scores and price targets
- **Performance Tracking**: Historical prediction accuracy monitoring and model performance metrics

### Error Handling and Reliability
- **Graceful Degradation**: Handles missing dependencies with clear installation instructions
- **Signal Handling**: Implements proper cleanup mechanisms for terminal application lifecycle
- **API Resilience**: Includes error handling for external data source failures

## External Dependencies

### Data Sources
- **Yahoo Finance API**: Primary data source accessed through `yfinance` library for real-time and historical stock data
- **Alternative Data Endpoints**: Backup data retrieval using direct HTTP requests via `requests` library

### Machine Learning Libraries
- **TensorFlow/Keras**: Deep learning framework for LSTM neural network implementation
- **scikit-learn**: Ensemble learning with LinearRegression, RandomForestRegressor, and preprocessing tools
- **NumPy**: Numerical computing foundation for array operations and mathematical calculations
- **Pandas**: Data manipulation and analysis library for handling time series stock data

### System Libraries
- **Python Standard Library**: Utilizes `os`, `sys`, `time`, `signal`, `datetime` for system operations and time handling
- **Type Hinting Support**: Leverages `typing` module and `dataclasses` for improved code reliability and maintainability