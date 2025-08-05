# AMD Stock Prediction System

## Overview

This is a terminal-based real-time stock analysis and prediction tool that uses machine learning to analyze AMD stock performance. The system provides live stock data monitoring with technical indicators and generates trading predictions using linear regression models. It's designed as a command-line application that delivers real-time insights for stock trading decisions.

## User Preferences

Preferred communication style: Simple, everyday language.

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
- **Linear Regression Model**: Implements scikit-learn's LinearRegression for price prediction
- **Feature Engineering**: Incorporates technical indicators (SMA, RSI) and price change metrics across multiple timeframes
- **Data Preprocessing**: Uses StandardScaler for feature normalization to improve model performance

### Technical Analysis Components
- **Multiple Timeframe Analysis**: Tracks price changes across 15-minute, 30-minute, and 1-hour intervals
- **Technical Indicators**: Implements Simple Moving Average (SMA-20) and Relative Strength Index (RSI-14)
- **Signal Generation**: Produces actionable trading signals (BUY/SELL/WAIT) with confidence scores

### Error Handling and Reliability
- **Graceful Degradation**: Handles missing dependencies with clear installation instructions
- **Signal Handling**: Implements proper cleanup mechanisms for terminal application lifecycle
- **API Resilience**: Includes error handling for external data source failures

## External Dependencies

### Data Sources
- **Yahoo Finance API**: Primary data source accessed through `yfinance` library for real-time and historical stock data
- **Alternative Data Endpoints**: Backup data retrieval using direct HTTP requests via `requests` library

### Machine Learning Libraries
- **scikit-learn**: Core machine learning framework providing LinearRegression and StandardScaler
- **NumPy**: Numerical computing foundation for array operations and mathematical calculations
- **Pandas**: Data manipulation and analysis library for handling time series stock data

### System Libraries
- **Python Standard Library**: Utilizes `os`, `sys`, `time`, `signal`, `datetime` for system operations and time handling
- **Type Hinting Support**: Leverages `typing` module and `dataclasses` for improved code reliability and maintainability