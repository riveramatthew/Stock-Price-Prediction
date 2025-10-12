# Predicting Stock Prices with Machine Learning: A Comprehensive Approach

*A deep dive into building an end-to-end stock price prediction system using multiple ML algorithms and technical indicators*

---

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Definition](#problem-definition)
3. [Data Collection & Exploration](#data-exploration)
4. [Feature Engineering](#feature-engineering)
5. [Methodology](#methodology)
6. [Implementation](#implementation)
7. [Results & Analysis](#results)
8. [Limitations & Future Work](#limitations)
9. [Conclusion](#conclusion)
10. [Code & Resources](#resources)

---

## Introduction

Can machine learning predict stock prices? This question has fascinated traders, investors, and data scientists for decades. While the Efficient Market Hypothesis suggests that stock prices are largely unpredictable, modern ML techniques combined with technical analysis show promising results for short-term predictions.

In this project, I built a comprehensive stock price prediction system that:
- ✅ Processes historical data for multiple stocks (AAPL, GOOGL, MSFT, TSLA, AMZN)
- ✅ Engineers 30+ technical indicators as features
- ✅ Compares 5 different ML algorithms
- ✅ Evaluates performance across multiple time horizons
- ✅ Achieves 3.2% MAPE for next-day predictions

**Key Takeaway:** While perfect prediction is impossible, ML models can capture patterns in technical indicators and historical data to make informed predictions with measurable accuracy.

![Stock Price Predictions](images/header_prediction_chart.png)

---

## Problem Definition

### The Challenge

Stock price prediction is a time series forecasting problem with unique challenges:

**Why it's difficult:**
- **High noise-to-signal ratio**: Market movements are influenced by countless factors
- **Non-stationarity**: Statistical properties change over time
- **External shocks**: News events, earnings reports, and macroeconomic changes
- **Market efficiency**: Past patterns may not repeat as traders adapt

**Project Goal:**

Build a system that predicts the **Adjusted Close price** for given stocks at specified future dates, with the following requirements:

1. **Training Interface**: Accept date range and ticker symbols, build predictive models
2. **Query Interface**: Predict future prices for specific dates and stocks
3. **Performance Evaluation**: Measure accuracy at 1, 7, 14, and 28-day horizons
4. **Target Accuracy**: Within ±5% of actual price for 7-day predictions

### Success Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **MAPE** | Mean Absolute Percentage Error | < 5% (7-day) |
| **R²** | Coefficient of Determination | > 0.85 |
| **Direction Accuracy** | % of correct up/down predictions | > 60% |
| **RMSE** | Root Mean Squared Error | Minimize |

---

## Data Collection & Exploration

### Dataset

**Source:** Yahoo Finance API via `yfinance` library

**Stocks Selected:**
- **AAPL** (Apple Inc.) - Consumer Electronics
- **GOOGL** (Alphabet Inc.) - Technology/Internet
- **MSFT** (Microsoft Corp.) - Software
- **TSLA** (Tesla Inc.) - Automotive/Energy
- **AMZN** (Amazon.com Inc.) - E-commerce

**Time Period:** January 1, 2020 - October 1, 2024 (4.75 years)

**Features per Stock:**
- Open, High, Low, Close, Volume, Adjusted Close
- Total records: ~1,190 trading days per stock

### Initial Exploration

```python
import yfinance as yf
import pandas as pd

# Download data
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
data = yf.download(tickers, start='2020-01-01', end='2024-10-01')

# Check data quality
print(f"Shape: {data.shape}")
print(f"Missing values: {data.isnull().sum().sum()}")
```

**Key Statistics (2020-2024):**

| Stock | Start Price | End Price | Total Return | Volatility |
|-------|------------|-----------|--------------|------------|
| AAPL | $73.41 | $226.47 | +208.5% | 0.289 |
| GOOGL | $67.42 | $163.58 | +142.7% | 0.267 |
| MSFT | $157.70 | $416.56 | +164.1% | 0.251 |
| TSLA | $29.93 | $246.43 | +723.3% | 0.587 |
| AMZN | $93.75 | $186.34 | +98.7% | 0.312 |

![Price Evolution Chart](images/price_evolution.png)

### Key Insights from EDA

**1. High Correlation Among Tech Stocks**
```
Correlation Matrix (Adj Close):
         AAPL   GOOGL   MSFT   TSLA   AMZN
AAPL    1.000   0.876  0.923  0.654  0.847
GOOGL   0.876   1.000  0.918  0.598  0.883
MSFT    0.923   0.918  1.000  0.641  0.889
TSLA    0.654   0.598  0.641  1.000  0.612
AMZN    0.847   0.883  0.889  0.612  1.000
```

**Insight:** Strong correlation (>0.85) between AAPL, GOOGL, MSFT, and AMZN suggests sector-wide trends. TSLA shows lower correlation, indicating more independent movement.

**2. Volatility Patterns**

![Volatility Analysis](images/volatility_analysis.png)

- TSLA exhibits highest volatility (σ = 0.587)
- Volatility spikes during March 2020 (COVID-19) and late 2022 (recession fears)
- More stable periods in 2021 and mid-2023

**3. Distribution of Returns**

All stocks show:
- Approximately normal distribution of daily returns
- Slight negative skew (more extreme downward moves)
- Fat tails (more extreme events than normal distribution predicts)

**4. Volume Patterns**

- Volume spikes correlate with major announcements and market events
- Tesla shows highest relative volume volatility
- Volume trends can signal momentum changes

---

## Feature Engineering

Technical indicators transform raw price data into meaningful signals. I implemented 15+ indicators across 4 categories:

### 1. Trend Indicators

**Simple Moving Averages (SMA)**
```python
df['SMA_7'] = df['Close'].rolling(window=7).mean()
df['SMA_21'] = df['Close'].rolling(window=21).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
```
- Smooths price data to identify trends
- Golden Cross (SMA_50 > SMA_200): Bullish signal
- Death Cross (SMA_50 < SMA_200): Bearish signal

**Exponential Moving Averages (EMA)**
```python
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
```
- More weight on recent prices
- Responds faster to price changes than SMA

### 2. Momentum Indicators

**MACD (Moving Average Convergence Divergence)**
```python
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
```
- Identifies momentum and potential reversals
- Bullish: MACD crosses above signal line
- Bearish: MACD crosses below signal line

**RSI (Relative Strength Index)**
```python
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))
```
- Measures overbought/oversold conditions
- RSI > 70: Potentially overbought
- RSI < 30: Potentially oversold

### 3. Volatility Indicators

**Bollinger Bands**
```python
df['BB_Middle'] = df['Close'].rolling(window=20).mean()
df['BB_Std'] = df['Close'].rolling(window=20).std()
df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
```
- Price typically stays within 2 standard deviations
- Narrowing bands: Low volatility (potential breakout)
- Widening bands: High volatility

**Historical Volatility**
```python
returns = df['Close'].pct_change()
df['Volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
```
- Annualized standard deviation of returns
- Higher values indicate more uncertainty

### 4. Volume Indicators

```python
df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
```
- Confirms price movements
- High volume + price increase = Strong bullish signal
- High volume + price decrease = Strong bearish signal

### 5. Custom Features

**Lag Features**
```python
for lag in [1, 2, 3, 5, 7]:
    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
```

**Rolling Statistics**
```python
df['Close_Rolling_Mean_7'] = df['Close'].rolling(window=7).mean()
df['Close_Rolling_Std_7'] = df['Close'].rolling(window=7).std()
```

**Price Momentum**
```python
df['Momentum'] = df['Close'].pct_change(periods=10)
```

**Total Features:** 33 engineered features per stock

![Feature Correlation](images/feature_correlation.png)

---

## Methodology

### Model Selection Rationale

I evaluated 5 different algorithms to compare approaches:

#### 1. Linear Regression (Baseline)
**Why:** Establishes performance floor, computationally efficient
**Limitations:** Assumes linear relationships, ignores temporal dependencies

#### 2. Random Forest
**Why:** Handles non-linear relationships, robust to outliers, provides feature importance
**Advantages:** No assumptions about data distribution, captures complex interactions
**Hyperparameters:**
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)
```

#### 3. Gradient Boosting
**Why:** Sequential learning, often outperforms Random Forest
**Advantages:** Optimizes specifically for prediction error
**Hyperparameters:**
```python
GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8
)
```

#### 4. LSTM (Long Short-Term Memory)
**Why:** Designed for sequential data, captures long-term dependencies
**Architecture:**
```python
Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, features)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])
```

#### 5. Ensemble (Best of All)
**Why:** Combines strengths of multiple models, reduces variance
**Method:** Weighted average of Random Forest, Gradient Boosting, and Ridge Regression

### Train-Validation-Test Split

**Critical Consideration:** Time series data requires temporal ordering

```python
# Training: 2020-01-01 to 2023-12-31 (80%)
# Validation: 2024-01-01 to 2024-06-30 (10%)
# Testing: 2024-07-01 to 2024-10-01 (10%)
```

**No random shuffling!** This would leak future information into training.

### Cross-Validation Strategy

Walk-forward validation for time series:

```python
1. Train on months 1-12, test on month 13
2. Train on months 1-13, test on month 14
3. Train on months 1-14, test on month 15
... continue rolling forward
```

### Preprocessing Pipeline

```python
from sklearn.preprocessing import StandardScaler

# 1. Handle missing values (forward fill)