"""
Stock Price Data Collection Module
Handles downloading and preprocessing historical stock data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
import os

class StockDataCollector:
    """
    Collects and preprocesses historical stock data from Yahoo Finance
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data collector
        
        Args:
            data_dir: Directory to save raw data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_stock_data(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Download historical stock data for given tickers
        
        Args:
            tickers: List of stock ticker symbols (e.g., ['AAPL', 'GOOG'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            save: Whether to save data to disk
            
        Returns:
            DataFrame with multi-level columns (ticker, metric)
        """
        print(f"Downloading data for {len(tickers)} stocks from {start_date} to {end_date}...")
        
        # Download data for all tickers
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=False,  # Keep Adj Close separate
            progress=True
        )
        
        if save:
            filename = f"{self.data_dir}/stock_data_{'_'.join(tickers)}_{start_date}_{end_date}.csv"
            data.to_csv(filename)
            print(f"Data saved to {filename}")
        
        return data
    
    def get_stock_info(self, ticker: str) -> dict:
        """
        Get additional information about a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'ticker': ticker,
            'company_name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'description': info.get('longBusinessSummary', 'N/A')
        }
    
    def add_technical_indicators(self, df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
        """
        Add technical indicators to the stock data
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Ticker symbol (for multi-ticker DataFrames)
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        # Handle multi-ticker vs single-ticker DataFrames
        if ticker:
            close = df[ticker]['Close']
            high = df[ticker]['High']
            low = df[ticker]['Low']
            volume = df[ticker]['Volume']
        else:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
        
        # Simple Moving Averages
        sma_7 = close.rolling(window=7).mean()
        sma_21 = close.rolling(window=21).mean()
        sma_50 = close.rolling(window=50).mean()
        
        # Exponential Moving Averages
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        
        # MACD
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_middle = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        # Volume indicators
        volume_sma = volume.rolling(window=20).mean()
        
        # Price momentum
        momentum = close.pct_change(periods=10)
        
        # Volatility
        volatility = close.pct_change().rolling(window=20).std()
        
        # Create new columns
        indicators = pd.DataFrame({
            'SMA_7': sma_7,
            'SMA_21': sma_21,
            'SMA_50': sma_50,
            'EMA_12': ema_12,
            'EMA_26': ema_26,
            'MACD': macd,
            'MACD_Signal': signal,
            'RSI': rsi,
            'BB_Upper': bb_upper,
            'BB_Middle': bb_middle,
            'BB_Lower': bb_lower,
            'Volume_SMA': volume_sma,
            'Momentum': momentum,
            'Volatility': volatility,
            'Daily_Return': close.pct_change()
        }, index=df.index)
        
        if ticker:
            # Add to multi-ticker DataFrame
            for col in indicators.columns:
                df[ticker, col] = indicators[col]
        else:
            # Concatenate for single-ticker
            df = pd.concat([df, indicators], axis=1)
        
        return df
    
    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        train_end_date: str,
        ticker: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets
        
        Args:
            df: DataFrame with stock data
            train_end_date: End date for training data
            ticker: Ticker symbol (for multi-ticker DataFrames)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = df[:train_end_date]
        test_df = df[train_end_date:]
        
        print(f"Training set: {len(train_df)} samples")
        print(f"Testing set: {len(test_df)} samples")
        
        return train_df, test_df


# Example usage
if __name__ == "__main__":
    # Initialize collector
    collector = StockDataCollector()
    
    # Define parameters
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    start_date = '2020-01-01'
    end_date = '2024-10-01'
    
    # Download data
    data = collector.download_stock_data(tickers, start_date, end_date)
    
    # Add technical indicators for each ticker
    for ticker in tickers:
        data = collector.add_technical_indicators(data, ticker)
    
    # Get company info
    for ticker in tickers:
        info = collector.get_stock_info(ticker)
        print(f"\n{info['company_name']} ({ticker})")
        print(f"Sector: {info['sector']}")
    
    print("\nData shape:", data.shape)
    print("\nFirst few rows:")
    print(data.head())
