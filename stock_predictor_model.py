"""
Stock Price Prediction Models
Implements multiple ML algorithms for stock price forecasting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Tuple
import pickle
import warnings
warnings.filterwarnings('ignore')

# For LSTM (requires TensorFlow/Keras)
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow not available. LSTM model will be disabled.")


class StockPredictor:
    """
    Multi-model stock price predictor with ensemble capabilities
    """
    
    def __init__(self, model_type: str = 'random_forest', lookback_days: int = 60):
        """
        Initialize the stock predictor
        
        Args:
            model_type: Type of model ('linear', 'random_forest', 'gradient_boosting', 'lstm', 'ensemble')
            lookback_days: Number of past days to use for prediction
        """
        self.model_type = model_type
        self.lookback_days = lookback_days
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_columns = []
        self.ticker_models = {}
        
    def prepare_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Prepare features for training/prediction
        
        Args:
            df: DataFrame with stock data and technical indicators
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with prepared features
        """
        # Select relevant columns
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_7', 'SMA_21', 'SMA_50',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
            'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
            'Volume_SMA', 'Momentum', 'Volatility', 'Daily_Return'
        ]
        
        # Extract features for the specific ticker
        features = pd.DataFrame()
        for col in feature_cols:
            if (ticker, col) in df.columns:
                features[col] = df[ticker][col]
            elif col in df.columns:
                features[col] = df[col]
        
        # Add lag features
        for lag in [1, 2, 3, 5, 7]:
            features[f'Close_Lag_{lag}'] = features['Close'].shift(lag)
        
        # Add rolling statistics
        features['Close_Rolling_Mean_7'] = features['Close'].rolling(window=7).mean()
        features['Close_Rolling_Std_7'] = features['Close'].rolling(window=7).std()
        features['Volume_Rolling_Mean_7'] = features['Volume'].rolling(window=7).mean()
        
        # Drop rows with NaN values
        features = features.dropna()
        
        self.feature_columns = features.columns.tolist()
        
        return features
    
    def create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction (for LSTM)
        
        Args:
            data: Scaled feature array
            lookback: Number of time steps to look back
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, 0])  # Predict 'Close' price (assuming it's first column)
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame, tickers: List[str]):
        """
        Train the model on historical data
        
        Args:
            df: DataFrame with stock data
            tickers: List of ticker symbols to train on
        """
        print(f"Training {self.model_type} model for {len(tickers)} stocks...")
        
        for ticker in tickers:
            print(f"\nTraining model for {ticker}...")
            
            # Prepare features
            features = self.prepare_features(df, ticker)
            
            # Target variable: Adjusted Close (or Close if Adj Close not available)
            if (ticker, 'Adj Close') in df.columns:
                target = df[ticker]['Adj Close'].loc[features.index]
            else:
                target = df[ticker]['Close'].loc[features.index]
            
            # Remove target from features if it exists
            if 'Adj Close' in features.columns:
                features = features.drop('Adj Close', axis=1)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(features)
            y = target.values
            
            # Train model based on type
            if self.model_type == 'linear':
                model = LinearRegression()
                model.fit(X_scaled, y)
                
            elif self.model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_scaled, y)
                
            elif self.model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                model.fit(X_scaled, y)
                
            elif self.model_type == 'lstm' and KERAS_AVAILABLE:
                # Reshape for LSTM
                X_seq, y_seq = self.create_sequences(
                    np.column_stack([y.reshape(-1, 1), X_scaled]),
                    self.lookback_days
                )
                
                # Build LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25),
                    Dense(1)
                ])
                
                model.compile(optimizer='adam', loss='mse')
                
                early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                model.fit(X_seq, y_seq, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])
                
            elif self.model_type == 'ensemble':
                # Train multiple models
                models_dict = {
                    'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'ridge': Ridge(alpha=1.0)
                }
                
                for name, m in models_dict.items():
                    m.fit(X_scaled, y)
                
                model = models_dict  # Store all models
            
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.ticker_models[ticker] = {
                'model': model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            
            print(f"✓ Model trained for {ticker}")
    
    def predict(self, df: pd.DataFrame, tickers: List[str], dates: List[str]) -> Dict[str, pd.Series]:
        """
        Predict stock prices for given dates
        
        Args:
            df: DataFrame with stock data up to prediction date
            tickers: List of ticker symbols
            dates: List of dates to predict (must be after training data)
            
        Returns:
            Dictionary mapping ticker to Series of predictions
        """
        predictions = {}
        
        for ticker in tickers:
            if ticker not in self.ticker_models:
                print(f"Warning: No trained model for {ticker}")
                continue
            
            # Prepare features
            features = self.prepare_features(df, ticker)
            
            # Filter for requested dates
            features_to_predict = features.loc[dates] if dates else features
            
            # Scale features
            model_info = self.ticker_models[ticker]
            X_scaled = model_info['scaler'].transform(features_to_predict)
            
            # Make predictions
            model = model_info['model']
            
            if self.model_type == 'ensemble':
                # Average predictions from all models
                preds = []
                for m in model.values():
                    preds.append(m.predict(X_scaled))
                pred = np.mean(preds, axis=0)
            else:
                pred = model.predict(X_scaled)
            
            predictions[ticker] = pd.Series(pred, index=features_to_predict.index)
        
        return predictions
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate prediction performance
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


# Example usage
if __name__ == "__main__":
    # This is a template - actual usage would require real data
    print("Stock Predictor Model Module")
    print("Available models: linear, random_forest, gradient_boosting, lstm, ensemble")
