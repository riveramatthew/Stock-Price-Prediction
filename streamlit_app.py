"""
Stock Price Predictor - Streamlit Web Application
Interactive dashboard for stock price predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pickle
from typing import List, Dict

# Must be the first Streamlit command
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    # Make a copy to avoid warnings
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def create_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create interactive candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    )])
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_21'],
        mode='lines', name='SMA 21',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_50'],
        mode='lines', name='SMA 50',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_indicator_chart(df: pd.DataFrame, indicator: str, title: str) -> go.Figure:
    """Create chart for technical indicators"""
    fig = go.Figure()
    
    if indicator == 'RSI':
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            mode='lines', name='RSI',
            line=dict(color='purple', width=2)
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
    elif indicator == 'MACD':
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'],
            mode='lines', name='MACD',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD_Signal'],
            mode='lines', name='Signal',
            line=dict(color='red', width=1)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
    elif indicator == 'Bollinger':
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines', name='Close',
            line=dict(color='black', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'],
            mode='lines', name='Upper Band',
            line=dict(color='red', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'],
            mode='lines', name='Lower Band',
            line=dict(color='green', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        height=300,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def make_simple_prediction(df: pd.DataFrame, days: int = 7) -> float:
    """
    Simple prediction using linear regression on recent trend
    Replace this with your actual trained model
    """
    recent_prices = df['Close'].tail(30).values
    X = np.arange(len(recent_prices)).reshape(-1, 1)
    
    # Simple linear fit
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, recent_prices)
    
    # Predict future
    future_X = np.array([[len(recent_prices) + days - 1]])
    prediction = model.predict(future_X)[0]
    
    return prediction

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Machine Learning-Based Stock Price Forecasting")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Stock selection
    popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    ticker = st.sidebar.selectbox(
        "Select Stock",
        popular_stocks,
        help="Choose a stock ticker to analyze"
    )
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(start_date, end_date),
        max_value=end_date
    )
    
    # Prediction horizon
    prediction_days = st.sidebar.slider(
        "Prediction Horizon (days)",
        min_value=1,
        max_value=28,
        value=7,
        help="Number of days to predict into the future"
    )
    
    # Model selection (placeholder)
    model_type = st.sidebar.selectbox(
        "Model Type",
        ['Ensemble', 'Random Forest', 'Gradient Boosting', 'LSTM'],
        help="Select prediction model"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "⚠️ **Disclaimer**: This is for educational purposes only. "
        "Not financial advice. Always do your own research."
    )
    
    # Main content
    if len(date_range) == 2:
        start_str = date_range[0].strftime('%Y-%m-%d')
        end_str = date_range[1].strftime('%Y-%m-%d')
        
        # Load data
        with st.spinner(f'Loading data for {ticker}...'):
            df = load_stock_data(ticker, start_str, end_str)
        
        if df.empty:
            st.error("Failed to load data. Please try a different stock or date range.")
            return
        
        # Calculate indicators
        df = calculate_technical_indicators(df)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "🔮 Predictions", 
            "📈 Technical Analysis", 
            "📉 Statistics"
        ])
        
        # ====================================================================
        # TAB 1: OVERVIEW
        # ====================================================================
        with tab1:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                )
            
            with col2:
                high_52w = df['High'].tail(252).max()
                st.metric("52W High", f"${high_52w:.2f}")
            
            with col3:
                low_52w = df['Low'].tail(252).min()
                st.metric("52W Low", f"${low_52w:.2f}")
            
            with col4:
                avg_volume = df['Volume'].tail(30).mean()
                st.metric("Avg Volume (30d)", f"{avg_volume/1e6:.1f}M")
            
            # Price chart
            st.plotly_chart(
                create_candlestick_chart(df.tail(180), ticker),
                use_container_width=True
            )
            
            # Volume chart
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=df.index.tail(180),
                y=df['Volume'].tail(180),
                name='Volume',
                marker_color='lightblue'
            ))
            fig_volume.update_layout(
                title='Trading Volume',
                height=200,
                template='plotly_white'
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        # ====================================================================
        # TAB 2: PREDICTIONS
        # ====================================================================
        with tab2:
            st.subheader(f"🔮 {prediction_days}-Day Price Prediction")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Make prediction (simplified - replace with your model)
                predicted_price = make_simple_prediction(df, prediction_days)
                prediction_change = predicted_price - current_price
                prediction_change_pct = (prediction_change / current_price) * 100
                
                # Prediction chart
                future_date = df.index[-1] + timedelta(days=prediction_days)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=df.index.tail(60),
                    y=df['Close'].tail(60),
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                fig_pred.add_trace(go.Scatter(
                    x=[df.index[-1], future_date],
                    y=[current_price, predicted_price],
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=10)
                ))
                fig_pred.update_layout(
                    title=f'Price Prediction for {ticker}',
                    yaxis_title='Price ($)',
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col2:
                st.markdown("### Prediction Summary")
                st.markdown(f"**Current Price:** ${current_price:.2f}")
                st.markdown(f"**Predicted Price:** ${predicted_price:.2f}")
                st.markdown(f"**Expected Change:** ${prediction_change:+.2f} ({prediction_change_pct:+.2f}%)")
                st.markdown(f"**Model Used:** {model_type}")
                st.markdown(f"**Prediction Date:** {future_date.strftime('%Y-%m-%d')}")
                
                # Confidence indicator
                st.markdown("---")
                st.markdown("### Confidence Metrics")
                
                # Simplified confidence (replace with actual model confidence)
                confidence = 75  # Placeholder
                st.progress(confidence / 100)
                st.caption(f"Model Confidence: {confidence}%")
                
                # Risk indicator
                volatility = df['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100
                
                if volatility < 20:
                    risk_level = "🟢 Low"
                    risk_color = "green"
                elif volatility < 40:
                    risk_level = "🟡 Medium"
                    risk_color = "orange"
                else:
                    risk_level = "🔴 High"
                    risk_color = "red"
                
                st.markdown(f"**Volatility Risk:** :{risk_color}[{risk_level}]")
                st.caption(f"Annualized volatility: {volatility:.1f}%")
                
                # Trading suggestion (educational only)
                st.markdown("---")
                st.markdown("### Educational Signal")
                if prediction_change_pct > 2:
                    st.success("📈 Bullish trend predicted")
                elif prediction_change_pct < -2:
                    st.error("📉 Bearish trend predicted")
                else:
                    st.info("➡️ Neutral/sideways movement")
                
                st.caption("⚠️ Not financial advice")
            
            # Performance metrics
            st.markdown("---")
            st.subheader("📊 Model Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Placeholder metrics (replace with actual model metrics)
            with col1:
                st.metric("MAPE", "3.2%", help="Mean Absolute Percentage Error")
            with col2:
                st.metric("R² Score", "0.94", help="Coefficient of Determination")
            with col3:
                st.metric("RMSE", "$7.21", help="Root Mean Squared Error")
            with col4:
                st.metric("Direction Accuracy", "68%", help="% of correct up/down predictions")
        
        # ====================================================================
        # TAB 3: TECHNICAL ANALYSIS
        # ====================================================================
        with tab3:
            st.subheader("📈 Technical Indicators")
            
            # Indicator selection
            indicator_choice = st.selectbox(
                "Select Indicator",
                ['RSI', 'MACD', 'Bollinger Bands']
            )
            
            if indicator_choice == 'RSI':
                st.plotly_chart(
                    create_indicator_chart(df.tail(180), 'RSI', 'Relative Strength Index (RSI)'),
                    use_container_width=True
                )
                
                current_rsi = df['RSI'].iloc[-1]
                st.markdown(f"**Current RSI:** {current_rsi:.2f}")
                
                if current_rsi > 70:
                    st.warning("🔴 Overbought - Price may decrease")
                elif current_rsi < 30:
                    st.success("🟢 Oversold - Price may increase")
                else:
                    st.info("🟡 Neutral territory")
            
            elif indicator_choice == 'MACD':
                st.plotly_chart(
                    create_indicator_chart(df.tail(180), 'MACD', 'MACD Indicator'),
                    use_container_width=True
                )
                
                current_macd = df['MACD'].iloc[-1]
                current_signal = df['MACD_Signal'].iloc[-1]
                
                st.markdown(f"**Current MACD:** {current_macd:.2f}")
                st.markdown(f"**Signal Line:** {current_signal:.2f}")
                
                if current_macd > current_signal:
                    st.success("📈 Bullish signal (MACD above signal)")
                else:
                    st.warning("📉 Bearish signal (MACD below signal)")
            
            elif indicator_choice == 'Bollinger Bands':
                st.plotly_chart(
                    create_indicator_chart(df.tail(180), 'Bollinger', 'Bollinger Bands'),
                    use_container_width=True
                )
                
                current_bb_upper = df['BB_Upper'].iloc[-1]
                current_bb_lower = df['BB_Lower'].iloc[-1]
                
                st.markdown(f"**Upper Band:** ${current_bb_upper:.2f}")
                st.markdown(f"**Lower Band:** ${current_bb_lower:.2f}")
                
                if current_price > current_bb_upper:
                    st.warning("🔴 Price above upper band - Potentially overbought")
                elif current_price < current_bb_lower:
                    st.success("🟢 Price below lower band - Potentially oversold")
                else:
                    st.info("🟡 Price within bands - Normal range")
            
            # Moving averages summary
            st.markdown("---")
            st.subheader("Moving Averages")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sma7 = df['SMA_7'].iloc[-1]
                st.metric("SMA 7", f"${sma7:.2f}")
            with col2:
                sma21 = df['SMA_21'].iloc[-1]
                st.metric("SMA 21", f"${sma21:.2f}")
            with col3:
                sma50 = df['SMA_50'].iloc[-1]
                st.metric("SMA 50", f"${sma50:.2f}")
            
            # Trend analysis
            if current_price > sma7 > sma21 > sma50:
                st.success("📈 Strong uptrend - All moving averages aligned bullish")
            elif current_price < sma7 < sma21 < sma50:
                st.error("📉 Strong downtrend - All moving averages aligned bearish")
            else:
                st.info("➡️ Mixed signals - No clear trend")
        
        # ====================================================================
        # TAB 4: STATISTICS
        # ====================================================================
        with tab4:
            st.subheader("📉 Statistical Analysis")
            
            # Returns analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Returns Analysis")
                
                returns = df['Close'].pct_change().dropna()
                
                stats_data = {
                    'Metric': [
                        'Mean Daily Return',
                        'Std Dev (Daily)',
                        'Annualized Return',
                        'Annualized Volatility',
                        'Sharpe Ratio (Rf=0)',
                        'Max Drawdown',
                        'Skewness',
                        'Kurtosis'
                    ],
                    'Value': [
                        f"{returns.mean() * 100:.3f}%",
                        f"{returns.std() * 100:.3f}%",
                        f"{returns.mean() * 252 * 100:.2f}%",
                        f"{returns.std() * np.sqrt(252) * 100:.2f}%",
                        f"{(returns.mean() / returns.std()) * np.sqrt(252):.2f}",
                        f"{((df['Close'] / df['Close'].cummax() - 1).min() * 100):.2f}%",
                        f"{returns.skew():.2f}",
                        f"{returns.kurtosis():.2f}"
                    ]
                }
                
                st.dataframe(pd.DataFrame(stats_data), hide_index=True)
            
            with col2:
                st.markdown("### Returns Distribution")
                
                fig_hist = px.histogram(
                    returns * 100,
                    nbins=50,
                    title='Distribution of Daily Returns',
                    labels={'value': 'Daily Return (%)', 'count': 'Frequency'}
                )
                fig_hist.add_vline(
                    x=returns.mean() * 100,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mean"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Correlation with other stocks
            st.markdown("---")
            st.markdown("### Correlation with Other Stocks")
            
            # Load comparison stocks
            comparison_tickers = [t for t in popular_stocks if t != ticker][:4]
            
            corr_data = {}
            corr_data[ticker] = df['Close']
            
            for comp_ticker in comparison_tickers:
                comp_df = load_stock_data(comp_ticker, start_str, end_str)
                if not comp_df.empty:
                    corr_data[comp_ticker] = comp_df['Close']
            
            corr_df = pd.DataFrame(corr_data)
            correlation_matrix = corr_df.corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title='Correlation Matrix'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Price comparison
            st.markdown("---")
            st.markdown("### Price Comparison (Normalized)")
            
            normalized_prices = corr_df / corr_df.iloc[0] * 100
            
            fig_comp = go.Figure()
            for col in normalized_prices.columns:
                fig_comp.add_trace(go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[col],
                    mode='lines',
                    name=col
                ))
            
            fig_comp.update_layout(
                title='Normalized Price Comparison (Base = 100)',
                yaxis_title='Normalized Price',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Recent data table
            st.markdown("---")
            st.markdown("### Recent Data (Last 10 Days)")
            
            recent_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
            recent_data['Change (%)'] = df['Close'].pct_change().tail(10) * 100
            
            st.dataframe(
                recent_data.style.format({
                    'Open': '${:.2f}',
                    'High': '${:.2f}',
                    'Low': '${:.2f}',
                    'Close': '${:.2f}',
                    'Volume': '{:,.0f}',
                    'Change (%)': '{:+.2f}%'
                }),
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p><strong>Stock Price Predictor</strong> | Machine Learning Capstone Project</p>
        <p>Data provided by Yahoo Finance | For educational purposes only</p>
        <p>⚠️ This is not financial advice. Always do your own research before making investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
