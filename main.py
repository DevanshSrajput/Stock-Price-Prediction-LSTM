import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except ImportError:
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
    except ImportError:
        st.error("❌ TensorFlow/Keras not found. Please install: pip install tensorflow")
        st.stop()

# -----------------------------
# Core LSTM Stock Prediction Functions
# -----------------------------

@st.cache_data
def download_stock_data(ticker, start_date, end_date):
    """Download stock data with caching"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Select only necessary columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]
        
        df.dropna(inplace=True)
        if len(df) == 0:
            raise ValueError(f"No valid data found for ticker {ticker}")
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        raise ValueError(f"Error downloading data for {ticker}: {str(e)}")

def create_sequences(data, window_size):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(window_size):
    """Build LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_next_days(model, last_seq, scaler, n_days=7):
    """Forecast future prices"""
    preds = []
    current_seq = last_seq.copy()
    for _ in range(n_days):
        pred = model.predict(current_seq.reshape(1, len(current_seq), 1), verbose=0)
        preds.append(pred[0,0])
        current_seq = np.append(current_seq[1:], pred[0,0])
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    return preds_inv.flatten()

# -----------------------------
# Streamlit App Configuration
# -----------------------------

st.set_page_config(
    page_title="LSTM Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
    }
    .forecast-item {
        padding: 0.5rem;
        margin: 0.2rem 0;
        background-color: #f8f9fa;
        border-radius: 5px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📈 LSTM Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Sidebar Configuration
# -----------------------------

with st.sidebar:
    st.header("🔧 Configuration")
    
    # Stock selection
    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter the stock symbol (e.g., AAPL, GOOGL, TSLA)"
    ).upper().strip()
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2020, 1, 1),
            help="Start date for historical data"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime(2023, 12, 31),
            help="End date for historical data"
        )
    
    # Model parameters
    st.subheader("🧠 Model Parameters")
    window_size = st.slider(
        "Window Size",
        min_value=10,
        max_value=100,
        value=60,
        help="Number of days to look back for prediction"
    )
    
    forecast_days = st.slider(
        "Forecast Days",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to forecast"
    )
    
    epochs = st.slider(
        "Training Epochs",
        min_value=5,
        max_value=50,
        value=20,
        help="Number of training epochs"
    )
    
    # Predict button
    predict_button = st.button("🚀 Run Prediction", use_container_width=True)

# -----------------------------
# Main Application Logic
# -----------------------------

if predict_button:
    if not ticker:
        st.error("❌ Please enter a stock ticker symbol")
        st.stop()
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Step 1: Download data
            status_text.text("📥 Downloading stock data...")
            progress_bar.progress(10)
            
            df = download_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if len(df) < window_size + forecast_days + 20:
                st.error(f"❌ Not enough data for the selected parameters. Found {len(df)} rows, need at least {window_size + forecast_days + 20}.")
                st.stop()
            
            status_text.text("✅ Data downloaded successfully!")
            progress_bar.progress(20)
            
            # Display stock info in sidebar column
            with col2:
                st.subheader(f"📊 {ticker} Overview")
                latest_price = float(df['Close'].iloc[-1])
                price_change = float(df['Close'].iloc[-1] - df['Close'].iloc[-2])
                price_change_pct = (price_change / float(df['Close'].iloc[-2])) * 100
                
                st.metric(
                    label="Latest Price",
                    value=f"${latest_price:.2f}",
                    delta=f"{price_change_pct:.2f}%"
                )
                
                st.metric("Data Points", len(df))
                st.metric("Date Range", f"{len(df)} days")
            
            # Step 2: Prepare data
            status_text.text("🔄 Preparing data...")
            progress_bar.progress(30)
            
            # Convert to numpy array and handle any potential issues
            close_prices = df['Close'].values.astype(np.float32)
            close_data = close_prices.reshape(-1, 1)
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_close = scaler.fit_transform(close_data)
            
            X, y = create_sequences(scaled_close, window_size)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Step 3: Build and train model
            status_text.text("🧠 Training LSTM model...")
            progress_bar.progress(50)
            
            model = build_lstm_model(window_size)
            
            # Training with progress tracking
            with st.spinner("Training model... This may take a few minutes."):
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
            
            # Step 4: Make predictions
            status_text.text("📊 Making predictions...")
            progress_bar.progress(80)
            
            y_pred = model.predict(X_test, verbose=0)
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_inv = scaler.inverse_transform(y_pred)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
            mae = mean_absolute_error(y_test_inv, y_pred_inv)
            
            status_text.text("✅ Prediction completed!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_container.empty()
            
            # Step 5: Display results
            st.subheader("📈 Prediction Results")
            
            # Metrics display
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>RMSE</h3>
                    <h2>${rmse:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>MAE</h3>
                    <h2>${mae:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                accuracy = max(0, 100 - (mae / np.mean(y_test_inv)) * 100)
                st.markdown(f"""
                <div class="metric-container">
                    <h3>Accuracy</h3>
                    <h2>{accuracy:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Main prediction chart
            fig = go.Figure()
            
            # Create date index for test data
            test_start_idx = split + window_size
            test_dates = df.index[test_start_idx:test_start_idx + len(y_test_inv)]
            
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=y_test_inv.flatten(),
                mode='lines',
                name='Actual Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=y_pred_inv.flatten(),
                mode='lines',
                name='Predicted Price',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'{ticker} Stock Price Prediction',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Future forecast section
            st.subheader("🔮 Future Price Forecast")
            
            last_seq = scaled_close[-window_size:]
            future_predictions = forecast_next_days(model, last_seq, scaler, n_days=forecast_days)
            
            # Create future dates (business days only)
            last_date = df.index[-1]
            future_dates = []
            current_date = last_date
            for i in range(forecast_days):
                current_date += timedelta(days=1)
                # Skip weekends for stock predictions
                while current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                    current_date += timedelta(days=1)
                future_dates.append(current_date)
            
            # Display forecast in two columns
            forecast_col1, forecast_col2 = st.columns([1, 1])
            
            with forecast_col1:
                st.write("**📈 Future Price Predictions**")
                for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
                    st.markdown(f"""
                    <div class="forecast-item">
                        <strong>Day {i+1}</strong> ({date.strftime('%Y-%m-%d')}): <strong>${price:.2f}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            with forecast_col2:
                # Forecast chart
                fig_forecast = go.Figure()
                
                # Add recent historical data (last 30 days)
                recent_data = df['Close'].tail(30)
                fig_forecast.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='#1f77b4')
                ))
                
                # Add forecast
                fig_forecast.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#ff7f0e', dash='dash'),
                    marker=dict(size=8)
                ))
                
                fig_forecast.update_layout(
                    title='Price Forecast',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Training history (expandable)
            with st.expander("📊 Training History", expanded=False):
                if history and hasattr(history, 'history'):
                    fig_history = go.Figure()
                    
                    epochs_range = list(range(1, len(history.history['loss']) + 1))
                    
                    fig_history.add_trace(go.Scatter(
                        x=epochs_range,
                        y=history.history['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='#1f77b4')
                    ))
                    
                    if 'val_loss' in history.history:
                        fig_history.add_trace(go.Scatter(
                            x=epochs_range,
                            y=history.history['val_loss'],
                            mode='lines',
                            name='Validation Loss',
                            line=dict(color='#ff7f0e')
                        ))
                    
                    fig_history.update_layout(
                        title='Model Training History',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_history, use_container_width=True)
                else:
                    st.info("Training history not available.")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please try adjusting the parameters or selecting a different stock.")

else:
    # Welcome screen
    welcome_col1, welcome_col2, welcome_col3 = st.columns([1, 2, 1])
    
    with welcome_col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>🎯 Welcome to LSTM Stock Predictor</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Predict stock prices using advanced LSTM neural networks
            </p>
            <br>
            <p>📊 Configure your parameters in the sidebar and click "Run Prediction" to get started!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("### ✨ Features")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            - 🧠 **Deep Learning**: Advanced LSTM neural networks
            - 📈 **Real-time Data**: Live stock data from Yahoo Finance
            - 🎯 **Accurate Predictions**: High-precision forecasting
            """)
        
        with feature_col2:
            st.markdown("""
            - 📊 **Interactive Charts**: Beautiful visualizations
            - 🔮 **Future Forecasting**: Predict multiple days ahead
            - ⚡ **Fast Processing**: Optimized for quick results
            """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit and TensorFlow</p>",
    unsafe_allow_html=True
)