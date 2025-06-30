# 📈 LSTM Stock Price Predictor

A **modern, intuitive web application** for predicting stock prices using LSTM neural networks. Built with cutting-edge technology to make stock prediction accessible, beautiful, and surprisingly accurate.

## ✨ Features

- **🧠 Deep Learning Power:** Advanced LSTM neural networks that actually learn from market patterns
- **🌐 Modern Web Interface:** Sleek Streamlit-based UI that puts old desktop apps to shame
- **📊 Interactive Charts:** Beautiful Plotly visualizations that respond to your every hover
- **📈 Real-time Data:** Live stock data from Yahoo Finance, because fresh data makes better predictions
- **🎯 Smart Forecasting:** Predict multiple days ahead with configurable parameters
- **📱 Responsive Design:** Works perfectly on desktop, tablet, and mobile
- **⚡ Lightning Fast:** Optimized processing with progress tracking
- **🔮 Future Vision:** See tomorrow's prices today (with scientific disclaimers, of course)

## 🚀 Quick Start

### Option 1: One-Click Launch

1. **Double-click** `run_app.bat`
2. **Wait** for your browser to open automatically
3. **Start predicting!**

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run main.py
```

## 💡 How to Use

### 1. Configure Your Prediction

Use the **sidebar controls** to set up your prediction:

- **📈 Stock Ticker:** Enter any stock symbol (AAPL, GOOGL, TSLA, etc.)
- **📅 Date Range:** Choose your historical data timeframe
- **🔢 Window Size:** Days to look back (10-100, default: 60)
- **🔮 Forecast Days:** Days to predict ahead (1-30, default: 7)
- **🧠 Training Epochs:** Model training intensity (5-50, default: 20)

### 2. Run Your Prediction

- **Click** the "🚀 Run Prediction" button
- **Watch** the real-time progress bar
- **See** your results unfold beautifully

### 3. Analyze Results

- **📊 Metrics Dashboard:** RMSE, MAE, and accuracy scores
- **📈 Interactive Charts:** Hover, zoom, and explore your predictions
- **🔮 Future Forecast:** See predicted prices with confidence intervals
- **📉 Training History:** Monitor how well your model learned

## 🔬 How It Works (The Science Behind the Magic)

### Data Pipeline

- **📥 Data Acquisition:** Fetches real-time stock data from Yahoo Finance API
- **🧹 Data Preprocessing:** Cleans and normalizes data using MinMaxScaler
- **📊 Sequence Generation:** Creates time-series windows for pattern recognition
- **✅ Data Validation:** Ensures data quality and completeness

### LSTM Architecture

- **🧠 Neural Network:** 2-layer LSTM with 50 units each
- **🎯 Dropout Layers:** 20% dropout to prevent overfitting
- **⚙️ Optimization:** Adam optimizer with mean squared error loss
- **📈 Training:** Configurable epochs with validation split

### Prediction Engine

- **🔮 Forecasting:** Multi-step ahead predictions using recursive approach
- **📊 Metrics:** RMSE, MAE, and custom accuracy calculations
- **📈 Visualization:** Interactive Plotly charts with hover details
- **💾 State Management:** Efficient memory usage with Streamlit caching

## 🛠️ Technical Stack

### Core Technologies

- **🌐 Streamlit:** Modern web app framework
- **🧠 TensorFlow/Keras:** Deep learning powerhouse
- **📊 Plotly:** Interactive visualization magic
- **📈 yfinance:** Real-time stock data
- **🔢 NumPy/Pandas:** Data manipulation masters
- **🎯 scikit-learn:** ML utilities and metrics

### Requirements

```
streamlit>=1.28.0
plotly>=5.15.0
tensorflow>=2.13.0
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

## 💡 Pro Tips for Better Predictions

### Data Selection

- **📅 Use 2+ years** of historical data for better training
- **🎯 Popular stocks** (AAPL, MSFT, GOOGL) have cleaner data
- **⏰ Avoid recent IPOs** - they lack historical patterns

### Model Tuning

- **📈 Window Size:** 60 days works well for most stocks
- **🔄 Epochs:** Start with 20, increase for better accuracy
- **📊 Forecast Days:** 7-14 days for reliable predictions

### Performance Optimization

- **⚡ Faster Training:** Reduce epochs for quick experiments
- **🎯 Better Accuracy:** Increase epochs and window size
- **💾 Memory:** Close other apps for large datasets

## ⚠️ Important Disclaimers

### Financial Advisory

- **🚨 Not Financial Advice:** This is an educational tool, not investment guidance
- **📊 Past Performance ≠ Future Results:** Markets are unpredictable
- **💰 Risk Management:** Never invest more than you can afford to lose
- **🤝 Consult Professionals:** Always seek qualified financial advice

### Technical Limitations

- **🎲 Market Randomness:** No model can predict black swan events
- **📈 Short-term Volatility:** Daily predictions are inherently noisy
- **🌍 External Factors:** News, politics, and sentiment affect prices
- **🔄 Model Retraining:** Markets evolve, models need updates

## 🐛 Troubleshooting

### Common Issues

| Problem                | Solution                                  |
| ---------------------- | ----------------------------------------- |
| **"No data found"**    | ✅ Check ticker symbol spelling           |
| **Slow performance**   | ✅ Reduce epochs or date range            |
| **Import errors**      | ✅ Run `pip install -r requirements.txt`  |
| **Memory issues**      | ✅ Close other applications               |
| **Browser won't open** | ✅ Manually go to `http://localhost:8501` |

### Getting Help

- **📖 Check the logs** in the terminal for error details
- **🔄 Restart the app** if it becomes unresponsive
- **🧹 Clear browser cache** if charts don't load
- **💻 Update packages** with `pip install --upgrade -r requirements.txt`

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

- **🐛 Bug Reports:** Found an issue? Let us know!
- **💡 Feature Requests:** Have ideas? We'd love to hear them!
- **🔧 Code Improvements:** PRs are always welcome
- **📚 Documentation:** Help make the docs even better
- **🎨 UI/UX:** Make the interface more beautiful

### Development Setup

```bash
# Fork the repository
git clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt

# Make your changes and test
streamlit run main.py
```

## 📄 License

**MIT License** - Feel free to use, modify, and distribute this code. Just remember to give credit where it's due!

## 🙏 Acknowledgments

- **📊 Yahoo Finance** for providing free stock data
- **🧠 TensorFlow team** for the amazing ML framework
- **🌐 Streamlit** for making web apps ridiculously easy
- **📈 Plotly** for beautiful, interactive charts
- **🎯 The open-source community** for endless inspiration

---

### 🎯 Ready to Predict the Future?

**Launch the app and start exploring the fascinating world of stock prediction!**

_Built with ❤️ by developers who believe that making complex AI accessible is the future of technology._

---

_May your predictions be accurate and your portfolios ever-growing! 📈✨_
