# LSTM Stock Price Predictor

Welcome to the **LSTM Stock Price Predictor** – because who *doesn't* want to predict the unpredictable? This project uses a Long Short-Term Memory (LSTM) neural network to forecast stock prices, so you can finally pretend you know what the market will do next.

## Features

- **Download Historical Data:** Fetches stock data from Yahoo Finance, because we trust free APIs with our financial future.
- **LSTM Model:** Uses a deep learning model that "remembers" the past, unlike most investors.
- **Customizable Parameters:** Ticker, date range, window size, and forecast days – because one size never fits all.
- **Interactive GUI:** Built with Tkinter, so you can click buttons instead of typing commands like it's 1995.
- **Pretty Plots:** Visualizes actual vs. predicted prices, so you can see just how wrong (or right) the model is.
- **Performance Metrics:** RMSE and MAE, for when you want numbers to confirm your suspicions.

## Installation

Because nothing says "fun" like installing dependencies:

```sh
pip install -r requirements.txt
```
## Usage
1. Run the app (assuming you have Python and a will to experiment):
```
python main.py
```
2. Enter your favorite (or least favorite) stock ticker, date range, window size, and how many days you want to "predict."

3. Click Predict and watch the magic (or chaos) unfold.

## How It Works (Allegedly)
- **Data Download:** download_stock_data grabs historical stock prices from Yahoo Finance.
- **Preprocessing:** Data is scaled using MinMaxScaler, because neural networks are picky eaters.
- **Sequence Creation:** create_sequences slices the data into overlapping windows, like a chef with OCD.
- **Model Building:** build_lstm_model creates a Sequential LSTM model with dropout layers, because overfitting is so last season.
- **Training:** The model is trained for 10 epochs. Why 10? Because patience is a virtue we don't have.
- **Prediction:** The model predicts on the test set and then tries to forecast the next N days, just to keep things interesting.
- **Results:** Actual vs. predicted prices are plotted, and metrics are displayed so you can quantify your disappointment.

## Requirements
See requirements.txt for the full list, but here's the gist:

- pandas
- numpy
- matplotlib
- yfinance
- scikit-learn
- tensorflow
## Limitations
- Not Financial Advice: If you use this to make investment decisions, that's on you.
- Short Training: 10 epochs won't make you rich, but it will make your CPU warm.
- No Hyperparameter Tuning: Because who has time for that?
- Market Randomness: LSTM can't predict Elon Musk's tweets (yet).
## Contributing
Feel free to fork, PR, or open issues. Or just stare at the code and wonder why you thought this would be easy.

## License
MIT – because sharing is caring, and lawsuits are boring.

---
May your predictions be accurate and your losses minimal.