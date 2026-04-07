# Market Muse

An upgraded Streamlit dashboard for stock forecasting with a more thoughtful UI and a stronger modeling workflow than the original close-only LSTM demo.

## What’s Better Now

- Multivariate LSTM training with technical signals like returns, moving averages, volatility, momentum, and RSI
- Baseline comparison against a naive previous-close model so forecast quality has context
- Confidence bands for forward forecasts
- Clearer diagnostics with RMSE, MAE, MAPE, and directional accuracy
- A redesigned interface with a guided overview, diagnostics tab, and forecast desk
- Exportable forecast table for quick downstream use

## App Experience

The app is now organized into three parts:

1. `Overview`
   Read recent price action, liquidity, volatility, and the model’s forward path.
2. `Diagnostics`
   Compare the LSTM against a naive baseline and inspect training history plus residual drift.
3. `Forecast Desk`
   Review the forecast table, narrative insights, and download the result as CSV.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

### Apple Silicon Note

If you are on an M-series Mac and see low-level TensorFlow crashes such as `mutex.cc` or `mutex lock failed`, recreate the environment with the Apple-specific packages from `requirements.txt`:

```bash
rm -rf .env
python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py
```

## Controls

- `Quick ticker`: fast entry for popular symbols or a custom symbol
- `Lookback window`: how many past sessions the model sees in each sample
- `Forecast horizon`: how many future business days to project
- `Training epochs`: max epochs before early stopping steps in
- `Train split`: proportion of data used for model fitting before evaluation

## Modeling Notes

- Historical data comes from Yahoo Finance via `yfinance`
- The model uses engineered technical features derived from price history
- Training uses early stopping and learning-rate reduction
- Forecast bands are heuristic ranges based on test-set residual dispersion

## Important Caveat

This project is for experimentation and education. It is not financial advice, and the forecast ranges are not guarantees.
