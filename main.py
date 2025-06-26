import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import matplotlib
matplotlib.use("TkAgg")  # Use Tkinter backend for matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# Core LSTM Stock Prediction Functions
# -----------------------------

def download_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(window_size):
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
    preds = []
    current_seq = last_seq.copy()
    for _ in range(n_days):
        pred = model.predict(current_seq.reshape(1, len(current_seq), 1), verbose=0)
        preds.append(pred[0,0])
        current_seq = np.append(current_seq[1:], pred[0,0])
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    return preds_inv.flatten()

# -----------------------------
# Tkinter UI
# -----------------------------

class StockPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LSTM Stock Price Predictor")
        self.root.geometry("900x650")
        
        # Input frame
        input_frame = ttk.LabelFrame(root, text="Parameters")
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(input_frame, text="Ticker:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.ticker_entry = ttk.Entry(input_frame, width=10)
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5)
        self.ticker_entry.insert(0, "AAPL")
        
        ttk.Label(input_frame, text="Start Date:").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.start_entry = ttk.Entry(input_frame, width=12)
        self.start_entry.grid(row=0, column=3, padx=5, pady=5)
        self.start_entry.insert(0, "2017-01-01")
        
        ttk.Label(input_frame, text="End Date:").grid(row=0, column=4, padx=5, pady=5, sticky='e')
        self.end_entry = ttk.Entry(input_frame, width=12)
        self.end_entry.grid(row=0, column=5, padx=5, pady=5)
        self.end_entry.insert(0, "2023-12-31")
        
        ttk.Label(input_frame, text="Window Size:").grid(row=0, column=6, padx=5, pady=5, sticky='e')
        self.window_entry = ttk.Entry(input_frame, width=5)
        self.window_entry.grid(row=0, column=7, padx=5, pady=5)
        self.window_entry.insert(0, "60")

        ttk.Label(input_frame, text="Forecast Days:").grid(row=0, column=8, padx=5, pady=5, sticky='e')
        self.forecast_entry = ttk.Entry(input_frame, width=5)
        self.forecast_entry.grid(row=0, column=9, padx=5, pady=5)
        self.forecast_entry.insert(0, "7")

        ttk.Button(input_frame, text="Predict", command=self.run_prediction).grid(row=0, column=10, padx=10, pady=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(root, text="Prediction Results")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(8,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=output_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # ScrolledText for logs/results
        self.result_text = ScrolledText(output_frame, height=8)
        self.result_text.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        self.result_text.config(state=tk.DISABLED)

    def log(self, msg):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, msg + "\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)

    def run_prediction(self):
        # Clear previous
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete('1.0', tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.ax.cla()
        self.canvas.draw()

        # Get params
        ticker = self.ticker_entry.get().strip().upper()
        start_date = self.start_entry.get().strip()
        end_date = self.end_entry.get().strip()
        try:
            window_size = int(self.window_entry.get().strip())
            forecast_days = int(self.forecast_entry.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Window size and Forecast days must be integers.")
            return

        self.log(f"Fetching {ticker} data from {start_date} to {end_date}...")
        try:
            df = download_stock_data(ticker, start_date, end_date)
        except Exception as e:
            messagebox.showerror("Data Error", f"Failed to fetch data: {e}")
            return
        if len(df) < window_size + forecast_days + 10:
            messagebox.showerror("Data Error", f"Not enough data for the selected window size and forecast days.")
            return

        close_data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_close = scaler.fit_transform(close_data)

        X, y = create_sequences(scaled_close, window_size)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.log(f"Training LSTM model...")
        model = build_lstm_model(window_size)
        try:
            history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to train model: {e}")
            return

        self.log(f"Predicting on test set...")
        y_pred = model.predict(X_test, verbose=0)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
        y_pred_inv = scaler.inverse_transform(y_pred)

        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        self.log(f"RMSE: {rmse:.2f}")
        self.log(f"MAE: {mae:.2f}")

        # Plot
        self.ax.plot(y_test_inv, label='Actual')
        self.ax.plot(y_pred_inv, label='Predicted')
        self.ax.set_title(f"{ticker} Close Price Prediction")
        self.ax.set_ylabel("Price (USD)")
        self.ax.legend()
        self.canvas.draw()

        # Forecast next days
        self.log(f"\nForecast for next {forecast_days} days:")
        last_seq = scaled_close[-window_size:]
        preds_next_days = forecast_next_days(model, last_seq, scaler, n_days=forecast_days)
        for i, price in enumerate(preds_next_days, 1):
            self.log(f"Day {i}: ${price:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictorApp(root)
    root.mainloop()