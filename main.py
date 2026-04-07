tfrom datetime import timedelta
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Market Muse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore")

try:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.models import Sequential
except ImportError:
    try:
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from keras.layers import LSTM, Dense, Dropout, Input
        from keras.losses import Huber
        from keras.models import Sequential
    except ImportError:
        st.error("TensorFlow/Keras is required. Install dependencies with `pip install -r requirements.txt`.")
        st.stop()


FEATURE_COLUMNS = [
    "Close",
    "Return",
    "MA_7",
    "MA_21",
    "EMA_12",
    "Volatility_14",
    "Momentum_10",
    "RSI_14",
]
QUICK_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "NFLX", "Custom"]
TRAIN_RATIO_OPTIONS = {70: 0.70, 75: 0.75, 80: 0.80, 85: 0.85, 90: 0.90}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

            :root {
                --bg-wash: linear-gradient(180deg, #f6efe5 0%, #eef6f3 44%, #fbfcfd 100%);
                --panel: rgba(255, 255, 255, 0.84);
                --panel-strong: rgba(255, 255, 255, 0.95);
                --ink: #112433;
                --muted: #5d7282;
                --accent: #0f9d8d;
                --accent-deep: #0b6f78;
                --warm: #ff7a45;
                --border: rgba(17, 36, 51, 0.08);
                --shadow: 0 18px 60px rgba(17, 36, 51, 0.08);
            }

            .stApp {
                background: var(--bg-wash);
                color: var(--ink);
                font-family: "IBM Plex Sans", sans-serif;
            }

            [data-testid="stSidebar"] {
                background:
                    radial-gradient(circle at top, rgba(15, 157, 141, 0.18), transparent 28%),
                    linear-gradient(180deg, #102231 0%, #18364a 100%);
            }

            [data-testid="stSidebar"] * {
                color: #f3f6f8;
            }

            [data-testid="stSidebar"] .stButton > button,
            [data-testid="stSidebar"] .stDownloadButton > button {
                background: linear-gradient(135deg, #0f9d8d 0%, #0b6f78 100%);
                border: 0;
                color: white;
                font-weight: 600;
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            .hero-shell {
                background:
                    radial-gradient(circle at top right, rgba(15, 157, 141, 0.20), transparent 28%),
                    radial-gradient(circle at left, rgba(255, 122, 69, 0.12), transparent 22%),
                    linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,255,255,0.82));
                border: 1px solid var(--border);
                border-radius: 28px;
                box-shadow: var(--shadow);
                padding: 2.2rem 2rem;
                margin-bottom: 1.2rem;
            }

            .hero-title {
                font-family: "Space Grotesk", sans-serif;
                font-size: clamp(2.2rem, 3vw, 3.6rem);
                line-height: 1.05;
                margin: 0;
                color: var(--ink);
            }

            .hero-copy {
                color: var(--muted);
                font-size: 1.02rem;
                max-width: 720px;
                margin-top: 0.9rem;
            }

            .hero-pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.7rem;
                margin-top: 1.25rem;
            }

            .hero-pill {
                padding: 0.5rem 0.8rem;
                border-radius: 999px;
                background: rgba(17, 36, 51, 0.05);
                color: var(--ink);
                font-size: 0.92rem;
            }

            .section-card {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 24px;
                box-shadow: var(--shadow);
                padding: 1.25rem 1.25rem 1rem;
            }

            .step-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.85rem;
                margin-top: 1.1rem;
            }

            .step-card,
            .insight-card,
            .stat-card {
                background: var(--panel-strong);
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 1rem 1.05rem;
                box-shadow: 0 10px 28px rgba(17, 36, 51, 0.05);
            }

            .step-index {
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--accent-deep);
                font-weight: 700;
            }

            .step-title,
            .stat-title {
                color: var(--ink);
                font-weight: 700;
                margin-top: 0.25rem;
            }

            .step-copy,
            .stat-copy {
                color: var(--muted);
                font-size: 0.92rem;
                margin-top: 0.35rem;
            }

            .stat-value {
                font-family: "Space Grotesk", sans-serif;
                font-size: 2rem;
                color: var(--ink);
                margin-top: 0.2rem;
            }

            .tone-accent {
                border-top: 4px solid var(--accent);
            }

            .tone-warm {
                border-top: 4px solid var(--warm);
            }

            .tone-ink {
                border-top: 4px solid #163d57;
            }

            .insight-card {
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,250,249,0.96));
            }

            .insight-title {
                font-weight: 700;
                color: var(--ink);
            }

            .insight-copy {
                color: var(--muted);
                margin-top: 0.35rem;
                font-size: 0.94rem;
            }

            .mini-note {
                color: var(--muted);
                font-size: 0.88rem;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.4rem;
            }

            .stTabs [data-baseweb="tab"] {
                background: rgba(255, 255, 255, 0.65);
                border-radius: 999px;
                border: 1px solid var(--border);
                padding: 0.5rem 1rem;
            }

            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, rgba(15, 157, 141, 0.16), rgba(11, 111, 120, 0.08));
                color: var(--ink);
            }

            .stMetric {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 0.8rem 1rem;
            }

            .stDataFrame,
            [data-testid="stPlotlyChart"] {
                border-radius: 22px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data found for ticker {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    available_columns = [col for col in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in df.columns]
    df = df[available_columns]

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").set_index("Date")
    numeric_cols = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in df.columns]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def create_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy()
    feature_df["Return"] = feature_df["Close"].pct_change()
    feature_df["MA_7"] = feature_df["Close"].rolling(7).mean()
    feature_df["MA_21"] = feature_df["Close"].rolling(21).mean()
    feature_df["EMA_12"] = feature_df["Close"].ewm(span=12, adjust=False).mean()
    feature_df["Volatility_14"] = feature_df["Return"].rolling(14).std()
    feature_df["Momentum_10"] = feature_df["Close"].pct_change(10)
    feature_df["RSI_14"] = compute_rsi(feature_df["Close"], 14)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
    return feature_df


def create_sequences(features: np.ndarray, target: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i])
        y.append(target[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_lstm_model(window_size: int, feature_count: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(window_size, feature_count)),
            LSTM(96, return_sequences=True),
            Dropout(0.15),
            LSTM(64),
            Dropout(0.15),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss=Huber())
    return model


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray, reference_close: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    safe_actual = np.where(actual == 0, 1e-8, actual)
    mape = float(np.mean(np.abs((actual - predicted) / safe_actual)) * 100)

    actual_direction = np.sign(actual - reference_close)
    predicted_direction = np.sign(predicted - reference_close)
    directional_accuracy = float(np.mean(actual_direction == predicted_direction) * 100)

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
    }


def build_close_only_feature_frame(close_values: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame({"Close": pd.Series(close_values, dtype=np.float32)})
    frame["Return"] = frame["Close"].pct_change()
    frame["MA_7"] = frame["Close"].rolling(7).mean()
    frame["MA_21"] = frame["Close"].rolling(21).mean()
    frame["EMA_12"] = frame["Close"].ewm(span=12, adjust=False).mean()
    frame["Volatility_14"] = frame["Return"].rolling(14).std()
    frame["Momentum_10"] = frame["Close"].pct_change(10)
    frame["RSI_14"] = compute_rsi(frame["Close"], 14)
    frame = frame.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return frame[FEATURE_COLUMNS]


def forecast_future_prices(
    model: Sequential,
    close_history: np.ndarray,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    window_size: int,
    horizon: int,
) -> np.ndarray:
    evolving_close = close_history.astype(float).tolist()
    forecasts = []

    for _ in range(horizon):
        feature_window = build_close_only_feature_frame(evolving_close).tail(window_size)
        scaled_window = feature_scaler.transform(feature_window)
        scaled_pred = model.predict(scaled_window.reshape(1, window_size, len(FEATURE_COLUMNS)), verbose=0)
        next_close = float(target_scaler.inverse_transform(scaled_pred)[0, 0])
        forecasts.append(next_close)
        evolving_close.append(next_close)

    return np.array(forecasts, dtype=float)


def build_market_snapshot(df: pd.DataFrame, feature_df: pd.DataFrame) -> dict[str, float]:
    latest_close = float(df["Close"].iloc[-1])
    previous_close = float(df["Close"].iloc[-2])
    day_change_pct = ((latest_close / previous_close) - 1) * 100

    trailing_20_return = np.nan
    if len(df) >= 21:
        trailing_20_return = ((latest_close / float(df["Close"].iloc[-21])) - 1) * 100

    realized_vol = float(feature_df["Return"].tail(21).std() * np.sqrt(252) * 100) if len(feature_df) >= 21 else np.nan
    rsi = float(feature_df["RSI_14"].iloc[-1])
    avg_volume = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else np.nan
    ma_distance = ((latest_close / float(feature_df["MA_21"].iloc[-1])) - 1) * 100

    return {
        "latest_close": latest_close,
        "day_change_pct": day_change_pct,
        "trailing_20_return": trailing_20_return,
        "realized_vol": realized_vol,
        "rsi": rsi,
        "avg_volume": avg_volume,
        "ma_distance": ma_distance,
    }


def generate_insights(results: dict) -> list[tuple[str, str]]:
    snapshot = results["snapshot"]
    metrics = results["metrics_model"]
    baseline = results["metrics_baseline"]
    forecast_table = results["forecast_table"]
    latest_close = snapshot["latest_close"]
    end_forecast = float(forecast_table["Predicted Close"].iloc[-1])
    forecast_move = ((end_forecast / latest_close) - 1) * 100

    insights = []

    if metrics["rmse"] < baseline["rmse"]:
        edge = ((baseline["rmse"] - metrics["rmse"]) / baseline["rmse"]) * 100 if baseline["rmse"] else 0.0
        insights.append(
            (
                "Model edge over a naive baseline",
                f"The LSTM is beating a simple previous-close baseline by {edge:.1f}% on RMSE, which suggests it is learning more than momentum carryover.",
            )
        )
    else:
        insights.append(
            (
                "Baseline is still competitive",
                "The model is not decisively outperforming a naive baseline yet, so this run should be treated as exploratory rather than decision-grade.",
            )
        )

    if forecast_move >= 2:
        insights.append(
            (
                "Upside leaning forecast",
                f"The projected {len(forecast_table)}-day path implies a {forecast_move:.1f}% move higher from the latest close, with the model leaning constructive on near-term price action.",
            )
        )
    elif forecast_move <= -2:
        insights.append(
            (
                "Defensive near-term bias",
                f"The forecast path implies a {abs(forecast_move):.1f}% pullback from the latest close, which points to softer momentum in the near term.",
            )
        )
    else:
        insights.append(
            (
                "Range-bound near-term path",
                "The forecast stays fairly close to the latest close, which often happens when trend and volatility signals are mixed.",
            )
        )

    if snapshot["rsi"] >= 70:
        insights.append(
            (
                "Momentum is overheated",
                f"RSI is sitting at {snapshot['rsi']:.1f}, which usually signals an overbought regime where pullbacks become more likely.",
            )
        )
    elif snapshot["rsi"] <= 30:
        insights.append(
            (
                "Momentum is washed out",
                f"RSI is at {snapshot['rsi']:.1f}, which often lines up with oversold conditions and a potential stabilization zone.",
            )
        )
    else:
        insights.append(
            (
                "Momentum is balanced",
                f"RSI is at {snapshot['rsi']:.1f}, which places the stock in a more neutral momentum regime.",
            )
        )

    if snapshot["realized_vol"] >= 45:
        insights.append(
            (
                "Volatility is elevated",
                f"Realized volatility is running around {snapshot['realized_vol']:.1f}% annualized, so wide forecast bands should be expected.",
            )
        )
    else:
        insights.append(
            (
                "Volatility is moderate",
                f"Realized volatility is around {snapshot['realized_vol']:.1f}% annualized, which keeps the forecast band comparatively contained.",
            )
        )

    return insights


def render_hero(active_ticker: Optional[str] = None) -> None:
    pill_text = "Multisignal LSTM • Baseline Comparison • Forecast Bands • Narrative Insights"
    if active_ticker:
        pill_text = f"{active_ticker} active • Multisignal LSTM • Baseline Comparison • Forecast Bands"

    st.markdown(
        f"""
        <div class="hero-shell">
            <p class="mini-note">Market-aware forecasting workspace</p>
            <h1 class="hero-title">Market Muse</h1>
            <p class="hero-copy">
                A sharper stock forecasting dashboard with richer technical features, clearer evaluation,
                and a friendlier interface for exploring what the model sees before you trust the output.
            </p>
            <div class="hero-pill-row">
                <span class="hero-pill">{pill_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(title: str, value: str, caption: str, tone: str = "accent") -> None:
    st.markdown(
        f"""
        <div class="stat-card tone-{tone}">
            <div class="stat-title">{title}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-copy">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_card(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_large_number(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:,.0f}"


def make_price_volume_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    recent = df.tail(120).copy()
    recent["MA_21"] = recent["Close"].rolling(21).mean()
    recent["MA_50"] = recent["Close"].rolling(50).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    fig.add_trace(
        go.Candlestick(
            x=recent.index,
            open=recent["Open"],
            high=recent["High"],
            low=recent["Low"],
            close=recent["Close"],
            name="Price",
            increasing_line_color="#0f9d8d",
            decreasing_line_color="#ff7a45",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["MA_21"],
            mode="lines",
            name="21D MA",
            line=dict(color="#163d57", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["MA_50"],
            mode="lines",
            name="50D MA",
            line=dict(color="#ff7a45", width=2, dash="dot"),
        ),
        row=1,
        col=1,
    )

    volume_colors = np.where(recent["Close"] >= recent["Open"], "#0f9d8d", "#ff7a45")
    fig.add_trace(
        go.Bar(
            x=recent.index,
            y=recent["Volume"],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.55,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{ticker} price action and liquidity",
        template="plotly_white",
        height=620,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", y=1.05, x=0),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def make_test_prediction_chart(test_frame: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=test_frame["Date"],
            y=test_frame["Actual"],
            mode="lines",
            name="Actual",
            line=dict(color="#163d57", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_frame["Date"],
            y=test_frame["Predicted"],
            mode="lines",
            name="LSTM",
            line=dict(color="#0f9d8d", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_frame["Date"],
            y=test_frame["Baseline"],
            mode="lines",
            name="Naive baseline",
            line=dict(color="#ff7a45", width=1.7, dash="dot"),
        )
    )
    fig.update_layout(
        title=f"{ticker} out-of-sample prediction window",
        template="plotly_white",
        height=430,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="Close (USD)")
    return fig


def make_forecast_chart(raw_df: pd.DataFrame, forecast_table: pd.DataFrame, ticker: str) -> go.Figure:
    history = raw_df["Close"].tail(60)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Recent close",
            line=dict(color="#163d57", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_table["Date"],
            y=forecast_table["Upper Band"],
            mode="lines",
            line=dict(color="rgba(15,157,141,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_table["Date"],
            y=forecast_table["Lower Band"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(15, 157, 141, 0.16)",
            line=dict(color="rgba(15,157,141,0)"),
            name="Confidence band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_table["Date"],
            y=forecast_table["Predicted Close"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#0f9d8d", width=3),
            marker=dict(size=7, color="#0f9d8d"),
        )
    )
    fig.update_layout(
        title=f"{ticker} forecast path",
        template="plotly_white",
        height=460,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="Close (USD)")
    return fig


def make_history_chart(history_frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_frame["Epoch"],
            y=history_frame["Training Loss"],
            mode="lines",
            name="Training",
            line=dict(color="#163d57", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history_frame["Epoch"],
            y=history_frame["Validation Loss"],
            mode="lines",
            name="Validation",
            line=dict(color="#ff7a45", width=2),
        )
    )
    fig.update_layout(
        title="Training history",
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_yaxes(title_text="Loss")
    return fig


def make_metric_comparison_chart(model_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> go.Figure:
    categories = ["RMSE", "MAE", "MAPE", "Directional Acc."]
    model_values = [
        model_metrics["rmse"],
        model_metrics["mae"],
        model_metrics["mape"],
        model_metrics["directional_accuracy"],
    ]
    baseline_values = [
        baseline_metrics["rmse"],
        baseline_metrics["mae"],
        baseline_metrics["mape"],
        baseline_metrics["directional_accuracy"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=model_values, name="LSTM", marker_color="#0f9d8d"))
    fig.add_trace(go.Bar(x=categories, y=baseline_values, name="Baseline", marker_color="#ff7a45"))
    fig.update_layout(
        title="Model versus baseline",
        template="plotly_white",
        barmode="group",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_residual_chart(test_frame: pd.DataFrame) -> go.Figure:
    residuals = test_frame["Actual"] - test_frame["Predicted"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=test_frame["Date"],
            y=residuals,
            mode="lines",
            line=dict(color="#163d57", width=2),
            name="Residual",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#ff7a45")
    fig.update_layout(
        title="Residual drift over time",
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_yaxes(title_text="Actual - Predicted")
    return fig


def run_analysis(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    window_size: int,
    forecast_days: int,
    epochs: int,
    train_ratio: float,
) -> dict:
    raw_df = download_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    feature_df = create_feature_frame(raw_df)

    required_rows = max(window_size + 50, 140)
    if len(feature_df) < required_rows:
        raise ValueError(
            f"Not enough processed data for the selected setup. Need at least {required_rows} rows after feature engineering, found {len(feature_df)}."
        )

    train_cutoff = int(len(feature_df) * train_ratio)
    train_cutoff = max(train_cutoff, window_size + 20)
    train_cutoff = min(train_cutoff, len(feature_df) - 20)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(feature_df[FEATURE_COLUMNS].iloc[:train_cutoff])
    target_scaler.fit(feature_df[["Close"]].iloc[:train_cutoff])

    scaled_features = feature_scaler.transform(feature_df[FEATURE_COLUMNS])
    scaled_target = target_scaler.transform(feature_df[["Close"]]).reshape(-1)

    X, y = create_sequences(scaled_features, scaled_target, window_size)
    dates = feature_df.index[window_size:]
    actual_close = feature_df["Close"].iloc[window_size:].to_numpy()
    previous_close = feature_df["Close"].shift(1).iloc[window_size:].to_numpy()

    split_idx = int(len(X) * train_ratio)
    min_test = max(12, forecast_days)
    split_idx = max(split_idx, 24)
    split_idx = min(split_idx, len(X) - min_test)
    if split_idx <= 20 or len(X) - split_idx < min_test:
        raise ValueError("The selected date range is too small for a stable train/test split. Try expanding the range.")

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    val_size = max(int(len(X_train) * 0.15), 8)
    val_size = min(val_size, len(X_train) - 8)
    X_fit, X_val = X_train[:-val_size], X_train[-val_size:]
    y_fit, y_val = y_train[:-val_size], y_train[-val_size:]

    model = build_lstm_model(window_size, len(FEATURE_COLUMNS))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5),
    ]

    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )

    pred_scaled = model.predict(X_test, verbose=0)
    predicted_close = target_scaler.inverse_transform(pred_scaled).flatten()
    actual_test_close = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    previous_test_close = previous_close[split_idx:]
    baseline_close = previous_test_close.copy()

    metrics_model = calculate_metrics(actual_test_close, predicted_close, previous_test_close)
    metrics_baseline = calculate_metrics(actual_test_close, baseline_close, previous_test_close)
    residual_std = float(np.std(actual_test_close - predicted_close, ddof=1)) if len(actual_test_close) > 1 else 0.0

    future_close = forecast_future_prices(
        model=model,
        close_history=feature_df["Close"].to_numpy(),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        window_size=window_size,
        horizon=forecast_days,
    )
    future_dates = pd.bdate_range(start=feature_df.index[-1] + pd.offsets.BDay(1), periods=forecast_days)
    confidence_width = residual_std * np.sqrt(np.arange(1, forecast_days + 1))

    forecast_table = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted Close": future_close,
            "Lower Band": future_close - confidence_width,
            "Upper Band": future_close + confidence_width,
        }
    )
    forecast_table["Move vs Latest %"] = ((forecast_table["Predicted Close"] / feature_df["Close"].iloc[-1]) - 1) * 100

    test_frame = pd.DataFrame(
        {
            "Date": dates[split_idx:],
            "Actual": actual_test_close,
            "Predicted": predicted_close,
            "Baseline": baseline_close,
            "Previous Close": previous_test_close,
        }
    )

    history_frame = pd.DataFrame(
        {
            "Epoch": list(range(1, len(history.history["loss"]) + 1)),
            "Training Loss": history.history["loss"],
            "Validation Loss": history.history.get("val_loss", [np.nan] * len(history.history["loss"])),
        }
    )

    snapshot = build_market_snapshot(raw_df, feature_df)

    return {
        "ticker": ticker,
        "raw_df": raw_df,
        "feature_df": feature_df,
        "test_frame": test_frame,
        "forecast_table": forecast_table,
        "history_frame": history_frame,
        "metrics_model": metrics_model,
        "metrics_baseline": metrics_baseline,
        "residual_std": residual_std,
        "snapshot": snapshot,
        "params": {
            "window_size": window_size,
            "forecast_days": forecast_days,
            "epochs": epochs,
            "train_ratio": train_ratio,
            "start_date": start_date,
            "end_date": end_date,
        },
    }


inject_styles()

today = pd.Timestamp.today().date()
default_start = today - timedelta(days=365 * 5)

with st.sidebar:
    st.markdown("## Forecast Controls")
    st.caption("Pick a ticker, set the memory window, then launch a richer LSTM analysis run.")

    with st.form("control_form"):
        preset_ticker = st.selectbox("Quick ticker", QUICK_TICKERS, index=0)
        if preset_ticker == "Custom":
            ticker_value = st.text_input("Custom ticker", value="AAPL", help="Examples: AAPL, MSFT, TCS.NS")
        else:
            ticker_value = preset_ticker

        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input("Start", value=default_start)
        with date_col2:
            end_date = st.date_input("End", value=today)

        st.markdown("### Model Setup")
        window_size = st.slider("Lookback window", min_value=20, max_value=120, value=60, step=5)
        forecast_days = st.slider("Forecast horizon", min_value=3, max_value=30, value=10)
        epochs = st.slider("Training epochs", min_value=10, max_value=80, value=30, step=5)
        train_ratio_key = st.select_slider("Train split", options=list(TRAIN_RATIO_OPTIONS.keys()), value=80)

        submitted = st.form_submit_button("Run Forecast Suite", use_container_width=True)

    st.markdown("---")
    st.markdown("### What changed")
    st.markdown(
        """
        - Richer technical signals instead of close-only training
        - Naive baseline comparison for context
        - Confidence bands and narrative insights
        - Cleaner workflow across overview, diagnostics, and forecast tabs
        """
    )
    st.caption("This is an educational forecasting workspace, not financial advice.")


if submitted:
    if not ticker_value.strip():
        st.error("Enter a valid ticker symbol before running the analysis.")
    elif start_date >= end_date:
        st.error("The start date needs to be earlier than the end date.")
    else:
        progress = st.progress(0)
        status = st.empty()
        try:
            status.info("Pulling market data and engineering technical features...")
            progress.progress(20)

            results = run_analysis(
                ticker=ticker_value.strip().upper(),
                start_date=pd.Timestamp(start_date),
                end_date=pd.Timestamp(end_date) + pd.Timedelta(days=1),
                window_size=window_size,
                forecast_days=forecast_days,
                epochs=epochs,
                train_ratio=TRAIN_RATIO_OPTIONS[train_ratio_key],
            )

            progress.progress(80)
            results["insights"] = generate_insights(results)
            st.session_state["analysis_result"] = results
            progress.progress(100)
            status.success("Analysis ready. Explore the tabs below for the market snapshot, diagnostics, and forecast desk.")
        except Exception as exc:
            st.session_state.pop("analysis_result", None)
            status.error(f"Unable to run the forecast suite: {exc}")
        finally:
            progress.empty()


results = st.session_state.get("analysis_result")
render_hero(results["ticker"] if results else None)

if results:
    snapshot = results["snapshot"]
    metrics_model = results["metrics_model"]
    metrics_baseline = results["metrics_baseline"]
    forecast_table = results["forecast_table"]
    latest_forecast = float(forecast_table["Predicted Close"].iloc[-1])
    latest_move = ((latest_forecast / snapshot["latest_close"]) - 1) * 100
    edge = metrics_baseline["rmse"] - metrics_model["rmse"]

    top_cols = st.columns(4)
    with top_cols[0]:
        render_stat_card(
            "Latest close",
            f"${snapshot['latest_close']:.2f}",
            f"{snapshot['day_change_pct']:+.2f}% vs previous session",
            "accent",
        )
    with top_cols[1]:
        render_stat_card(
            "20-day move",
            f"{snapshot['trailing_20_return']:+.2f}%",
            f"21D MA distance: {snapshot['ma_distance']:+.2f}%",
            "ink",
        )
    with top_cols[2]:
        render_stat_card(
            "Model edge",
            f"${edge:.2f}",
            "Positive means the model cut RMSE versus the baseline",
            "warm",
        )
    with top_cols[3]:
        render_stat_card(
            f"{len(forecast_table)}-day view",
            f"{latest_move:+.2f}%",
            f"Forecast end price: ${latest_forecast:.2f}",
            "accent",
        )

    tabs = st.tabs(["Overview", "Diagnostics", "Forecast Desk"])

    with tabs[0]:
        overview_col1, overview_col2 = st.columns([1.9, 1.1])
        with overview_col1:
            st.plotly_chart(make_price_volume_chart(results["raw_df"], results["ticker"]), use_container_width=True)
        with overview_col2:
            render_insight_card("Momentum read", f"RSI(14) is at {snapshot['rsi']:.1f}, which helps frame whether recent price action looks stretched or balanced.")
            render_insight_card("Volatility pulse", f"Realized volatility is running near {snapshot['realized_vol']:.1f}% annualized, giving you a sense of how noisy the current regime is.")
            render_insight_card("Liquidity check", f"Average 20-day volume is about {format_large_number(snapshot['avg_volume'])} shares, which is useful context for how clean the tape usually trades.")
            render_insight_card("Training setup", f"The model used a {results['params']['window_size']}-day memory window and trained for up to {results['params']['epochs']} epochs with adaptive early stopping.")

        st.plotly_chart(make_forecast_chart(results["raw_df"], forecast_table, results["ticker"]), use_container_width=True)

        insight_cols = st.columns(2)
        for index, (title, copy) in enumerate(results["insights"][:4]):
            with insight_cols[index % 2]:
                render_insight_card(title, copy)

    with tabs[1]:
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("RMSE", f"${metrics_model['rmse']:.2f}", f"{metrics_baseline['rmse'] - metrics_model['rmse']:+.2f} vs baseline")
        with metric_cols[1]:
            st.metric("MAE", f"${metrics_model['mae']:.2f}", f"{metrics_baseline['mae'] - metrics_model['mae']:+.2f} vs baseline")
        with metric_cols[2]:
            st.metric("MAPE", f"{metrics_model['mape']:.2f}%", f"{metrics_baseline['mape'] - metrics_model['mape']:+.2f} pts")
        with metric_cols[3]:
            st.metric("Directional Acc.", f"{metrics_model['directional_accuracy']:.1f}%", f"{metrics_model['directional_accuracy'] - metrics_baseline['directional_accuracy']:+.1f} pts")

        chart_col1, chart_col2 = st.columns([1.5, 1])
        with chart_col1:
            st.plotly_chart(make_test_prediction_chart(results["test_frame"], results["ticker"]), use_container_width=True)
        with chart_col2:
            st.plotly_chart(make_metric_comparison_chart(metrics_model, metrics_baseline), use_container_width=True)

        lower_chart_col1, lower_chart_col2 = st.columns(2)
        with lower_chart_col1:
            st.plotly_chart(make_history_chart(results["history_frame"]), use_container_width=True)
        with lower_chart_col2:
            st.plotly_chart(make_residual_chart(results["test_frame"]), use_container_width=True)

    with tabs[2]:
        desk_col1, desk_col2 = st.columns([1.25, 1])
        with desk_col1:
            styled_table = forecast_table.copy()
            styled_table["Date"] = styled_table["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                styled_table.style.format(
                    {
                        "Predicted Close": "${:.2f}",
                        "Lower Band": "${:.2f}",
                        "Upper Band": "${:.2f}",
                        "Move vs Latest %": "{:+.2f}%",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download forecast CSV",
                data=forecast_table.to_csv(index=False),
                file_name=f"{results['ticker']}_forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with desk_col2:
            for title, copy in results["insights"]:
                render_insight_card(title, copy)

            st.markdown(
                """
                <div class="section-card" style="margin-top: 0.9rem;">
                    <div class="stat-title">How to read the forecast band</div>
                    <div class="stat-copy">
                        The upper and lower bands are heuristic ranges built from test-set residual dispersion.
                        They are useful for scenario framing, not guarantees.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

else:
    st.markdown(
        """
        <div class="section-card">
            <div class="stat-title">A better default workflow</div>
            <div class="stat-copy">
                This upgrade turns the project into a guided forecasting workspace instead of a single chart and a button.
                Start with a ticker in the sidebar, run the suite, then use the tabs to move from market context to model diagnostics to the forecast desk.
            </div>
            <div class="step-grid">
                <div class="step-card">
                    <div class="step-index">Step 1</div>
                    <div class="step-title">Choose the setup</div>
                    <div class="step-copy">Pick a ticker, set the date range, and adjust the memory window based on how much history you want the model to learn from.</div>
                </div>
                <div class="step-card">
                    <div class="step-index">Step 2</div>
                    <div class="step-title">Run the forecast suite</div>
                    <div class="step-copy">The app engineers technical signals, trains a multivariate LSTM, and compares it against a naive baseline.</div>
                </div>
                <div class="step-card">
                    <div class="step-index">Step 3</div>
                    <div class="step-title">Read the story</div>
                    <div class="step-copy">Use the overview for context, diagnostics for trust, and the forecast desk for a clean exportable outlook.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("---")
st.caption("Built with Streamlit, TensorFlow, Plotly, and Yahoo Finance. Educational use only.")
