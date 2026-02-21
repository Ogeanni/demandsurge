"""
models/demand_forecast.py
Prophet-based demand forecasting for ShopMind.

Trains one Prophet model per product category using Google Trends data
as a demand proxy, then generates 30-day forecasts.

Usage:
    python models/demand_forecast.py                  # Train all categories
    python models/demand_forecast.py --category electronics
    python models/demand_forecast.py --forecast-only  # Skip training, load saved models
    python models/demand_forecast.py --preview        # Show forecast plots

Output:
    models/prophet_{category}.pkl   — trained Prophet model per category
    data/forecasts_{category}.csv   — forecast output with confidence bounds
"""

import os
import sys
import json
import pickle
import logging
import argparse
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Stan.*")

# ── Path setup ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demand_forecast")

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
except ImportError:
    log.error("prophet not installed. Run: pip install prophet")
    sys.exit(1)

from db.models import get_db, DemandSignal

# ── Paths ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT, "models/prophet/saved_models")
DATA_DIR   = os.path.join(ROOT, "results/prophet")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# ── Category → primary keyword mapping ───────────────────────────────
# Must match CATEGORY_KEYWORD in features.py
CATEGORY_KEYWORD = {
    "electronics": "wireless headphones",
    "fashion":     "leather wallet",
    "home_goods":  "bamboo cutting board",
    "sports":      "yoga mat",
}

FORECAST_DAYS    = 30    # How many days ahead to forecast
MIN_WEEKS_NEEDED = 12    # Minimum weeks of data to train a meaningful model


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_keyword_data(keyword: str) -> pd.DataFrame:
    """
    Loads demand signal rows for a keyword from the DB.
    Returns a DataFrame with columns [ds, y] as required by Prophet.
    ds = date, y = trend_index (0-100)
    """
    with get_db() as db:
        rows = (
            db.query(DemandSignal)
            .filter(DemandSignal.keyword == keyword)
            .order_by(DemandSignal.week_date)
            .all()
        )

    if not rows:
        return pd.DataFrame(columns=["ds", "y"])

    df = pd.DataFrame([{
        "ds": pd.Timestamp(r.week_date),
        "y":  float(r.trend_index),
    } for r in rows])

    # Prophet requires ds to be datetime, y to be float
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"]  = df["y"].astype(float)

    # Remove duplicates — keep last if same date appears twice
    df = df.drop_duplicates(subset="ds", keep="last").reset_index(drop=True)

    log.info(f"  Loaded {len(df)} weeks for '{keyword}' "
             f"({df['ds'].min().date()} → {df['ds'].max().date()})")
    return df


def make_synthetic_data(keyword: str, n_weeks: int = 104) -> pd.DataFrame:
    """
    Generates realistic synthetic demand data when no DB data exists.
    Uses category-appropriate seasonal patterns so the model is still meaningful.

    This runs automatically when demand_signals is empty (e.g. before
    Pytrends has been fetched). Remove this fallback once real data is in.
    """
    log.warning(f"  No DB data for '{keyword}' — generating synthetic data for training")

    np.random.seed(abs(hash(keyword)) % 2**31)
    dates = pd.date_range(end=date.today(), periods=n_weeks, freq="W")

    # Base index with gentle upward trend
    base   = 45 + np.linspace(0, 10, n_weeks)

    # Annual seasonality (peak varies by category keyword)
    peaks  = {
        "wireless headphones": 11,   # Nov (holiday electronics)
        "bluetooth speaker":   7,    # Jul (summer)
        "smart watch":         11,
        "mechanical keyboard": 11,
        "leather wallet":      11,
        "tote bag":            6,    # Jun (summer fashion)
        "minimalist watch":    4,    # Apr (spring)
        "wool beanie":         10,   # Oct (fall)
        "bamboo cutting board":11,
        "stainless steel water bottle": 6,
        "soy candle":          11,
        "essential oil diffuser": 11,
        "yoga mat":            1,    # Jan (new year fitness)
        "resistance bands":    1,
        "running shoes":       3,    # Mar (spring)
        "foam roller":         1,
    }
    peak_month = peaks.get(keyword, 11)
    t          = np.arange(n_weeks)
    seasonal   = 18 * np.sin(2 * np.pi * (t / 52 - (peak_month - 1) / 12))

    # Short-term noise
    noise = np.random.normal(0, 4, n_weeks)

    y = np.clip(base + seasonal + noise, 0, 100).round(0)

    return pd.DataFrame({"ds": dates, "y": y})


# ══════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════

def build_prophet_model(category: str) -> Prophet:
    """
    Builds and configures a Prophet model for a given category.
    Tuned for weekly Google Trends data (not daily sales).
    """
    model = Prophet(
        # Seasonality
        yearly_seasonality=True,
        weekly_seasonality=False,   # Trends data is weekly — no sub-weekly pattern
        daily_seasonality=False,

        # Trend flexibility
        # changepoint_prior_scale: higher = more flexible trend (0.05 is conservative)
        changepoint_prior_scale=0.08,

        # Seasonality flexibility
        seasonality_prior_scale=12.0,

        # Uncertainty interval width (80% = tighter, good for pricing decisions)
        interval_width=0.80,

        # Cap prediction noise for stable pricing recommendations
        seasonality_mode="additive",
    )

    # US retail holidays — picks up Black Friday, Christmas, New Year etc.
    model.add_country_holidays(country_name="US")

    # Quarterly retail seasonality (back-to-school, end-of-quarter)
    model.add_seasonality(
        name="quarterly",
        period=91.25,
        fourier_order=5,
    )

    # Category-specific extra seasonality
    if category in ("electronics", "home_goods"):
        # Strong holiday spike Nov-Dec
        model.add_seasonality(name="holiday_spike", period=365.25 / 6, fourier_order=3)
    elif category == "sports":
        # Jan fitness peak + summer peak
        model.add_seasonality(name="fitness_cycle", period=365.25 / 2, fourier_order=3)
    elif category == "fashion":
        # Spring and fall fashion seasons
        model.add_seasonality(name="fashion_season", period=365.25 / 2, fourier_order=4)

    return model


def train_category(category: str, keyword: str) -> dict:
    """
    Trains a Prophet model for one category keyword.
    Returns a result dict with model, forecast, and metrics.
    """
    log.info(f"\n{'='*55}")
    log.info(f"Category: {category.upper()}  |  Keyword: '{keyword}'")
    log.info(f"{'='*55}")

    # ── Load data ─────────────────────────────────────────────────────
    df = load_keyword_data(keyword)

    if len(df) < MIN_WEEKS_NEEDED:
        log.warning(f"  Only {len(df)} weeks — need {MIN_WEEKS_NEEDED} minimum. "
                    f"Using synthetic data.")
        df = make_synthetic_data(keyword)

    log.info(f"  Training on {len(df)} weeks of data")

    # ── Train/test split for eval ─────────────────────────────────────
    # Hold out last 4 weeks to evaluate accuracy
    split_idx  = max(len(df) - 4, int(len(df) * 0.85))
    train_df   = df.iloc[:split_idx].copy()
    test_df    = df.iloc[split_idx:].copy()

    log.info(f"  Train: {len(train_df)} weeks  |  Test: {len(test_df)} weeks")

    # ── Fit model ─────────────────────────────────────────────────────
    log.info("  Fitting Prophet model...")
    model = build_prophet_model(category)

    # Suppress Prophet's verbose Stan output
    import logging as _logging
    _logging.getLogger("prophet").setLevel(_logging.WARNING)
    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

    model.fit(train_df)
    log.info("  Model fitted")

    # ── Evaluate on held-out test weeks ──────────────────────────────
    test_forecast = model.predict(test_df[["ds"]])
    test_merged   = test_df.merge(test_forecast[["ds", "yhat"]], on="ds")
    mae  = float(np.mean(np.abs(test_merged["y"] - test_merged["yhat"])))
    rmse = float(np.sqrt(np.mean((test_merged["y"] - test_merged["yhat"]) ** 2)))
    log.info(f"  Eval on {len(test_merged)} test weeks — MAE: {mae:.2f}  RMSE: {rmse:.2f}")

    # ── Generate forecast ─────────────────────────────────────────────
    log.info(f"  Generating {FORECAST_DAYS}-day forecast...")

    # Build a fresh model and fit on full data (Prophet can only be fit once)
    model = build_prophet_model(category)
    model.fit(df)

    future   = model.make_future_dataframe(periods=FORECAST_DAYS, freq="D")
    forecast = model.predict(future)

    # Clip predictions to valid trend index range [0, 100]
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(0, 100).round(2)

    # ── Save model ────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, f"prophet_{category}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"  Model saved → {model_path}")

    # ── Save forecast CSV ─────────────────────────────────────────────
    forecast_path = os.path.join(DATA_DIR, f"forecasts_{category}.csv")
    forecast_out  = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_out["category"] = category
    forecast_out["keyword"]  = keyword
    forecast_out.to_csv(forecast_path, index=False)
    log.info(f"  Forecast saved → {forecast_path}")

    # ── Print upcoming 30-day window ──────────────────────────────────
    upcoming = forecast[forecast["ds"] > df["ds"].max()].head(30)
    if not upcoming.empty:
        avg_upcoming   = upcoming["yhat"].mean()
        current_latest = float(df["y"].iloc[-1])
        trend_dir      = "↑ rising" if avg_upcoming > current_latest else "↓ falling"
        log.info(f"  30-day outlook: avg index={avg_upcoming:.1f}  "
                 f"(current={current_latest:.0f})  {trend_dir}")
        log.info(f"  Confidence band: "
                 f"{upcoming['yhat_lower'].mean():.1f} – {upcoming['yhat_upper'].mean():.1f}")

    return {
        "category":    category,
        "keyword":     keyword,
        "n_weeks":     len(df),
        "mae":         round(mae, 2),
        "rmse":        round(rmse, 2),
        "model_path":  model_path,
        "model":       model,
        "forecast":    forecast,
    }


# ══════════════════════════════════════════════════════════════════════
# INFERENCE — called by LangChain agent tools
# ══════════════════════════════════════════════════════════════════════

def load_model(category: str) -> Prophet:
    """Loads a trained Prophet model from disk."""
    model_path = os.path.join(MODELS_DIR, f"prophet_{category}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model for '{category}'. "
            f"Run: python models/demand_forecast.py --category {category}"
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_demand_forecast(category: str, days: int = 30) -> dict:
    """
    Public API — called by LangChain agent tool in Block 8.

    Returns:
    {
        "category":       str,
        "keyword":        str,
        "current_index":  float,   # latest known trend index
        "forecast_avg":   float,   # average predicted index over next N days
        "forecast_high":  float,   # peak predicted index
        "forecast_low":   float,   # trough predicted index
        "trend_direction":str,     # "rising" | "falling" | "stable"
        "confidence_low": float,   # lower bound average
        "confidence_high":float,   # upper bound average
        "demand_signal":  str,     # human-readable summary for the agent
        "forecast_df":    DataFrame  # full forecast for charting
    }
    """
    keyword = CATEGORY_KEYWORD.get(category)
    if not keyword:
        raise ValueError(f"Unknown category: '{category}'. "
                         f"Valid: {list(CATEGORY_KEYWORD.keys())}")

    model = load_model(category)

    # Load recent actuals for context
    actuals = load_keyword_data(keyword)
    if actuals.empty:
        actuals = make_synthetic_data(keyword)

    current_index = float(actuals["y"].iloc[-1]) if not actuals.empty else 50.0

    # Generate forecast
    future   = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future)

    # Clip to valid range
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(0, 100)

    # Focus on the forward-looking window only
    last_actual_date = actuals["ds"].max() if not actuals.empty else pd.Timestamp.now()
    future_fc = forecast[forecast["ds"] > last_actual_date].head(days)

    if future_fc.empty:
        future_fc = forecast.tail(days)

    avg_forecast  = float(future_fc["yhat"].mean())
    high_forecast = float(future_fc["yhat"].max())
    low_forecast  = float(future_fc["yhat"].min())
    conf_low      = float(future_fc["yhat_lower"].mean())
    conf_high     = float(future_fc["yhat_upper"].mean())

    # Trend direction
    delta = avg_forecast - current_index
    if delta > 5:
        direction = "rising"
    elif delta < -5:
        direction = "falling"
    else:
        direction = "stable"

    # Build human-readable demand signal for the agent
    if direction == "rising":
        signal_text = (
            f"Demand for {category} ({keyword}) is expected to RISE over the next {days} days. "
            f"Trend index: currently {current_index:.0f}/100, "
            f"forecasted avg {avg_forecast:.0f}/100 (peak {high_forecast:.0f}). "
            f"Rising demand supports pricing at or above market median."
        )
    elif direction == "falling":
        signal_text = (
            f"Demand for {category} ({keyword}) is expected to FALL over the next {days} days. "
            f"Trend index: currently {current_index:.0f}/100, "
            f"forecasted avg {avg_forecast:.0f}/100 (low {low_forecast:.0f}). "
            f"Falling demand suggests pricing competitively to maintain volume."
        )
    else:
        signal_text = (
            f"Demand for {category} ({keyword}) is expected to remain STABLE "
            f"over the next {days} days. "
            f"Trend index: currently {current_index:.0f}/100, "
            f"forecasted avg {avg_forecast:.0f}/100. "
            f"Stable demand — price near market median."
        )

    return {
        "category":        category,
        "keyword":         keyword,
        "current_index":   round(current_index, 1),
        "forecast_avg":    round(avg_forecast, 1),
        "forecast_high":   round(high_forecast, 1),
        "forecast_low":    round(low_forecast, 1),
        "trend_direction": direction,
        "confidence_low":  round(conf_low, 1),
        "confidence_high": round(conf_high, 1),
        "demand_signal":   signal_text,
        "forecast_df":     future_fc,
    }


# ══════════════════════════════════════════════════════════════════════
# OPTIONAL PLOT
# ══════════════════════════════════════════════════════════════════════

def plot_forecast(result: dict):
    """Saves a forecast plot to data/forecast_{category}.png"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")
        return

    model    = result["model"]
    forecast = result["forecast"]
    category = result["category"]

    fig = model.plot(forecast, figsize=(12, 5))
    plt.title(f"ShopMind Demand Forecast — {category.upper()} ({result['keyword']})")
    plt.xlabel("Date")
    plt.ylabel("Trend Index (0-100)")
    plt.tight_layout()

    plot_path = os.path.join(DATA_DIR, f"forecast_{category}.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    log.info(f"  Plot saved → {plot_path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Prophet demand models for ShopMind")
    parser.add_argument(
        "--category", "-c",
        choices=list(CATEGORY_KEYWORD.keys()),
        default=None,
        help="Train a single category (default: all)",
    )
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Skip training — load saved models and generate fresh forecasts",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save forecast plots to data/",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=FORECAST_DAYS,
        help=f"Days to forecast ahead (default: {FORECAST_DAYS})",
    )
    args = parser.parse_args()

    categories = (
        [args.category] if args.category
        else list(CATEGORY_KEYWORD.keys())
    )

    log.info("ShopMind — Prophet Demand Forecasting")
    log.info(f"Categories    : {categories}")
    log.info(f"Forecast days : {args.days}")
    log.info(f"Forecast only : {args.forecast_only}")

    results = []

    if args.forecast_only:
        # Load saved models and run inference only
        for category in categories:
            keyword = CATEGORY_KEYWORD[category]
            log.info(f"\nLoading saved model for '{category}'...")
            try:
                fc = get_demand_forecast(category, days=args.days)
                log.info(f"  {category}: {fc['trend_direction']}  "
                         f"(current={fc['current_index']}, "
                         f"forecast avg={fc['forecast_avg']})")
                log.info(f"  Signal: {fc['demand_signal']}")
                results.append({"category": category, "status": "ok"})
            except FileNotFoundError as e:
                log.error(f"  {e}")
                results.append({"category": category, "status": "missing_model"})
    else:
        # Full training run
        for category in categories:
            keyword = CATEGORY_KEYWORD.get(category)
            try:
                result = train_category(category, keyword)
                results.append({
                    "category": category,
                    "status":   "ok",
                    "mae":      result["mae"],
                    "rmse":     result["rmse"],
                })
                if args.preview:
                    plot_forecast(result)
            except Exception as e:
                log.error(f"  Failed for '{category}': {e}")
                results.append({"category": category, "status": "error", "error": str(e)})

    # ── Summary ───────────────────────────────────────────────────────
    log.info("\n" + "="*55)
    log.info("SUMMARY")
    log.info("="*55)
    for r in results:
        if r["status"] == "ok" and "mae" in r:
            log.info(f"  [OK] {r['category']:<15}  MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}")
        elif r["status"] == "ok":
            log.info(f"  [OK] {r['category']}")
        else:
            log.info(f"  [!!] {r['category']:<15}  {r.get('error', r['status'])}")

    # List saved model files
    log.info("\nSaved models:")
    for category in categories:
        path = os.path.join(MODELS_DIR, f"prophet_{category}.pkl")
        size = f"{os.path.getsize(path)/1024:.0f}KB" if os.path.exists(path) else "MISSING"
        log.info(f"  {path}  ({size})")

    errors = [r for r in results if r["status"] != "ok"]
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())