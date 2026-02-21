"""
models/features.py
Feature engineering for DemandSurge's ML pricing model.

Builds a feature matrix by joining:
  - products          (base info, current price, inventory)
  - price_history     (rolling averages, price momentum)
  - demand_signals    (Google Trends index, slope, seasonality)
  - competitor_prices (market median, price position, bands)
  + time features     (month, retail events, days-until-holiday)

Usage:
    python models/features.py               # Build and save features.parquet
    python models/features.py --preview     # Print feature matrix summary
    python models/features.py --validate    # Run data quality checks only

Output:
    data/features.parquet   — feature matrix ready for XGBoost
    data/features_meta.json — column types and feature descriptions
"""

import os
import sys
import json
import logging
import argparse
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

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
log = logging.getLogger("features")

from db.models import get_db, Product, PriceHistory, DemandSignal, CompetitorPrice

# ── Output paths ──────────────────────────────────────────────────────
DATA_DIR     = os.path.join(ROOT, "data/processed")
PARQUET_PATH = os.path.join(DATA_DIR, "features.parquet")
META_PATH    = os.path.join(DATA_DIR, "features_meta.json")
os.makedirs(DATA_DIR, exist_ok=True)


# ── Category → Pytrends keyword mapping ──────────────────────────────
# Maps each product category to its primary demand keyword.
# These must match what was fetched in fetch_trends.py.
CATEGORY_KEYWORD = {
    "electronics": "wireless headphones",
    "fashion":     "leather wallet",
    "home_goods":  "bamboo cutting board",
    "sports":      "yoga mat",
}

# ── Retail event calendar ─────────────────────────────────────────────
# (month, day) tuples for major retail events
RETAIL_EVENTS = {
    "black_friday":  (11, 29),   # approximate — last Fri of Nov
    "cyber_monday":  (12,  2),   # Monday after Black Friday
    "christmas":     (12, 25),
    "valentines":    ( 2, 14),
    "mothers_day":   ( 5, 12),   # approximate — 2nd Sun of May
    "prime_day":     ( 7, 16),   # approximate — mid July
    "back_to_school":( 8, 15),   # approximate — mid August
    "new_year":      ( 1,  1),
}

# ══════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════
def load_products()-> pd.DataFrame:
    """Load all products from DB."""
    with get_db() as db:
        rows = db.query(Product).all()

    if not rows:
        raise ValueError("No products found in DB. Run db/schema.sql first.")
    
    df = pd.DataFrame([{
        "product_id": r.id,
        "product_name": r.name,
        "category": r.category,
        "base_price": float(r.base_price),
        "current_price": float(r.current_price),
        "inventory_qty": r.inventory_qty,
    } for r in rows])

    log.info(f"Loaded {len(df)} products across {df['category'].nunique()} categories")
    return df

def load_price_history()-> pd.DataFrame:
    """Load price history and compute rolling stats per product."""
    with get_db() as db:
        rows = db.query(PriceHistory).order_by(PriceHistory.product_id, PriceHistory.recorded_at).all()

    if not rows:
        log.warning("No price_history rows found — price features will use current_price only")
        return pd.DataFrame(columns=["product_id", "price", "recorded_at"])
    
    df = pd.DataFrame([{
        "product_id": r.product_id,
        "price": float(r.price),
        "recorded_at": r.recorded_at,
    } for r in rows])

    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    log.info(f"Loaded {len(df)} price_history rows for {df['product_id'].nunique()} products")
    return df

def load_demand_signals()-> pd.DataFrame:
    """Load demand signals and compute trend slope per keyword."""
    with get_db() as db:
        rows = db.query(DemandSignal).order_by(DemandSignal.keyword, DemandSignal.week_date).all()

    if not rows:
        log.warning("No demand_signals found — trend features will be set to neutral (50)")
        return pd.DataFrame(columns=["keyword", "trend_index", "week_date"])
    
    df = pd.DataFrame([{
        "keyword": r.keyword,
        "trend_index": r.trend_index,
        "week_date": r.week_date,
    } for r in rows])

    df["week_date"] = pd.to_datetime(df["week_date"])
    log.info(f"Loaded {len(df)} demand signal rows for {df['keyword'].nunique()} keywords")
    return df

def load_competitor_prices()-> pd.DataFrame:
    """Load competitor prices and compute price band stats per product."""
    with get_db() as db:
        rows = db.query(CompetitorPrice).all()

    if not rows:
        log.warning("No competitor_prices found — competitor features will be set to current_price")
        return pd.DataFrame(columns=["product_id", "platform", "competitor_price"])
    
    df = pd.DataFrame([{
        "product_id":       r.product_id,
        "platform":         r.platform,
        "competitor_price": float(r.competitor_price),
        "scraped_at":       r.scraped_at,
    } for r in rows])

    log.info(f"Loaded {len(df)} competitor price rows for {df['product_id'].nunique()} products")
    return df

# ══════════════════════════════════════════════════════════════════════
# FEATURE BUILDERS
# ══════════════════════════════════════════════════════════════════════
def build_price_features(product_df: pd.DataFrame, price_history_df: pd.DataFrame)-> pd.DataFrame:
    """
    Per product:
      - price_7d_avg     : 7-day rolling average price
      - price_30d_avg    : 30-day rolling average price
      - price_90d_avg    : 90-day rolling average price
      - price_momentum   : % change from 30d avg to current price
      - price_volatility : std dev of prices in last 30 days
    """
    if price_history_df.empty:
        # Fall back to current_price for all price features
        feats = product_df[["product_id", "current_price"]].copy()
        for col in ["price_7d_avg", "price_30d_avg", "price_90d_avg", "price_momentum", "price_volatility"]:
            feats[col] = feats["current_price"] if "avg" in col else 0.0

        return feats.drop(columns=["current_price"])
    
    now = datetime.utcnow()
    records = []

    for product_id in product_df["product_id"]:
        ph = price_history_df[price_history_df["product_id"] == product_id].copy()
        current_price = float(product_df.loc[product_df["product_id"] == product_id, "current_price"])

        def avg_last_n_days(n):
            cutoff = now - timedelta(days=n)
            subset = ph[ph["recorded_at"] >= cutoff]["price"]
            return float(subset.mean()) if len(subset) > 0 else current_price
        
        avg_7d  = avg_last_n_days(7)
        avg_30d = avg_last_n_days(30)
        avg_90d = avg_last_n_days(90)

        # Price momentum: how much has price changed vs 30d avg (signed %)
        momentum = ((current_price - avg_30d) / avg_30d * 100) if avg_30d > 0 else 0.0

        # Volatility: std dev of prices in last 30 days
        cutoff_30 = now - timedelta(days=30)
        recent_prices = ph[ph["recorded_at"] >= cutoff_30]["price"]
        volatility = float(recent_prices.std()) if len(recent_prices) > 1 else 0.0

        records.append({
            "product_id":       product_id,
            "price_7d_avg":     round(avg_7d, 2),
            "price_30d_avg":    round(avg_30d, 2),
            "price_90d_avg":    round(avg_90d, 2),
            "price_momentum":   round(momentum, 3),
            "price_volatility": round(volatility, 3),
        })

    return pd.DataFrame(records)

def build_demand_features(products_df: pd.DataFrame, demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per product (via category keyword):
      - trend_index_latest : most recent Google Trends index (0-100)
      - trend_index_4w_avg : 4-week average trend index
      - trend_slope        : linear slope over last 8 weeks (rising/falling)
      - is_trending        : 1 if latest index > 4w avg by >10 points
      - demand_percentile  : where latest index sits in the 52w distribution
    """
    records = []

    for _, product in products_df.iterrows():
        category = product["category"]
        keyword = CATEGORY_KEYWORD.get(category)

        # Defaults if no demand data
        defaults = {
            "product_id": product["product_id"],
            "trend_index_latest": 50,
            "trend_index_4w_avg": 50,
            "trend_slope": 0.0,
            "is_trending": 0,
            "demand_percentile": 50.0,
        }

        if demand_df.empty or keyword is None:
            records.append(defaults)
            continue

        kw_data = demand_df[demand_df["keyword"] == keyword].sort_values("week_date")

        if kw_data.empty:
            records.append(defaults)
            continue

        # Latest index
        latest_idx = int(kw_data.loc["trend_index"].iloc[-1])

        # 4-week average (last 4 rows = last 4 weeks)
        last_4w    = kw_data.tail(4)["trend_index"]
        avg_4w     = float(last_4w.mean())

        # Linear slope over last 8 weeks
        last_8w    = kw_data.tail(8)
        if len(last_8w) >= 2:
            x = np.arange(len(last_8w))
            y = last_8w["trend_index"].values.astype(float)
            slope = float(np.polyfit(x, y, 1)[0])   # units: index points per week
        else:
            slope = 0.0

         # Is trending: latest index is >10pts above its 4w average
        is_trending = 1 if (latest_idx - avg_4w) > 10 else 0

        # Percentile within the full 52w distribution
        all_indices = kw_data["trend_index"].values
        percentile  = float(np.mean(all_indices <= latest_idx) * 100)

        records.append({
            "product_id": product["product_id"],
            "trend_index_latest": latest_idx,
            "trend_index_4w_avg": round(avg_4w, 1),
            "trend_slope": round(slope, 3),
            "is_trending": is_trending,
            "demand_percentile": round(percentile, 1),
        })

    return pd.DataFrame(records)

def build_competitor_features(products_df: pd.DataFrame, comp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per product:
      - comp_count       : number of competitor listings found
      - comp_price_p25   : 25th percentile competitor price
      - comp_price_med   : median competitor price
      - comp_price_p75   : 75th percentile competitor price
      - comp_price_range : p75 - p25 (market spread)
      - price_position   : our price vs comp median (positive = above market)
      - is_price_competitive : 1 if our price is within ±15% of comp median
    """
    records = []

    for _, product in products_df.iterrows():
        pid = product["product_id"]
        current_price = product["current_price"]

        # Default: use current_price as proxy for all competitor stats
        defaults = {
            "product_id": pid,
            "comp_count": 0,
            "comp_price_p25": current_price,
            "comp_price_med": current_price,
            "comp_price_p75": current_price,
            "comp_price_range": 0.0,
            "price_position": 0.0,
            "is_price_competitive":1,
        }

        if comp_df.empty:
            records.append(defaults)
            continue

        product_comps = comp_df[comp_df["product_id"] == pid]["competitor_price"]

        if len(product_comps) == 0:
            records.append(defaults)
            continue

        prices = product_comps.values
        p25    = float(np.percentile(prices, 25))
        median = float(np.percentile(prices, 50))
        p75    = float(np.percentile(prices, 75))
        spread = p75 - p25

        # Price position: % difference of our price vs competitor median
        # Positive = we're priced above market, negative = below
        position = ((current_price - median) / median * 100) if median > 0 else 0.0

        # Competitive: within ±15% of market median
        is_competitive = 1 if abs(position) <= 15.0 else 0

        records.append({
            "product_id": pid,
            "comp_count": len(prices),
            "comp_price_p25": round(p25, 2),
            "comp_price_med": round(median, 2),
            "comp_price_p75": round(p75, 2),
            "comp_price_range": round(spread, 2),
            "price_position": round(position, 2),
            "is_price_competitive":is_competitive,
        })

    return pd.DataFrame(records)


def build_inventory_features(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per product:
      - inventory_qty      : current stock level
      - inventory_tier     : 0=critical(<10), 1=low(<50), 2=normal, 3=high(>200)
      - inventory_pressure : 1 if stock is low and should push price UP,
                            -1 if overstocked and should push price DOWN
    """
    records = []

    for _, product in products_df.iterrows():
        qty = product["inventory_qty"]

        if qty < 10:
            tier     = 0
            pressure = 1    # Scarce → can price higher
        elif qty < 50:
            tier     = 1
            pressure = 0    # Normal
        elif qty <= 200:
            tier     = 2
            pressure = 0    # Normal
        else:
            tier     = 3
            pressure = -1   # Overstocked → need to clear inventory

        records.append({
            "product_id": product["product_id"],
            "inventory_qty": qty,
            "inventory_tier": tier,
            "inventory_pressure": pressure,
        })

    return pd.DataFrame(records)


def build_time_features(today: date = None) -> pd.DataFrame:
    """
    Global time features (same for all products on a given day):
      - month              : 1-12
      - day_of_week        : 0=Mon ... 6=Sun
      - is_weekend         : 1 if Sat/Sun
      - quarter            : 1-4
      - days_until_*       : days until each retail event (capped at 365)
      - is_holiday_season  : 1 if Nov or Dec
      - is_back_to_school  : 1 if July or August
    """
    if today is None:
        today = date.today()

    def days_until(month, day):
        target = date(today.year, month, day)
        if target < today:
            target = date(today.year + 1, month, day)
        return (target - today).days

    feats = {
        "snapshot_date": today.isoformat(),
        "month": today.month,
        "day_of_week": today.weekday(),
        "is_weekend": 1 if today.weekday() >= 5 else 0,
        "quarter": (today.month - 1) // 3 + 1,
        "is_holiday_season": 1 if today.month in (11, 12) else 0,
        "is_back_to_school": 1 if today.month in (7, 8) else 0,
    }

    for event, (month, day) in RETAIL_EVENTS.items():
        feats[f"days_until_{event}"] = days_until(month, day)

    return feats   # Returns a dict, not a DataFrame (will be broadcast to all rows)


def build_target(products_df: pd.DataFrame, comp_feats: pd.DataFrame, demand_feats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the training target price.

    Formula:
        target_price = comp_price_med * demand_multiplier

    Where:
        demand_multiplier = 1.0 + (trend_index_latest - 50) / 200
            → At trend_index=50 (average): multiplier = 1.0  (no adjustment)
            → At trend_index=100 (peak):   multiplier = 1.25 (25% premium)
            → At trend_index=0  (trough):  multiplier = 0.75 (25% discount)

    This creates a plausible target that ties demand signals to pricing.
    For products with no competitor data, we use current_price as the base.
    """
    merged = products_df[["product_id", "current_price"]].merge(
        comp_feats[["product_id", "comp_price_med"]], on="product_id", how="left"
    ).merge(
        demand_feats[["product_id", "trend_index_latest"]], on="product_id", how="left"
    )

    # Use comp_price_med if available, else fall back to current_price
    merged["price_base"] = merged["comp_price_med"].fillna(merged["current_price"])

    # Demand multiplier: clamp trend index between 0 and 100
    merged["trend_clamped"] = merged["trend_index_latest"].fillna(50).clip(0, 100)
    merged["demand_multiplier"] = 1.0 + (merged["trend_clamped"] - 50) / 200

    merged["target_price"] = (merged["price_base"] * merged["demand_multiplier"]).round(2)

    return merged[["product_id", "price_base", "demand_multiplier", "target_price"]]


# ══════════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_feature_matrix(today: date = None) -> pd.DataFrame:
    """
    Orchestrates all feature builders and joins them into a single DataFrame.
    Returns the complete feature matrix.
    """
    today = today or date.today()
    log.info(f"Building feature matrix for snapshot date: {today}")

    # ── Load raw data ─────────────────────────────────────────────────
    log.info("\n[1/6] Loading products...")
    products_df = load_products()

    log.info("[2/6] Loading price history...")
    price_history_df = load_price_history()

    log.info("[3/6] Loading demand signals...")
    demand_df = load_demand_signals()

    log.info("[4/6] Loading competitor prices...")
    comp_df = load_competitor_prices()

    # ── Build feature groups ─────────────────────────────────────────
    log.info("\n[5/6] Building features...")

    log.info("  Building price features...")
    price_feats = build_price_features(products_df, price_history_df)

    log.info("  Building demand features...")
    demand_feats = build_demand_features(products_df, demand_df)

    log.info("  Building competitor features...")
    comp_feats = build_competitor_features(products_df, comp_df)

    log.info("  Building inventory features...")
    inv_feats = build_inventory_features(products_df)

    log.info("  Building time features...")
    time_feats = build_time_features(today)

    log.info("  Computing target prices...")
    target_df = build_target(products_df, comp_feats, demand_feats)

    # ── Join everything ───────────────────────────────────────────────
    log.info("\n[6/6] Joining feature groups...")

    df = products_df.copy()
    # Drop inventory_qty from base df — inv_feats contains it plus tier/pressure
    df = df.drop(columns=["inventory_qty"], errors="ignore")
    df = df.merge(price_feats,   on="product_id", how="left")
    df = df.merge(demand_feats,  on="product_id", how="left")
    df = df.merge(comp_feats,    on="product_id", how="left")
    df = df.merge(inv_feats,     on="product_id", how="left")
    df = df.merge(target_df[["product_id", "target_price", "demand_multiplier"]], on="product_id", how="left")

    # Broadcast time features to all rows
    for col, val in time_feats.items():
        df[col] = val

    # ── Category encoding ─────────────────────────────────────────────
    # One-hot encode category (XGBoost works fine with integer categoricals too,
    # but explicit dummies make feature importance plots cleaner)
    cat_dummies = pd.get_dummies(df["category"], prefix="cat").astype(int)
    df = pd.concat([df, cat_dummies], axis=1)

    # ── Final cleanup ─────────────────────────────────────────────────
    df = df.fillna(0)

    log.info(f"\nFeature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════

def validate_features(df: pd.DataFrame) -> bool:
    """
    Runs basic data quality checks. Returns True if all pass.
    """
    log.info("\nRunning validation checks...")
    passed = True

    checks = {
        "No missing product_ids": df["product_id"].isna().sum() == 0,
        "No missing target prices": df["target_price"].isna().sum() == 0,
        "All target prices > 0": (df["target_price"] > 0).all(),
        "No negative inventory": (df["inventory_qty"] >= 0).all(),
        "Trend index in range [0,100]": df["trend_index_latest"].between(0, 100).all(),
        "Price momentum is finite": np.isfinite(df["price_momentum"]).all(),
        "No duplicate product_ids": df["product_id"].nunique() == len(df),
        "Has at least 4 categories": df["category"].nunique() >= 4,
    }

    for check, result in checks.items():
        icon = "OK" if result else "FAIL"
        log.info(f"  [{icon}] {check}")
        if not result:
            passed = False

    if passed:
        log.info("\n  All checks passed.")
    else:
        log.warning("\n  Some checks failed — review features before training.")

    return passed

def print_preview(df: pd.DataFrame):
    """Prints a human-readable summary of the feature matrix."""
    log.info("\n" + "="*60)
    log.info("FEATURE MATRIX PREVIEW")
    log.info("="*60)
    log.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Per-product summary
    cols = [
        "product_id", "product_name", "category",
        "current_price", "target_price", "demand_multiplier",
        "trend_index_latest", "comp_price_med",
        "price_position", "inventory_qty", "inventory_pressure",
    ]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False, max_colwidth=30))

    log.info("\nAll feature columns:")
    feature_cols = [c for c in df.columns
                    if c not in ("product_id", "product_name", "category",
                                 "snapshot_date", "target_price")]
    for i, col in enumerate(feature_cols):
        dtype = df[col].dtype
        mn = df[col].min() if np.issubdtype(dtype, np.number) else "N/A"
        mx = df[col].max() if np.issubdtype(dtype, np.number) else "N/A"
        log.info(f"  {i+1:>2}. {col:<35} {str(dtype):<10} min={mn}  max={mx}")


def save_meta(df: pd.DataFrame, path: str):
    """Saves column metadata to JSON for reference in other modules."""
    feature_cols = [c for c in df.columns
                    if c not in ("product_id", "product_name", "category",
                                 "snapshot_date")]
    meta = {
        "created_at": datetime.utcnow().isoformat(),
        "shape": list(df.shape),
        "feature_cols": [c for c in feature_cols if c != "target_price"],
        "target_col": "target_price",
        "id_cols": ["product_id", "product_name", "category"],
        "n_products": int(df["product_id"].nunique()),
        "categories": list(df["category"].unique()),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata saved to {path}")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build DemandSurge feature matrix")
    parser.add_argument("--preview",  action="store_true", help="Print feature matrix summary")
    parser.add_argument("--validate", action="store_true", help="Run validation checks only")
    parser.add_argument("--date",     type=str, default=None, help="Snapshot date in YYYY-MM-DD format (default: today)")
    args = parser.parse_args()

    snapshot_date = (date.fromisoformat(args.date) if args.date else date.today())

    try:
        df = build_feature_matrix(today=snapshot_date)
    except Exception as e:
        log.error(f"Feature build failed: {e}")
        raise

    passed = validate_features(df)

    if args.preview or args.validate:
        print_preview(df)
        return 0 if passed else 1

    # Save outputs
    df.to_parquet(PARQUET_PATH, index=False)
    log.info(f"\nFeatures saved to {PARQUET_PATH}")

    save_meta(df, META_PATH)

    # Print a quick preview regardless
    log.info("\nSample output (first 5 rows, key columns):")
    key_cols = ["product_name", "current_price", "target_price",
                "trend_index_latest", "comp_price_med",
                "price_position", "inventory_pressure"]
    available = [c for c in key_cols if c in df.columns]
    print(df[available].head().to_string(index=False, max_colwidth=35))

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())


