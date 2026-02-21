"""
models/pricing_model.py
XGBoost pricing model for DemandSurge.

Trains a regression model on the feature matrix built by models/features.py,
predicts optimal price recommendations, and exposes a clean inference API
for the LangChain agent tools in Block 8.

Usage:
    python models/pricing_model.py               # Train and save model
    python models/pricing_model.py --evaluate    # Train + detailed eval report
    python models/pricing_model.py --predict     # Load saved model, run predictions
    python models/pricing_model.py --importance  # Show feature importance chart

Output:
    models/saved_models/pricing_model.ubj         — trained XGBoost model
    models/saved_models/pricing_model_meta.json   — feature columns, metrics, training date
    results/predictions.csv             — per-product price recommendations
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime

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
log = logging.getLogger("pricing_model")

try:
    from xgboost import XGBRegressor
    import xgboost as xgb
except ImportError:
    log.error("xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────
MODELS_DIR   = os.path.join(ROOT, "models/saved_models")
RESULT_DIR   = os.path.join(ROOT, "results")
DATA_DIR     = os.path.join(ROOT, "data/processed")
MODEL_PATH   = os.path.join(MODELS_DIR, "pricing_model.ubj")
META_PATH    = os.path.join(MODELS_DIR, "pricing_model_meta.json")
FEATURES_PATH = os.path.join(DATA_DIR, "features.parquet")
PREDICTIONS_PATH = os.path.join(RESULT_DIR, "predictions.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# ── Columns to exclude from training ─────────────────────────────────
# These are identifiers, labels, or leakage columns — not features
EXCLUDE_COLS = [
    "product_id",
    "product_name",
    "category",
    "snapshot_date",
    "target_price",        # This is the label
    "price_base",          # Used to compute target — would be leakage
    "demand_multiplier",   # Used to compute target — would be leakage
    "current_price",       # Direct leakage — we're trying to improve on this
]

# ══════════════════════════════════════════════════════════════════════
# DATA LOADING & SPLITTING
# ══════════════════════════════════════════════════════════════════════
def load_features()-> pd.DataFrame:
    """Load feature matrix from parquet."""
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Feature matrix not found at {FEATURES_PATH}. "
            "Run: python models/features.py")
    
    df = pd.read_parquet(FEATURES_PATH)
    log.info(f"Loaded feature matrix: {df.shape[0]} rows  {df.shape[1]} columns")
    return df

def get_feature_columns(df: pd.DataFrame)-> list:
    """Returns the list of columns to use as model features."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    log.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    return feature_cols

def time_based_split(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    """
    Splits the feature matrix into train and test sets.

    IMPORTANT: We use the product_id order as a proxy for time since
    we have one snapshot per product (not a time series of snapshots yet).
    Products with higher IDs were seeded later — a weak but consistent
    ordering. In production with daily snapshots, split on snapshot_date.

    test_size=0.2 means the last 20% of products go to test.
    """
    n = len(df)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))



    # Sort by product_id to get a consistent, reproducible split
    df_sorted  = df.sort_values("product_id").reset_index(drop=True)
    train_df = df_sorted.iloc[:train_end].copy()
    val_df   = df_sorted.iloc[train_end:val_end].copy()
    test_df  = df_sorted.iloc[val_end:].copy()

    log.info(f"Split — Train: {len(train_df)} products  | Val: {len(val_df)} products | Test: {len(test_df)} products")
    return train_df, val_df, test_df

# ══════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════
def build_model()-> XGBRegressor:
     """
    Builds and configures the XGBoost pricing model.

    Parameter choices explained:
    - n_estimators=300       : Number of trees. More = better fit, slower training.
                               300 is enough for 20 products; scale up with more data.
    - max_depth=4            : Max depth per tree. Shallow trees (3-5) generalise
                               better on small datasets. Deep trees overfit.
    - learning_rate=0.05     : Step size shrinkage. Smaller = more conservative,
                               needs more trees to compensate, but generalises better.
    - subsample=0.8          : Each tree trains on 80% of rows (random). Reduces
                               overfitting by introducing variance between trees.
    - colsample_bytree=0.8   : Each tree uses 80% of features (random). Same idea —
                               prevents any single feature from dominating all trees.
    - min_child_weight=3     : Minimum sum of instance weight in a child node.
                               Higher = more conservative splits. Prevents the model
                               from learning from tiny subgroups.
    - reg_alpha=0.1          : L1 regularisation. Pushes less important feature
                               weights toward zero (feature selection effect).
    - reg_lambda=1.5         : L2 regularisation. Penalises large weights.
                               Prevents any single feature from having outsized impact.
    - random_state=42        : Reproducibility.
    - early_stopping_rounds  : Set at fit time — stops if test loss doesn't improve
                               for N rounds. Prevents overfitting automatically.
    """
     return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.5,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        verbosity=0,
     )

def train(df: pd.DataFrame)-> dict:
    """
    Full training pipeline:
      1. Split into train/test
      2. Extract features and target
      3. Train XGBoost with early stopping
      4. Evaluate on held-out test set
      5. Retrain on full data for production model
      6. Save model and metadata

    Returns a result dict with model, metrics, and predictions.
    """
    log.info("\n" + "="*55)
    log.info("Training XGBoost Pricing Model")
    log.info("="*55)

    feature_cols = get_feature_columns(df)
    target_col = "target_price"

    # ── Split ─────────────────────────────────────────────────────────
    train_df, val_df, test_df = time_based_split(df, test_size=0.2)

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df[target_col].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target_col].values

    log.info(f"\nX_train: {X_train.shape}  | X_val: {X_val.shape} |  X_test: {X_test.shape}")
    log.info(f"y_train range: ${y_train.min():.2f} - ${y_train.max():.2f}")
    log.info(f"y_val range: ${y_val.min():.2f} - ${y_val.max():.2f}")
    log.info(f"y_test  range: ${y_test.min():.2f} - ${y_test.max():.2f}")

    # ── Train with early stopping ─────────────────────────────────────
    log.info("\nFitting XGBoost model...")
    model = build_model()
    model.set_params(early_stopping_rounds=30)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_round = model.best_iteration
    log.info(f"Best iteration: {best_round} / {model.n_estimators}")

    # ── Evaluate ──────────────────────────────────────────────────────
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    metrics = compute_metrics(y_test, y_pred_test, y_train, y_pred_train)
    log.info(f"\nTest  — MAE: ${metrics['test_mae']:.2f}  "
             f"RMSE: ${metrics['test_rmse']:.2f}  "
             f"R²: {metrics['test_r2']:.3f}")
    log.info(f"Train — MAE: ${metrics['train_mae']:.2f}  "
             f"RMSE: ${metrics['train_rmse']:.2f}  "
             f"R²: {metrics['train_r2']:.3f}")
    
    # Overfitting check
    overfit_gap = metrics["train_mae"] - metrics["test_mae"]
    if abs(overfit_gap) > 5:
        log.warning(f"  Large train/test MAE gap (${abs(overfit_gap):.2f}) — "
                    f"possible overfitting. Consider increasing regularisation.")
    else:
        log.info(f"  Train/test MAE gap: ${abs(overfit_gap):.2f} — looks healthy")

     # ── Retrain on full data ──────────────────────────────────────────
    log.info("\nRetraining on train+val for production model...")
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    final_model = build_model()
    # Use best_iteration from early stopping as n_estimators for full retrain
    final_model.set_params(n_estimators=max(best_round + 1, 50))
    final_model.fit(
        train_val_df[feature_cols].values,
        train_val_df[target_col].values,
        verbose=False,
    )
    # ── Save model ────────────────────────────────────────────────────
    final_model.save_model(MODEL_PATH)
    log.info(f"Model saved → {MODEL_PATH}")

    # ── Save metadata ─────────────────────────────────────────────────
    meta = {
        "trained_at":      datetime.utcnow().isoformat(),
        "feature_cols":    feature_cols,
        "target_col":      target_col,
        "n_features":      len(feature_cols),
        "n_train":         len(train_df),
        "n_test":          len(test_df),
        "best_iteration":  best_round,
        "metrics":         metrics,
        "model_params":    final_model.get_params(),
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2, default=str)
        log.info(f"Metadata saved → {META_PATH}")

    return {
        "model":        final_model,
        "feature_cols": feature_cols,
        "metrics":      metrics,
        "train_df":     train_df,
        "test_df":      test_df,
        "y_pred_test":  y_pred_test,
    }

def compute_metrics(y_test, y_pred_test, y_train, y_pred_train)-> dict:
    """Computes MAE, RMSE, R² for both train and test sets."""
    return {
        "test_mae": round(float(mean_absolute_error(y_test, y_pred_test)), 2),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_test))), 2),
        "test_r2": round(float(r2_score(y_test, y_pred_test)), 3),
        "train_mae": round(float(mean_absolute_error(y_train, y_pred_train)), 2),
        "train_rmse": round(float(np.sqrt(mean_squared_error(y_train, y_pred_train))), 2),
        "train_r2": round(float(r2_score(y_train, y_pred_train)), 3)
    }

# ══════════════════════════════════════════════════════════════════════
# EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════
def print_evaluation_report(result: dict, df: pd.DataFrame):
    """
    Prints a detailed per-product evaluation showing:
    - Current price (what the merchant has now)
    - Target price (what our formula says)
    - Predicted price (what XGBoost recommends)
    - Error (predicted vs target)
    - Business signal (too high / competitive / too low)
    """
    feature_cols = get_feature_columns(df)
    model = result["model"]

    all_preds = model.predict(df[feature_cols].values)

    log.info("\n" + "="*90)
    log.info("PER-PRODUCT EVALUATION REPORT")
    log.info("="*90)
    log.info(
        f"{'Product':<35} {'Current':>9} {'Target':>9} "
        f"{'Predicted':>10} {'Error':>8} {'Signal':<20}"
    )
    log.info("-"*90)

    for i, (_, row) in enumerate(df.iterrows()):
        current = float(row["current_price"])
        target = float(row["target_price"])
        predicted = round(float(all_preds[i]),2)
        error = round(predicted - current, 2)
        name = str(row["product_name"])[:34]

        # Determine business signal vs competitor median
        comp_med = float(row.get("comp_price_med", current))
        pct_vs_market = (predicted - comp_med) / comp_med * 100 if comp_med > 0 else 0

        if pct_vs_market > 15:
            signal = "Above market"
        elif pct_vs_market < -15:
            signal = "Below market"
        else:
            signal = "Competitive"

        log.info(
            f"{name:<35} ${current:>8.2f} ${target:>8.2f} "
            f"${predicted:>9.2f} ${error:>+7.2f} {signal}"
        )

    log.info("="*90)

    # Save predictions CSV
    pred_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        predicted = round(float(all_preds[i]), 2)
        comp_med = float(row.get("comp_price_med", row["current_price"]))
        pred_rows.append({
            "product_id":       int(row["product_id"]),
            "product_name":     row["product_name"],
            "category":         row["category"],
            "current_price":    float(row["current_price"]),
            "target_price":     float(row["target_price"]),
            "predicted_price":  predicted,
            "comp_price_med":   comp_med,
            "pct_vs_market":    round((predicted - comp_med) / comp_med * 100, 1) if comp_med > 0 else 0,
        })
    
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    log.info(f"\nPredictions saved → {PREDICTIONS_PATH}")

def print_feature_importance(model: XGBRegressor, feature_cols: list, top_n: int = 15):
    """Prints top N most important features."""
    importances = model.feature_importances_
    pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    log.info(f"\nTop {top_n} Feature Importances:")
    log.info("-"*50)
    max_imp = pairs[0][1] if pairs else 1
    for name, imp in pairs[:top_n]:
        bar_len = int((imp / max_imp) * 30)
        bar = "█" * bar_len
        log.info(f"  {name:<35} {imp:.4f}  {bar}")
    log.info("-"*50)

    # Try to save a matplotlib plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [p[0] for p in pairs[:top_n]]
        imps  = [p[1] for p in pairs[:top_n]]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names[::-1], imps[::-1], color="#1B7F79")
        ax.set_xlabel("Feature Importance (F-score)")
        ax.set_title("ShopMind XGBoost — Top Feature Importances")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        plot_path = os.path.join(RESULT_DIR, "feature_importance.png")
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        log.info(f"Importance plot saved → {plot_path}")
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════
# INFERENCE — called by LangChain agent tools in Block 8
# ══════════════════════════════════════════════════════════════════════
def load_model()-> tuple:
    """
    Loads the trained XGBoost model and its metadata from disk.
    Returns (model, meta) tuple.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run: python models/pricing_model.py"
        )
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Model metadata not found at {META_PATH}.")
    
    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    with open(META_PATH) as f:
        meta = json.load(f)

    return model, meta

def get_price_recommendation(product_id: int = None, product_row: pd.Series = None):
    """
    Public API — called by LangChain agent tool in Block 8.

    Accepts either:
      - product_id: looks up the product in features.parquet
      - product_row: a pre-built feature row (Series)

    Returns:
    {
        "product_id":         int,
        "product_name":       str,
        "current_price":      float,
        "recommended_price":  float,
        "lower_bound":        float,   # 10th percentile estimate
        "upper_bound":        float,   # 90th percentile estimate
        "comp_price_med":     float,
        "pct_vs_market":      float,   # recommended vs market median
        "confidence":         str,     # "high" | "medium" | "low"
        "rationale":          str,     # human-readable explanation for agent
    }
    """
    model, meta = load_model()
    feature_cols = meta["feature_cols"]

    # ── Get feature row ───────────────────────────────────────────────
    if product_row is None:
        df = load_features()
        matches = df[df["product_id"] == product_id]
        if matches.empty:
            raise ValueError(f"Product ID {product_id} not found in feature matrix.")
        product_row = matches.iloc[0]

    # ── Predict ───────────────────────────────────────────────────────
    X = product_row[feature_cols].values.reshape(1, -1)
    predicted = float(model.predict(X)[0])
    predicted = round(predicted, 2)

    # Confidence bounds: ±8% as a simple interval
    # In production: use quantile regression or bootstrap for real intervals
    lower = round(predicted * 0.92, 2)
    upper = round(predicted * 1.08, 2)

    # ── Context values ────────────────────────────────────────────────
    current_price = float(product_row.get("current_price", predicted))
    comp_med = float(product_row.get("comp_price_med", predicted))
    trend_idx = float(product_row.get("trend_index_latest", 50))
    inv_pressure = int(product_row.get("inventory_pressure", 0))
    is_trending = int(product_row.get("is_trending", 0))

    pct_vs_market = round((predicted - comp_med) / comp_med * 100, 1) if comp_med > 0 else 0
    pct_vs_current = round((predicted - current_price) / current_price * 100, 1) if current_price > 0 else 0

    # ── Confidence ────────────────────────────────────────────────────
    comp_count = int(product_row.get("comp_count", 0))
    if comp_count >= 5 and trend_idx > 0:
        confidence = "High"
    elif comp_count >= 2 or trend_idx > 0:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # ── Build rationale for agent ─────────────────────────────────────
    product_name = str(product_row.get("product_name", f"Product {product_id}"))
    category = str(product_row.get("category", "unknown"))

    rationale_parts = [
        f"Recommended price for {product_name}: ${predicted:.2f}."
    ]

    # Price vs market
    if abs(pct_vs_market) <= 5:
        rationale_parts.append(f"This is in line with the market median (${comp_med:.2f}).")
    elif abs(pct_vs_market) > 5:
        rationale_parts.append(
            f"This is {pct_vs_market:.1f}% above the market median (${comp_med:.2f}), "
            f"justified by elevated demand signals."
        )
    else:
        rationale_parts.append(
            f"This is {abs(pct_vs_market):.1f}% below the market median (${comp_med:.2f}), "
            f"reflecting softening demand or inventory pressure."
        )

    # Demand context
    if is_trending:
        rationale_parts.append(
            f"Demand is trending up (index {trend_idx:.0f}/100) — "
            f"supports pricing at the higher end of the range."
        )
    elif trend_idx >= 60:
        rationale_parts.append(f"Demand is healthy at {trend_idx:.0f}/100.")
    else:
        rationale_parts.append(
            f"Demand is below average ({trend_idx:.0f}/100) — "
            f"pricing conservatively to maintain volume."
        )

    # Inventory context
    if inv_pressure == 1:
        rationale_parts.append("Stock is critically low — scarcity supports a price premium.")
    elif inv_pressure == -1:
        rationale_parts.append(
            "Inventory is high — a competitive price helps clear stock faster."
        )
    
    # Change vs current
    if pct_vs_current > 0:
        rationale_parts.append(
            f"This is a ${predicted - current_price:.2f} increase "
            f"({pct_vs_current:.1f}%) from the current price of ${current_price:.2f}."
        )
    elif pct_vs_current < 0:
        rationale_parts.append(
            f"This is a ${current_price - predicted:.2f} reduction "
            f"({abs(pct_vs_current):.1f}%) from the current price of ${current_price:.2f}."
        )
    else:
        rationale_parts.append("No change from current price recommended.")

    rationale_parts.append(
        f"Confidence: {confidence} "
        f"(based on {comp_count} competitor listings and demand signal availability)."
    )

    return {
        "product_id":        int(product_row.get("product_id", product_id or 0)),
        "product_name":      product_name,
        "category":          category,
        "current_price":     round(current_price, 2),
        "recommended_price": predicted,
        "lower_bound":       lower,
        "upper_bound":       upper,
        "comp_price_med":    round(comp_med, 2),
        "pct_vs_market":     pct_vs_market,
        "pct_vs_current":    pct_vs_current,
        "trend_index":       trend_idx,
        "confidence":        confidence,
        "rationale":         " ".join(rationale_parts),
    }

def get_all_recommendations()-> pd.DataFrame:
    """
    Runs get_price_recommendation for every product in the feature matrix.
    Returns a DataFrame — used by the agent's weekly review tool.
    """
    df = load_features()
    recs  = []

    for _, row in df.iterrows():
        try:
            rec = get_price_recommendation(product_row=row)
            recs.append(rec)
        except Exception as e:
            log.warning(f"  Skipped product {row.get('product_id')}: {e}")

    return pd.DataFrame(recs)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate DemandSurge XGBoost pricing model")
    parser.add_argument("--evaluate",   action="store_true",
                        help="Print detailed per-product evaluation report after training")
    parser.add_argument("--predict",    action="store_true",
                        help="Load saved model and run predictions")
    parser.add_argument("--importance", action="store_true",
                        help="Print feature importance table")
    args = parser.parse_args()

    # ── Predict-only mode ─────────────────────────────────────────────
    if args.predict:
        log.info("Loading saved model and running predictions...")
        try:
            recs = get_all_recommendations()
            log.info(f"\nRecommendations for {len(records)} products:")
            log.info("-"*70)
            for _, r in recs.iterrows():
                direction = "↑" if r["pct_vs_current"] > 0 else ("↓" if r["pct_vs_current"] < 0 else "→")
                log.info(
                    f"  [{r['category']:<12}] {r['product_name'][:30]:<30} "
                    f"${r['current_price']:.2f} → ${r['recommended_price']:.2f} "
                    f"{direction} ({r['pct_vs_current']:+.1f}%)  [{r['confidence']}]"
                )
        except FileNotFoundError as e:
            log.error(str(e))
            return 1
        return 0

    # ── Training mode ─────────────────────────────────────────────────
    try:
        df = load_features()
    except FileNotFoundError as e:
        log.error(str(e))
        return 1

    result = train(df)

    if args.importance or args.evaluate:
        print_feature_importance(result["model"], result["feature_cols"])

    if args.evaluate:
        print_evaluation_report(result, df)

    # Always print a quick recommendation preview
    log.info("\nSample recommendations (run --predict for full list):")
    log.info("-"*70)
    try:
        recs = get_all_recommendations()
        for _, r in recs.head(8).iterrows():
            direction = "↑" if r["pct_vs_current"] > 0 else ("↓" if r["pct_vs_current"] < 0 else "→")
            log.info(
                f"  {r['product_name'][:32]:<32} "
                f"${r['current_price']:.2f} → ${r['recommended_price']:.2f} "
                f"{direction} ({r['pct_vs_current']:+.1f}%)"
            )
    except Exception as e:
        log.warning(f"Could not generate preview: {e}")

    metrics = result["metrics"]
    log.info("\n" + "="*55)
    log.info("FINAL METRICS")
    log.info("="*55)
    log.info(f"  Test  MAE  : ${metrics['test_mae']:.2f}")
    log.info(f"  Test  RMSE : ${metrics['test_rmse']:.2f}")
    log.info(f"  Test  R²   : {metrics['test_r2']:.3f}")
    log.info(f"  Train MAE  : ${metrics['train_mae']:.2f}")
    log.info(f"  Train R²   : {metrics['train_r2']:.3f}")
    log.info(f"\n  Model saved → {MODEL_PATH}")
    log.info(f"  Run with --evaluate for full per-product breakdown")
    log.info(f"  Run with --predict  to score products from saved model")

    return 0


if __name__ == "__main__":
    sys.exit(main())






