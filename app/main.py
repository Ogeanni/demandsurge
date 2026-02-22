"""
api/main.py
DemandSurge FastAPI backend.

Exposes the LangChain agent and ML models as REST endpoints.
Streamlit calls these endpoints over HTTP — the UI never imports
ML code directly.

Endpoints:
    POST /chat                     — send a message to the ReAct agent
    GET  /products                 — list all products
    GET  /products/{id}/recommend  — XGBoost price recommendation
    GET  /products/{id}/competitors— competitor price band
    GET  /forecast/{category}      — Prophet 30-day demand forecast
    GET  /review                   — full weekly pricing review
    GET  /health                   — health check

Usage:
    uvicorn api.main:app --reload --port 8000
    # Swagger docs: http://localhost:8000/docs
"""

import os
import sys
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
log = logging.getLogger("app")

# Suppress noisy loggers
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── DemandSurge imports ──────────────────────────────────────────────────
from db.models import get_db, Product, CompetitorPrice
from src.pricing_model import get_price_recommendation, get_all_recommendations, load_features
from src.demand_forecast import get_demand_forecast
from src.agent import DemandSurgeAgent

# ── Pydantic Models──────────────────────────────────────────────────
from app.models import ChatRequest, ChatResponse, ProductOut, RecommendationOut, ForecastOut, CompetitorOut, WeeklyReviewItem, HealthOut

# ══════════════════════════════════════════════════════════════════════
# APP STATE  — agent loaded once on startup
# ══════════════════════════════════════════════════════════════════════
class AppState:
    agent = None
    agent_error: Optional[str] = None
    started_at: Optional[float] = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: build the LangChain agent once and hold it in memory."""
    log.info("DemandSurge API starting up...")
    state.started_at = time.time()
    try:
        state.agent = DemandSurgeAgent()
        log.info("Agent loaded successfully")
    except Exception as e:
        state.agent_error = str(e)
        log.error(f"Agent failed to load: {e}")
    yield
    log.info("DemandSurge API shutting down")

# ══════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="DemandSurge Pricing API",
    description=(
        "AI-powered pricing intelligence for Shopify merchants. "
        "Combines XGBoost pricing recommendations, Prophet demand forecasts, "
        "and a LangChain ReAct agent for natural language pricing queries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit (running on a different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════
@app.get("/health", response_model=HealthOut, tags=["System"])
def health():
    """
    Health check. Returns agent readiness and uptime.
    Streamlit polls this on startup to confirm the API is reachable.
    """
    uptime = time.time() - state.started_at if state.started_at else 0
    return HealthOut(
        status="ok",
        agent_ready=state.agent is not None,
        agent_error=state.agent_error,
        uptime_seconds=round(uptime, 1),
        version="1.0.0",
    )

# ── Chat ──────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(request: ChatRequest):
    """
    Send a natural language message to the DemandSurge ReAct agent.

    The agent reasons over three tools — pricing recommendation,
    demand forecast, and competitor prices — before responding.

    Example questions:
    - "What price for the yoga mat heading into summer?"
    - "Is demand rising for electronics this month?"
    - "Give me a full weekly pricing review."
    """
    if state.agent is None:
        raise HTTPException(
            status_code=503,
            detail=f"Agent not available: {state.agent_error or 'unknown error'}"
        )
    start = time.time()
    try:
        agent = DemandSurgeAgent()
        result = agent.run_query(user_query=request.message, session_id=request.session_id)
    except Exception as e:
        log.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    
    elapsed_ms = int((time.time() - start) * 1000)
    log.info(f"Chat query completed in {elapsed_ms}ms")

    return ChatResponse (
        response=result["response"],
        session_id=request.session_id,
        elapsed_ms=elapsed_ms,
    )

# ── Products ──────────────────────────────────────────────────────────
@app.get("/products", response_model=list[ProductOut], tags=["Products"])
def list_products(category: Optional[str] = Query(None)):
    """
    List all products, optionally filtered by category.
    Used by Streamlit sidebar to populate the product selector.
    """
    try:
        with get_db() as db:
            query = db.query(Product)
            if category:
                query = query.filter(Product.category == category)
            products = query.order_by(Product.category, Product.name).all()
        
        return [
            ProductOut(
                id=p.id,
                name=p.name,
                category=p.category,
                current_price=float(p.current_price),
                inventory_qty=p.inventory_qty,
            )
            for p in products
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/{product_id}/recommend", response_model=RecommendationOut, tags=["Products"])
def recommend(product_id: int):
    """
    Get an XGBoost price recommendation for a specific product.
    Used by Streamlit sidebar metrics cards.
    """
    try:
        df = load_features()
        matches = df[df["product_id"] == product_id]
        if matches.empty:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        row = matches.iloc[0]
        rec = get_price_recommendation(product_row=row)
        return RecommendationOut(**rec)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/products/{product_id}/competitors", response_model=CompetitorOut, tags=["Products"])
def competitor_prices(product_id: int):
    """
    Get competitor price band for a product from eBay and Etsy data.
    Falls back to feature matrix values if no live DB data exists.
    """
    try:
        with get_db() as db:
            product = db.query(Product).filter(Product.id == product_id).first()
            if not product:
                raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
            
            comp_rows = db.query(CompetitorPrice).filter(CompetitorPrice.product_id == product_id).all()
        
        current_price = float(product.current_price)

        if not comp_rows:
            # Fallback to feature matrix
            df = load_features()
            row = df[df["product_id"] == product_id]
            if row.empty:
                raise HTTPException(status_code=404, detail="No feature data found")
            row = row.iloc[0]
            comp_med = float(row.get("competitor_price", current_price))
            position = float(row.get("price_position", 0))
            return CompetitorOut(
                    product_id=product_id,
                    product_name=product.name,
                    current_price=current_price,
                    comp_count=0,
                    comp_median=comp_med,
                    comp_min=comp_med,
                    comp_max=comp_med,
                    price_position_pct=position,
                    assessment="feature matrix fallback — no live data",
                    by_platform={},
                )
        prices = [float(r.competitor_price) for r in comp_rows]
        import statistics
        median = statistics.median(prices)
        position = ((current_price - median) / median * 100) if median > 0 else 0

        by_platform: dict = {}
        for r in comp_rows:
            by_platform.setdefault(r.platform, []).append(float(r.competitor_price))
        by_platform_summary = {
            p: {"count": len(v),
                "median:": round(statistics.median(v), 2)}
            for p, v in by_platform.items()
        }

        assessment = (
            "above market" if position > 10 else
            "below market" if position < -10 else
            "in line with market"
        )

        return CompetitorOut(
                product_id=product_id,
                product_name=product.name,
                current_price=current_price,
                comp_count=len(prices),
                comp_median=round(median, 2),
                comp_min=round(min(prices), 2),
                comp_max=round(max(prices), 2),
                price_position_pct=round(position, 1),
                assessment=assessment,
                by_platform=by_platform_summary,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ── Forecast ──────────────────────────────────────────────────────────
@app.get("/forecast/{category}", response_model=ForecastOut, tags=["Demand"])
def forecast(category: str, days: int = Query(30, ge=7, le=90, description = "Forecast horizon in days"),):
    """
    Get a Prophet demand forecast for a product category.
    Valid categories: electronics, fashion, home_goods, sports
    """
    valid = ["electronics", "fashion", "home_goods", "sports"]
    cat = category.strip().lower()
    if cat not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{category}'. Valid: {valid}"
        )
    try:
        fc = get_demand_forecast(cat, days=days)
        fc.pop("forecast_df", None)   # Remove DataFrame — not JSON serialisable
        return ForecastOut(**fc)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ── Weekly Review ─────────────────────────────────────────────────────
@app.get("/review", response_model=list[WeeklyReviewItem], tags=["Review"])
def weekly_review(category: Optional[str] = Query(None)):
    """
    Run a full weekly pricing review across all products.
    Returns products sorted by urgency (largest % change first).
    """
    try:
        recs = get_all_recommendations()
        if recs.empty:
            return []
        
        if category:
            recs = recs[recs["category"] == category]

        recs["abs_change"] = recs["pct_vs_current"].abs()
        recs.sort_values("abs_change", ascending=False)

        def priority(pct):
            if abs(pct) >= 10:
                return "urgent"
            elif abs(pct) >= 5:
                return "moderate"
            return "stable"
        
        return [
            WeeklyReviewItem(
                product_id=int(r["product_id"]),
                product_name=r["product_name"],
                category=r["category"],
                current_price=round(float(r["current_price"]), 2),
                recommended_price=round(float(r["recommended_price"]), 2),
                pct_vs_current=round(float(r["pct_vs_current"]), 1),
                priority=priority(r["pct_vs_current"]),
            )
            for _, r in recs.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
