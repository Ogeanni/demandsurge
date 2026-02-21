from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

# ══════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Natural language pricing question")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "What price should I set for the leather wallet heading into Black Friday?",
                "session_id": "demo-session-001",
            }
        }
    )


class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str]
    elapsed_ms: int

class ProductOut(BaseModel):
    id: int
    name: str
    category: str
    current_price: float
    inventory_qty: int

class RecommendationOut(BaseModel):
    product_id: int
    product_name: str
    category: str
    current_price: float
    recommended_price: float
    lower_bound: float
    upper_bound: float
    comp_price_med: float
    pct_vs_market: float
    pct_vs_current: float
    trend_index: float
    confidence: str
    rationale: str

class ForecastOut(BaseModel):
    category: str
    keyword: str
    current_index: float
    forecast_avg: float
    forecast_high: float
    forecast_low: float
    trend_direction: str
    confidence_low: float
    confidence_high: float
    demand_signal: str

class CompetitorOut(BaseModel):
    product_id: int
    product_name: str
    current_price: float
    comp_count: int
    comp_median: float
    comp_min: float
    comp_max: float
    price_position_pct: float
    assessment: str
    by_platform: dict

class WeeklyReviewItem(BaseModel):
    product_id: int
    product_name: str
    category: str
    current_price: float
    recommended_price: float
    pct_vs_current: float
    priority: str   # "urgent" | "moderate" | "stable"

class HealthOut(BaseModel):
    status: str
    agent_ready: bool
    agent_error: Optional[str]
    uptime_seconds: float
    version: str