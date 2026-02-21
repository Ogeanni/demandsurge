"""
agent/pricing_agent.py
LangChain ReAct pricing agent for DemandSurgw.

The agent answers natural language pricing questions by reasoning over
three tools:
  1. get_pricing_recommendation  — XGBoost model prediction per product
  2. get_demand_forecast         — Prophet 30-day demand outlook per category
  3. get_competitor_prices       — Live competitor price band from DB

Usage:
    python agent/pricing_agent.py                        # Interactive chat
    python agent/pricing_agent.py --query "..."          # Single query
    python agent/pricing_agent.py --demo                 # Run 5 demo queries

Requirements in .env:
    OPENAI_API_KEY=sk-...
    # OR
    ANTHROPIC_API_KEY=sk-ant-...
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime

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
log = logging.getLogger("pricing_agent")

# ── LangChain imports ─────────────────────────────────────────────────
try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_core.tools import tool
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError as e:
    log.error(f"LangChain import failed: {e}")
    log.error("Run: pip install langchain langchain-openai langchain-core")
    sys.exit(1)

# ── DemandSurge ML imports ───────────────────────────────────────────────
try:
    from src.pricing_model import get_price_recommendation, get_all_recommendations, load_features
    from src.demand_forecast import get_demand_forecast
    from db.models import get_db, Product, CompetitorPrice
except ImportError as e:
    log.error(f"ShopMind module import failed: {e}")
    log.error("Ensure you are running from the project root.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
# TOOL HELPERS
# ══════════════════════════════════════════════════════════════════════
def _find_product_row(product_name: str):
    """
    Fuzzy-matches a product name against the feature matrix.
    Returns the first matching row as a Series, or None.
    """
    try:
        df = load_features()
    except FileNotFoundError:
        return None, "Feature matrix not found. Run: python src/features.py"
    
    name_lower = product_name.strip().lower()

    # Try exact match first
    exact = df[df["product_name"].str.lower() == product_name]
    if not exact.empty:
        return exact.iloc[0], None
    
    # Then partial match
    partial = df[df["product"].str.lower().str.contains(name_lower, na=False)]
    if not partial.empty:
        return partial.iloc[0], None
    
    # List available products to help the agent
    available = df["product_name"].tolist()
    return None, (
        f"Product '{product_name}' not found. "
        f"Available products: {', '.join(available)}"
    )

def _get_competitor_summary(product_id: int)-> dict:
    """Loads competitor prices for a product directly from DB."""
    try:
        with get_db() as db:
            rows = db.query(CompetitorPrice).filter(CompetitorPrice.product_id == product_id).all()

        if not rows:
            return {"available": False, "message": "No competitor data in DB yet."}
        
        prices = [float(r.competitor_price) for r in rows]
        by_platform = {}
        for r in rows:
            by_platform.setdefault(r.platform, []).append(float(r.competitor_price))
        
        import statistics
        result = {
            "available":  True,
            "count":      len(prices),
            "median":     round(statistics.median(prices), 2),
            "min":        round(min(prices), 2),
            "max":        round(max(prices), 2),
            "by_platform": {p: {
                    "count":  len(v),
                    "median": round(statistics.median(v), 2),}
                for p, v in by_platform.items() },
        }
        return result
    except Exception as e:
        return {"available": False, "message": str(e)}
    
# ══════════════════════════════════════════════════════════════════════
# LANGCHAIN TOOLS
# ══════════════════════════════════════════════════════════════════════
@tool
def get_pricing_recommendation_tool(product_name: str)-> str:
    """
    Get an XGBoost-powered price recommendation for a specific product.

    Use this tool when the user asks:
    - What price should I set for [product]?
    - Is [product] priced correctly?
    - Should I raise or lower the price of [product]?
    - What is the optimal price for [product]?

    Input: the product name (partial names are supported, e.g. 'yoga mat', 'wallet')
    Output: recommended price, market comparison, demand context, and rationale.
    """
    row, error = _find_product_row(product_name)
    if error:
        return error
    
    try:
        rec = get_price_recommendation(product_row=row)
    except Exception as e:
        return f"Error generating recommendation: {e}"
    
    # Format a clean string response for the agent to reason over
    direction = "increase" if rec["pct_vs_current"] > 0 else ("decrease" if rec["pct_vs_current"] < 0 else "maintain") 

    return (
        f"PRICING RECOMMENDATION — {rec['product_name']}\n"
        f"Current price:     ${rec['current_price']:.2f}\n"
        f"Recommended price: ${rec['recommended_price']:.2f} "
        f"({rec['pct_vs_current']:+.1f}% — {direction})\n"
        f"Price range:       ${rec['lower_bound']:.2f} - ${rec['upper_bound']:.2f}\n"
        f"Market median:     ${rec['comp_price_med']:.2f} "
        f"({rec['pct_vs_market']:+.1f}% vs market)\n"
        f"Demand index:      {rec['trend_index']:.0f}/100\n"
        f"Confidence:        {rec['confidence']}\n"
        f"Rationale:         {rec['rationale']}"
    )

@tool
def get_demand_forecast_tool(category: str)-> str:
    """
    Get a Prophet-powered 30-day demand forecast for a product category.

    Use this tool when the user asks:
    - What is the demand outlook for [category]?
    - Is demand rising or falling for [category] products?
    - Should I stock up / discount based on upcoming demand?
    - What does the next 30 days look like for [category]?

    Valid categories: electronics, fashion, home_goods, sports
    Input: category name (e.g. 'electronics', 'sports', 'home_goods', 'fashion')
    Output: current trend index, 30-day forecast, direction, and pricing implication.
    """
    # Normalise category input
    cat = category.strip().lower().replace(" ", "_").replace("_", " ")

    # Map common synonyms
    synonyms = {
        "tech":         "electronics",
        "gadgets":      "electronics",
        "clothes":      "fashion",
        "clothing":     "fashion",
        "accessories":  "fashion",
        "home":         "home_goods",
        "household":    "home_goods",
        "fitness":      "sports",
        "gym":          "sports",
        "athletic":     "sports",
    }

    cat = synonyms.get(cat, cat)

    valid = ["electronics", "fashion", "home_goods", "sports"]
    if cat not in valid:
        return (
            f"Unknown category '{category}'. "
            f"Valid options: {', '.join(valid)}. "
            f"Please retry with one of these."
        )
    try:
        fc = get_demand_forecast(cat, days=30)
    except FileNotFoundError:
        return (
            f"No trained Prophet model found for '{cat}'. "
            f"Run: python src/demand_forecast.py --category {cat}"
        )
    except Exception as e:
        return f"Error fetching demand forecast: {e}"
    
    arrow = "↑" if fc["trend_direction"] == "rising" else("↓" if fc["trend_direction"] == "falling" else "→")

    return (
        f"DEMAND FORECAST — {cat.upper()} ({fc['keyword']})\n"
        f"Current trend index:  {fc['current_index']}/100\n"
        f"30-day forecast avg:  {fc['forecast_avg']}/100  {arrow} {fc['trend_direction'].upper()}\n"
        f"Forecast range:       {fc['forecast_low']} – {fc['forecast_high']}\n"
        f"Confidence band:      {fc['confidence_low']} – {fc['confidence_high']}\n"
        f"Demand signal:        {fc['demand_signal']}"
    )

@tool
def get_competitor_prices_tool(product_name: str)-> str:
    """
    Get competitor price data for a product from eBay and Etsy.

    Use this tool when the user asks:
    - How does [product] compare to competitors?
    - What are competitors charging for [product]?
    - Am I priced above or below the market for [product]?
    - What is the market price for [product]?

    Input: the product name (partial names supported)
    Output: competitor price band (min, median, max) by platform.
    """
    row, error = _find_product_row(product_name)
    if error:
        return error
    
    product_id = int(row["product_id"])
    current_price = float(row["current_price"])
    product_name = str(row["product_name"])

    comp = _get_competitor_summary(product_id)

    if not comp.get("available"):
        # Fall back to feature matrix competitor stats if DB is empty
        comp_med = float(row.get("comp_price_med", current_price))
        position = float(row.get("price_position", 0))
        return (
            f"COMPETITOR PRICES — {product_name}\n"
            f"Source: feature matrix (no live DB data yet)\n"
            f"Our price:        ${current_price:.2f}\n"
            f"Market median:    ${comp_med:.2f}\n"
            f"Price position:   {position:+.1f}% vs market median\n"
            f"Note: Run scripts/fetch_competitors.py to populate live competitor data."
        )
    
    position = ((current_price - comp["median"]) / comp["median"] * 100 if comp["median"] > 0 else 0)
    assessment = (
        "ABOVE market" if position > 10 else
        "BELOW market" if position < -10 else
        "IN LINE with market"
    )    

    platform_lines = "\n".join([
        f"  {p.capitalize():<8}: {v['count']} listings, median ${v['median']:.2f}"
        for p, v in comp.get("by_platform", {}).items()
    ])

    return (
        f"COMPETITOR PRICES — {product_name}\n"
        f"Our price:         ${current_price:.2f}\n"
        f"Market median:     ${comp['median']:.2f}  ({position:+.1f}% — {assessment})\n"
        f"Market range:      ${comp['min']:.2f} – ${comp['max']:.2f}\n"
        f"Total listings:    {comp['count']}\n"
        f"By platform:\n{platform_lines}"
    )

@tool
def get_weekly_review_tool(category: str = "all")-> str:
    """
    Run a full weekly pricing review across all products or a specific category.

    Use this tool when the user asks:
    - Give me a weekly pricing review
    - Which products need repricing?
    - What prices should I change this week?
    - Review all my products
    - What's underpriced / overpriced?

    Input: category name or 'all' for all products
    Output: ranked list of products with current vs recommended price and action priority.
    """
    try:
        recs = get_all_recommendations()
    except Exception as e:
        return f"Error running weekly review: {e}"
    
    if recs.empty:
        return "No recommendations available. Ensure the model is trained and features are built."
    
    # Filter by category if specified
    if category.lower() not in ("all", ""):
        cat = category.strip().lower().replace(" ", "_")
        recs = recs[recs["category"] == cat]

        if recs.empty:
            return f"No products found in category '{category}'."
        
    # Sort by absolute % change — biggest repricing opportunities first
    recs["abs_change"] = recs["pct_vs_current"].abs()
    recs = recs.sort_values("abs_change", ascending=False)

    # Build output
    lines = [f"WEEKLY PRICING REVIEW — {datetime.now().strftime('%Y-%m-%d')}\n"]

    urgent = recs[recs["abs_change"] >= 10]
    moderate = recs[recs["abs_change"] >= 5]
    stable = recs[recs["abs_change"] < 5]

    if not urgent.empty:
        lines.append(f" URGENT REPRICING ({len(urgent)} products — >10% change recommended):")
        for _, r in urgent.iterrows():
            arrow = "↑ RAISE" if r["pct_vs_current"] > 0 else "↓ LOWER"
            lines.append(
                f"  {r['product_name'][:35]:<35} "
                f"${r['current_price']:.2f} → ${r['recommended_price']:.2f} "
                f"{arrow} {r['pct_vs_current']:+.1f}%"
            )

    if not moderate.empty:
        lines.append(f"\n MODERATE CHANGES ({len(moderate)} products — 5–10% change):")
        for _, r in moderate.iterrows():
            arrow = "↑" if r["pct_vs_current"] > 0 else "↓"
            lines.append(
                f"  {r['product_name'][:35]:<35} "
                f"${r['current_price']:.2f} → ${r['recommended_price']:.2f} "
                f"{arrow} {r['pct_vs_current']:+.1f}%"
            )

    if not stable.empty:
        lines.append(f"\n STABLE ({len(stable)} products — <5% change, hold current price):")
        for _, r in stable.iterrows():
            lines.append(
                f"  {r['product_name'][:35]:<35} "
                f"${r['current_price']:.2f}  (no change needed)"
            )
    
    lines.append(
        f"\nSummary: {len(urgent)} urgent | {len(moderate)} moderate | {len(stable)} stable"
    )
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════
# AGENT SETUP
# ══════════════════════════════════════════════════════════════════════
TOOLS = [
    get_pricing_recommendation_tool,
    get_demand_forecast_tool,
    get_competitor_prices_tool,
    get_weekly_review_tool
]

SYSTEM_PROMPT = """You are DemandSurge, an AI pricing strategist for Shopify merchants/vendors.
You help merchants make data-driven pricing decisions using real market data,
demand forecasts, and competitor intelligence.

Use available tools when necessary to answer the user's request.

ALWAYS follow this reasoning format exactly:

Question: the input question you must answer
Thought: think step by step about which tool(s) to use and why
Action: the tool name to use (use aavailkable tools
Action Input: the input to pass to the tool
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to give a final answer
Final Answer: your complete, actionable pricing recommendation

GUIDELINES:
- Always ground recommendations in the tool output — never guess prices
- When recommending a price change, always state the specific dollar amount and percentage
- Mention demand context (rising/falling/stable) in every pricing recommendation
- If competitor data is available, compare our price to the market median
- Be concise and specific — merchants need clear actions, not vague advice
- If a product is not found, list available products to help the user
- For weekly reviews, prioritise urgent changes first

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

def build_agent(verbose: bool = True)-> AgentExecutor:
    """
    Builds and returns the LangChain ReAct AgentExecutor.

    Architecture:
    - LLM: GPT-4o-mini (fast, cheap, capable enough for structured reasoning)
    - Agent type: ReAct (Reason + Act) — interleaves thinking and tool use
    - Memory: ConversationBufferWindowMemory (last 5 turns)
    - Max iterations: 6 (prevents infinite loops)
    - handle_parsing_errors: True (recovers gracefully from malformed LLM output)
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set in .env. "
            "Add: OPENAI_API_KEY=sk-... to your .env file."
        )
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,        # Deterministic — pricing recommendations should be consistent
        api_key=api_key,
        max_tokens=1500,
    )

    prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT), 
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name = "agent_scratchpad"),
        ])

    agent = create_openai_functions_agent(
        llm=llm,
        tools=TOOLS,
        prompt=prompt
    )

    memory = ConversationBufferWindowMemory(
        k=5,                          # Remember last 5 conversation turns
        memory_key="chat_history",
        return_messages=True,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=verbose,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=False
    )

    log.info("Agent built successfully")
    log.info(f"  LLM     : gpt-4o-mini (temperature=0)")
    log.info(f"  Tools   : {[t.name for t in TOOLS]}")
    log.info(f"  Memory  : last {memory.k} turns")

    return executor

def run_query(executor: AgentExecutor, query: str)-> str:
    """Runs a single query through the agent and returns the response."""
    try:
        result = executor.invoke({"input": query})
        return result.get("output", "No response generated.")
    except Exception as e:
        return f"Agent error: {e}"
    
# ══════════════════════════════════════════════════════════════════════
# DEMO QUERIES
# ══════════════════════════════════════════════════════════════════════
DEMO_QUERIES = [
    (
        "What price should I set for the leather wallet heading into next month?",
        "Tests: product lookup + XGBoost recommendation + demand context"
    ),
    (
        "What's the demand outlook for electronics over the next 30 days?",
        "Tests: Prophet forecast tool + trend interpretation"
    ),
    (
        "How do my yoga mat prices compare to competitors?",
        "Tests: competitor price tool + market position"
    ),
    (
        "I have 200 units of running shoes sitting in inventory. What price helps me clear them in 3 weeks?",
        "Tests: inventory-aware reasoning + pricing urgency"
    ),
    (
        "Give me a full weekly pricing review — which products need to change this week?",
        "Tests: weekly review tool + prioritisation across all products"
    ),
]

def run_demo(executor: AgentExecutor):
    """Runs all 5 demo queries and prints results."""
    log.info("\n" + "="*60)
    log.info("DEMANDSURGE DEMO — 5 Interview Queries")
    log.info("="*60)

    for i, (query, note) in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"QUERY {i}/5: {query}")
        print(f"Note: {note}")
        print("="*60)

        response = run_query(executor, query)

        print(f"\nDEMANDSURGE RESPONSE:\n{response}")
        print()

        # Small pause between queries
        import time
        time.sleep(1)

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="DemandSurge LangChain ReAct Pricing Agent")
    parser.add_argument("--query",   type=str, default=None,
                        help="Run a single query and exit")
    parser.add_argument("--demo",    action="store_true",
                        help="Run all 5 demo queries")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress agent reasoning trace (verbose=False)")
    args = parser.parse_args()

    verbose = not args.quiet

     # Build agent
    try:
        executor = build_agent(verbose=verbose)
    except ValueError as e:
        log.error(str(e))
        return 1
    
    # ── Single query mode ─────────────────────────────────────────────
    if args.query:
        print(f"\nQuery: {args.query}\n")
        response = run_query(executor, args.query)
        print(f"Response:\n{response}")
        return 0
    
    # ── Demo mode ─────────────────────────────────────────────────────
    if args.demo:
        run_demo(executor)
        return 0
    
    # ── Interactive chat mode ─────────────────────────────────────────
    print("\n" + "="*60)
    print("DemandSurge Pricing Agent — Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'demo' to run all 5 demo queries")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if user_input.lower() == "demo":
            run_demo(executor)
            continue

        response = run_query(executor, user_input)
        print(f"\nDemandSurge: {response}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())