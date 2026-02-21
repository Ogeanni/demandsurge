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
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_history_aware_retriever
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from typing import TypedDict, Annotated, Sequence
    import operator
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    from langchain.tools import Tool
    import os
    from datetime import datetime
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
    partial = df[df["product_name"].str.lower().str.contains(name_lower, na=False)]
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

# STEP 3: Build the Agent
class AgentState(TypedDict):
    """State of the agent"""
    messages: Annotated[Sequence[HumanMessage | AIMessage | SystemMessage], operator.add]
    user_input: str
    chat_history: list

class DemandSurgeAgent:
    """
    LangGraph-based conversational agent for DemandSurge
    """
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model = "gpt-4o-mini",
            temperature = 0,
            streaming = True
        )
        # Create the agent
        self.agent = create_openai_functions_agent(
            llm = self.llm,
            tools = TOOLS,
            prompt = self._create_prompt()
        )
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent = self.agent,
            tools = TOOLS,
            verbose = True,
            handle_parsing_errors = True,
            max_iterations = 3
        )
        # Message history storage
        self.message_histories = {}

    def _create_prompt(self):
        """Create the agent prompt template"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are DemandSurge, an AI pricing strategist for Shopify merchants/vendors.
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
                - For weekly reviews, prioritise urgent changes first"""), 
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name = "agent_scratchpad"),
        ])
    
    def get_session_history(self, session_id: str):
        """Get or create chat history for a session"""
        if session_id not in self.message_histories:
            self.message_histories[session_id] = ChatMessageHistory()
        return self.message_histories[session_id]
    
    def run_query(self, user_query: str, session_id: str = "default")-> dict:
        """
        Main chat method
        
        Args:
            user_input: User's message
            session_id: Session identifier for conversation history
            
        Returns:
            dict with response and metadata
        """
        # Get chat history
        history = self.get_session_history(session_id)

        # Prepare input with history
        chat_history_message = history.messages[-5:] # Keep last 5 messages

        # Invoke agent
        result = self.agent_executor.invoke({
            "input": user_query,
            "chat_history": chat_history_message
        })

        # Save to history
        history.add_user_message(user_query)
        history.add_ai_message(result["output"])

        return {
            "response": result["output"],
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_history(self, session_id: str = "default"):
        """Clear chat history for a session"""
        if session_id in self.message_histories:
            self.message_histories[session_id].clear()

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

def run_demo():
    """Runs all 5 demo queries and prints results."""
    log.info("\n" + "="*60)
    log.info("DEMANDSURGE DEMO — 5 Interview Queries")
    log.info("="*60)

    for i, (query, note) in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"QUERY {i}/5: {query}")
        print(f"Note: {note}")
        print("="*60)

        agent = DemandSurgeAgent()
        response = agent.run_query(query)

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
        agent = DemandSurgeAgent()
    except ValueError as e:
        log.error(str(e))
        return 1
    
    # ── Single query mode ─────────────────────────────────────────────
    if args.query:
        print(f"\nQuery: {args.query}\n")
        response = agent.run_query(args.query)
        print(f"Response:\n{response}")
        return 0
    
    # ── Demo mode ─────────────────────────────────────────────────────
    if args.demo:
        run_demo()
        return 0
    
    # ── Interactive chat mode ─────────────────────────────────────────
    print("\n" + "="*60)
    print("DemandSurge Pricing Agent — Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'demo' to run all 5 demo queries")
    print("="*60 + "\n")

    while True:
        try:
            user_query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_query:
            continue
        if user_query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if user_query.lower() == "demo":
            run_demo()
            continue

        response = agent.run_query(user_query)
        print(f"\nDemandSurge: {response}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())