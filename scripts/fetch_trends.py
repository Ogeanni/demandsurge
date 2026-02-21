"""
scripts/fetch_trends.py
Fetches Google Trends data for DemandSurge product categories using Pytrends.

Usage:
    python scripts/fetch_trends.py              # Fetch all categories
    python scripts/fetch_trends.py --category electronics
    python scripts/fetch_trends.py --dry-run    # Print data without saving

Dependencies:
    pip install pytrends
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError


# ── Path setup — must be first, before any local imports ─────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from db.models import get_db, DemandSignal, upsert_demand_signal

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))


# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",)
log = logging.getLogger("fetch_trends")


# ── Keyword map: category → search keywords ───────────────────────────
# Each category maps to 1 primary keyword (Pytrends works best with
# focused queries — we'll fetch categories individually to stay under
# the rate limit).
CATEGORY_KEYWORDS = {
    "electronics": [
        "wireless headphones",
        "bluetooth speaker",
        "smart watch",
        "mechanical keyboard",
    ],
    "fashion": [
        "leather wallet",
        "tote bag",
        "minimalist watch",
        "wool beanie",
    ],
    "home_goods": [
        "bamboo cutting board",
        "stainless steel water bottle",
        "soy candle",
        "essential oil diffuser",
    ],
    "sports": [
        "yoga mat",
        "resistance bands",
        "running shoes",
        "foam roller",
    ],
}

VALID_CATEGORIES = list(CATEGORY_KEYWORDS.keys())

# Flat list for convenience: keyword -> category
KEYWORD_TO_CATEGORY = {
    kw: cat
    for cat, keywords in CATEGORY_KEYWORDS.items()
    for kw in keywords
}

# ── Pytrends client ───────────────────────────────────────────────────
def build_client():
    """Initialise a Pytrends session with sensible defaults."""
    return TrendReq(
        hl='en-US',
        tz=360,                 # US Central (adjust if needed)
        timeout=(10, 25),       # (connect_timeout, read_timeout)
        retries=3,
        backoff_factor=2,
        )

# ── Core fetch function ───────────────────────────────────────────────
def fetch_keyword(client, keyword, timeframe="today 12-m"):
    """
    Returns list of dicts:
        [{"keyword": str, "trend_index": int, "week_date": date, "region": "US"}, ...]
    Returns [] if no data found. Auto-retries on 429 with exponential backoff.
    """
    log.info(f"  Fetching: '{keyword}'")

    client.build_payload(
        kw_list=[keyword],
        cat=0,
        timeframe=timeframe,
        geo="US",
        gprop="",
    )

    # Exponential backoff on 429: wait 60s, 120s, 240s before giving up
    df = None
    for attempt in range(3):
        try:
            df = client.interest_over_time()
            break
        except Exception as e:
            if "429" in str(e):
                wait = 60 * (2 ** attempt)
                log.warning(f"  429 rate limit hit — waiting {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
                client = build_client()  # fresh session after cooldown
            else:
                raise
    if df is None:
        log.error(f"  All retries exhausted for '{keyword}'")
        return []

    if df is None or df.empty:
        log.warning(f"  No data returned for '{keyword}'")
        return []

    # Remove the isPartial flag column before iterating
    if "isPartial" in df.columns:
        df = df[df["isPartial"] == False].copy()
        df = df.drop(columns=["isPartial"], errors="ignore")

    rows = []
    for week_date, row in df.iterrows():
        value = row.get(keyword, 0)
        rows.append({
            "keyword":     keyword,
            "trend_index": int(value),
            "week_date":   week_date.date(),
            "region":      "US",
        })

    if rows:
        avg = int(sum(r["trend_index"] for r in rows) / len(rows))
        log.info(f"  OK '{keyword}': {len(rows)} weeks  "
                 f"(latest={rows[-1]['trend_index']}, avg={avg})")
    return rows


# ── Save to DB ────────────────────────────────────────────────────────
def save_rows(rows: list[dict], dry_run: bool = False)-> int:
    """
    Upsert rows into demand_signals.
    Returns the number of rows written.
    """
    if not rows:
        return 0

    if dry_run:
        log.info(f"  [DRY RUN] Would upsert {len(rows)} rows")
        for r in rows[-3:]:  # Show last 3 as sample
            log.info(f"{r}")
        return 0

    with get_db() as db:
        for r in rows:
            upsert_demand_signal(
                db,
                keyword=r["keyword"],
                trend_index=r["trend_index"],
                week_date=r["week_date"],
                region=r["region"],
            )

    log.info(f"  Upserted {len(rows)} rows into demand_signals")
    return len(rows)

# ── Rate-limit safe fetch loop ────────────────────────────────────────
def fetch_category(
    category: str,
    dry_run: bool = False,
    sleep_between: float = 3.0,
    timeframe: str = "today 12-m",
) -> dict:
    """
    Fetch all keywords for a category with rate-limit protection.
    Returns a summary dict.
    """
    category = str(category).strip().lower()
    keywords = CATEGORY_KEYWORDS.get(category)
  
    if not keywords:
        log.error(f"Unknown category: '{category}'  valid={VALID_CATEGORIES}")
        return {"category": category, "keywords": 0, "rows": 0, "errors": 1}

    log.info(f"\n{'='*55}")
    log.info(f"Category: {category.upper()}  ({len(keywords)} keywords)")
    log.info(f"{'='*55}")

    client = build_client()
    total_rows = 0
    errors = 0

    for i, keyword in enumerate(keywords):
        try:
            rows = fetch_keyword(client, keyword, timeframe=timeframe)
            saved = save_rows(rows, dry_run=dry_run)
            total_rows += saved

        except ResponseError as e:
            errors += 1
            log.error(f"  ResponseError for '{keyword}': {e}")
            if "429" in str(e):
                log.warning("  Rate limited — waiting 60s then retrying once...")
                time.sleep(60)
                try:
                    client = build_client()
                    rows = fetch_keyword(client, keyword, timeframe=timeframe)
                    saved = save_rows(rows, dry_run=dry_run)
                    total_rows += saved
                    errors -= 1
                except Exception as retry_err:
                    log.error(f"  Retry also failed: {retry_err}")

        except Exception as e:
            errors += 1
            log.error(f"  Error for '{keyword}': {e}")

        if i < len(keywords) - 1:
            log.info(f"  Sleeping {sleep_between}s...")
            time.sleep(sleep_between)

    log.info(f"  Category done: {total_rows} rows, {errors} errors")
    return {"category": category, "keywords": len(keywords), "rows": total_rows, "errors": errors}


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch Google Trends for DemandSurge")
    parser.add_argument("--category", "-c", choices=VALID_CATEGORIES, default=None,
                        help="Single category to fetch (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without saving to DB")
    parser.add_argument("--timeframe", default="today 12-m",
                        help="Pytrends timeframe (default: 'today 12-m')")
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="Seconds between keyword requests (default: 3.0)")
    args = parser.parse_args()

    # Build category list explicitly — fresh copy, no reference aliasing
    if args.category:
        categories = [str(args.category).strip().lower()]
    else:
        categories = [str(c) for c in VALID_CATEGORIES]

    log.info("ShopMind — Pytrends Ingestion")
    log.info(f"Categories : {categories}")
    log.info(f"Timeframe  : {args.timeframe}")
    log.info(f"Sleep      : {args.sleep}s between keywords")
    log.info(f"Dry run    : {args.dry_run}")

    start = datetime.now()
    results = []

    for i, category in enumerate(categories):
        result = fetch_category(
            category=category,
            dry_run=args.dry_run,
            sleep_between=args.sleep,
            timeframe=args.timeframe,
        )
        results.append(result)

        if i < len(categories) - 1:
            pause = args.sleep * 4
            log.info(f"\nPausing {pause}s between categories...")
            time.sleep(pause)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = int((datetime.now() - start).total_seconds())
    total_rows   = sum(r["rows"]   for r in results)
    total_errors = sum(r["errors"] for r in results)

    log.info("\n" + "="*55)
    log.info("SUMMARY")
    log.info("="*55)
    for r in results:
        icon = "OK" if r["errors"] == 0 else "!!"
        log.info(f"  [{icon}]  {r['category']:<15}  {r['rows']:>4} rows   {r['errors']} errors")
    log.info(f"\n  Total rows : {total_rows}")
    log.info(f"  Errors     : {total_errors}")
    log.info(f"  Elapsed    : {elapsed}s")

    if not args.dry_run and total_rows > 0:
        with get_db() as db:
            count = db.query(DemandSignal).count()
            log.info(f"\n  DB check — demand_signals total rows: {count}")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
