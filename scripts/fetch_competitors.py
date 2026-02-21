"""
scripts/fetch_competitors.py
Scrapes competitor prices from eBay and Etsy for DemandSurge products.

Usage:
    python scripts/fetch_competitors.py              # Fetch all products
    python scripts/fetch_competitors.py --platform ebay
    python scripts/fetch_competitors.py --platform etsy
    python scripts/fetch_competitors.py --category electronics
    python scripts/fetch_competitors.py --dry-run
"""

import os
import sys
import time
import logging
import argparse
import statistics
from datetime import datetime
import requests

# ── Path setup ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_competitors")

# ── Optional imports with clear errors ───────────────────────────────
try:
    from db.models import get_db, Product, CompetitorPrice
except ImportError as e:
    log.error(f"Could not import db.models: {e}")
    log.error("Run from project root: python scripts/fetch_competitors.py")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────
EBAY_APP_ID  = os.getenv("EBAY_APP_ID", "")
ETSY_API_KEY = os.getenv("ETSY_API_KEY", "")

# Max listings to fetch per product per platform
MAX_RESULTS = 10

# How many seconds to wait between API calls
SLEEP_BETWEEN = 1.5

# ── eBay Fetcher ──────────────────────────────────────────────────────
def fetch_ebay_prices(keyword: str, max_results: int = MAX_RESULTS)-> list[dict]:
    """
    Fetches Buy It Now listings from eBay Finding API.

    Returns list of dicts:
        [{"price": float, "title": str, "url": str, "platform": "ebay"}, ...]

    eBay Finding API docs:
    https://developer.ebay.com/devzone/finding/callref/findItemsByKeywords.html
    """
    if not EBAY_APP_ID:
        log.warning("  EBAY_APP_ID not set in .env — skipping eBay")
        return []
    
    endpoint = "https://svcs.ebay.com/services/search/FindingService/v1"
    params = {
        "OPERATION-NAME":         "findItemsByKeywords",
        "SERVICE-VERSION":        "1.0.0",
        "SECURITY-APPNAME":       EBAY_APP_ID,
        "RESPONSE-DATA-FORMAT":   "JSON",
        "REST-PAYLOAD":           "",
        "keywords":               keyword,
        "paginationInput.entriesPerPage": max_results,
        # Filter to Buy It Now only (ListingType = FixedPrice or StoreInventory)
        "itemFilter(0).name":     "ListingType",
        "itemFilter(0).value(0)": "FixedPrice",
        "itemFilter(0).value(1)": "StoreInventory",
        # USD only
        "itemFilter(1).name":     "Currency",
        "itemFilter(1).value":    "USD",
        # Exclude listings with no price
        "itemFilter(2).name":     "MinPrice",
        "itemFilter(2).value":    "1.00",
        # Sort by best match
        "sortOrder":              "BestMatch",
    }

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    
    except requests.RequestException as e:
        log.error(f"  eBay API error for '{keyword}': {e}")
        return []
    
    try:
        items = (data.get("findItemsByKeywordsResponse", [{}])[0].get("searchResult", [{}])[0].get("item", []))
    except (KeyError, IndexError):
        log.warning(f"  eBay: unexpected response structure for '{keyword}'")
        return []
    
    results = []
    for item in items:
        try:
            price_str = (item.get("sellingStatus", [{}])[0].get("currentPrice", [{}])[0].get("__value__", None))
            if price_str is None:
                continue

            price = float(price_str)
            if price <= 0:
                continue

            title = item.get("title", [""])[0]
            url   = item.get("viewItemURL", [""])[0]

            results.append({
                "price":    price,
                "title":    title,
                "url":      url,
                "platform": "ebay",
            })
        except (KeyError, ValueError, IndexError):
            continue

    log.info(f"  eBay '{keyword}': {len(results)} listings found")
    return results

# ── Etsy Fetcher ──────────────────────────────────────────────────────
def fetch_etsy_prices(keyword: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """
    Fetches active listings from Etsy Open API v3.

    Returns list of dicts:
        [{"price": float, "title": str, "url": str, "platform": "etsy"}, ...]

    Etsy API docs:
    https://developers.etsy.com/documentation/reference#operation/findAllListingsActive
    """
    if not ETSY_API_KEY:
        log.warning("  ETSY_API_KEY not set in .env — skipping Etsy")
        return []
    
    endpoint = "https://openapi.etsy.com/v3/application/listings/active"
    headers  = {"x-api-key": ETSY_API_KEY}
    params   = {
        "keywords": keyword,
        "limit":    max_results,
        "sort_on":  "score",         # best match
        "sort_order": "desc",
    }

    try:
        resp = requests.get(endpoint, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            log.error("  Etsy 401 Unauthorized — check ETSY_API_KEY in .env")
        elif resp.status_code == 403:
            log.error("  Etsy 403 Forbidden — your API key may need Etsy approval")
        else:
            log.error(f"  Etsy HTTP error for '{keyword}': {e}")
        return []
    except requests.RequestException as e:
        log.error(f"  Etsy API error for '{keyword}': {e}")
        return []
    
    results = []
    for listing in data.get("results", []):
        try:
            # Etsy prices are in the listing currency's smallest unit (cents for USD)
            price_data = listing.get("price", {})
            amount     = price_data.get("amount", 0)
            divisor    = price_data.get("divisor", 100)

            if divisor == 0:
                continue

            price = float(amount) / float(divisor)
            if price <= 0:
                continue

            title    = listing.get("title", "")
            listing_id = listing.get("listing_id", "")
            shop_id  = listing.get("shop_id", "")
            url      = f"https://www.etsy.com/listing/{listing_id}"

            results.append({
                "price":    price,
                "title":    title,
                "url":      url,
                "platform": "etsy",
            })
        except (KeyError, ValueError, TypeError):
            continue

    log.info(f"  Etsy '{keyword}': {len(results)} listings found")
    return results

# ── Price band helper ─────────────────────────────────────────────────
def compute_price_band(prices: list[float]) -> dict:
    """Returns summary stats for a list of prices."""
    if not prices:
        return {}
    sorted_prices = sorted(prices)
    n = len(sorted_prices)
    return {
        "count":  n,
        "min":    round(min(sorted_prices), 2),
        "max":    round(max(sorted_prices), 2),
        "median": round(statistics.median(sorted_prices), 2),
        "mean":   round(statistics.mean(sorted_prices), 2),
        "p25":    round(sorted_prices[max(0, int(n * 0.25) - 1)], 2),
        "p75":    round(sorted_prices[min(n - 1, int(n * 0.75))], 2),
    }

# ── Save to DB ────────────────────────────────────────────────────────
def save_competitor_prices(
    product_id: int,
    listings: list[dict],
    dry_run: bool = False,
) -> int:
    """
    Inserts competitor price rows for a product.
    Does NOT upsert — each scrape run adds a fresh timestamped snapshot.
    Returns count saved.
    """
    if not listings:
        return 0

    if dry_run:
        log.info(f"  [DRY RUN] Would save {len(listings)} listings for product {product_id}")
        for l in listings[:3]:
            log.info(f"    [{l['platform']}] ${l['price']:.2f}  —  {l['title'][:50]}")
        return len(listings)
    
    with get_db() as db:
        for listing in listings:
            row = CompetitorPrice(
                product_id=product_id,
                platform=listing["platform"],
                competitor_price=listing["price"],
                listing_url=listing.get("url", ""),
                scraped_at=datetime.utcnow(),
            )
            db.add(row)
        db.commit()

    log.info(f"  Saved {len(listings)} competitor prices for product {product_id}")
    return len(listings)


# ── Per-product fetch ─────────────────────────────────────────────────
def fetch_product(
    product,
    platforms: list[str],
    dry_run: bool = False,
    sleep_between: float = SLEEP_BETWEEN,
) -> dict:
    """
    Fetches competitor prices for a single product from all platforms.
    Uses the product name as the search keyword.
    Returns summary dict.
    """
    # Build search keyword from product name
    # Strip generic words that hurt search quality
    keyword = product.name
    for strip in ["Set", "1L", "6mm", "TKL", "7-in-1", "5 levels"]:
        keyword = keyword.replace(strip, "").strip()
    keyword = " ".join(keyword.split())   # collapse whitespace

    log.info(f"\n  Product [{product.id}]: {product.name}")
    log.info(f"  Search keyword: '{keyword}'")
    log.info(f"  Current price:  ${product.current_price}  |  Category: {product.category}")

    all_listings = []
    total_saved  = 0

    if "ebay" in platforms:
        listings = fetch_ebay_prices(keyword)
        saved = save_competitor_prices(product.id, listings, dry_run=dry_run)
        all_listings.extend(listings)
        total_saved += saved
        if len(platforms) > 1:
            time.sleep(sleep_between)

    if "etsy" in platforms:
        listings = fetch_etsy_prices(keyword)
        saved    = save_competitor_prices(product.id, listings, dry_run=dry_run)
        all_listings.extend(listings)
        total_saved += saved

    # Log price band comparison vs our price
    prices = [l["price"] for l in all_listings]
    if prices:
        band = compute_price_band(prices)
        our_price = float(product.current_price)
        position = "ABOVE" if our_price > band["median"] else "BELOW"
        pct_diff  = abs(our_price - band["median"]) / band["median"] * 100

        log.info(
            f" Price band: min=${band['min']}  "
            f"p25=${band['p25']}  median=${band['median']}  "
            f"p75=${band['p75']}  max=${band['max']}"
        )
        log.info(
            f"  Our price ${our_price:.2f} is {pct_diff:.1f}% {position} market median"
        )

    return {
        "product_id":   product.id,
        "product_name": product.name,
        "listings":     len(all_listings),
        "saved":        total_saved,
    }


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fetch competitor prices from eBay and Etsy for ShopMind"
    )
    parser.add_argument(
        "--platform", "-p",
        choices=["ebay", "etsy", "both"],
        default="both",
        help="Which platform to scrape (default: both)",
    )
    parser.add_argument(
        "--category", "-c",
        choices=["electronics", "fashion", "home_goods", "sports"],
        default=None,
        help="Only fetch products in this category (default: all)",
    )
    parser.add_argument(
        "--product-id",
        type=int,
        default=None,
        help="Only fetch a single product by ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without saving to DB",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=SLEEP_BETWEEN,
        help=f"Seconds between API calls (default: {SLEEP_BETWEEN})",
    )
    args = parser.parse_args()

    platforms = ["ebay", "etsy"] if args.platform == "both" else [args.platform]

    log.info("ShopMind — Competitor Price Ingestion")
    log.info(f"Platforms  : {platforms}")
    log.info(f"Category   : {args.category or 'all'}")
    log.info(f"Dry run    : {args.dry_run}")
    log.info(f"Sleep      : {args.sleep}s between calls")

    # Check API keys
    if "ebay" in platforms and not EBAY_APP_ID:
        log.warning("EBAY_APP_ID not set — eBay will be skipped")
    if "etsy" in platforms and not ETSY_API_KEY:
        log.warning("ETSY_API_KEY not set — Etsy will be skipped")

    # Load products
    with get_db() as db:
        query = db.query(Product)
        if args.product_id:
            query = query.filter(Product.id == args.product_id)
        elif args.category:
            query = query.filter(Product.category == args.category)
        products = query.order_by(Product.category, Product.id).all()

    if not products:
        log.error("No products found. Check your DB has seed data.")
        return 1

    log.info(f"\nFetching competitor prices for {len(products)} products...\n")

    start   = datetime.now()
    results = []

    for i, product in enumerate(products):
        result = fetch_product(
            product=product,
            platforms=platforms,
            dry_run=args.dry_run,
            sleep_between=args.sleep,
        )
        results.append(result)

        # Sleep between products to avoid hammering APIs
        if i < len(products) - 1:
            time.sleep(args.sleep)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed     = int((datetime.now() - start).total_seconds())
    total_saved = sum(r["saved"] for r in results)
    no_data     = [r for r in results if r["listings"] == 0]

    log.info("\n" + "="*55)
    log.info("SUMMARY")
    log.info("="*55)
    for r in results:
        icon = "OK" if r["listings"] > 0 else "!!"
        log.info(
            f"  [{icon}] [{r['product_id']:>2}] {r['product_name'][:35]:<35} "
            f"{r['listings']:>3} listings"
        )

    log.info(f"\n  Total listings saved : {total_saved}")
    log.info(f"  Products with no data: {len(no_data)}")
    log.info(f"  Elapsed              : {elapsed}s")

    if not args.dry_run:
        with get_db() as db:
            count = db.query(CompetitorPrice).count()
            log.info(f"\n  DB check — competitor_prices total rows: {count}")

    # Print verification query to run manually
    log.info("\n  Verify in psql:")
    log.info("  SELECT p.name, cp.platform, COUNT(*) as n,")
    log.info("  ROUND(AVG(cp.competitor_price)::numeric, 2) as avg_price")
    log.info("  FROM competitor_prices cp")
    log.info("  JOIN products p ON p.id = cp.product_id")
    log.info("  GROUP BY p.name, cp.platform ORDER BY p.name;")

    return 0


if __name__ == "__main__":
    sys.exit(main())




