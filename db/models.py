from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Numeric,
    DateTime, Date, Text, ForeignKey, UniqueConstraint, CheckConstraint, Index
)
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL") 

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ── Context manager helper ────────────────────────────────────────────
class get_db:
    """Usage:
        with get_db() as db:
            products = db.query(Product).all()
    """
    def __enter__(self):
        self.db = SessionLocal()
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.db.rollback()
        self.db.close()

# ── TABLE 1: products ─────────────────────────────────────────────────
class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    base_price = Column(Numeric(10, 2), nullable=False)
    current_price = Column(Numeric(10, 2), nullable=False)
    inventory_qty = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    price_history = relationship("PriceHistory", back_populates="product", cascade="all, delete-orphan")
    competitor_prices = relationship("CompetitorPrice", back_populates="product", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Product id={self.id} name='{self.name}' price=${self.current_price}>"

    @property
    def margin(self):
        """Gross margin as a percentage."""
        if self.base_price and self.base_price > 0:
            return round(float((self.current_price - self.base_price) / self.base_price * 100), 1)
        return None

# ── TABLE 2: price_history ────────────────────────────────────────────
class PriceHistory(Base):
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"))
    price = Column(Numeric(10, 2), nullable=False)
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship
    product = relationship("Product", back_populates="price_history")

    __table_args__ = (
        Index("idx_price_history_product_id",  "product_id"),
        Index("idx_price_history_recorded_at", "recorded_at"),
    )

    def __repr__(self):
        return f"<PriceHistory product_id={self.product_id} price=${self.price} at={self.recorded_at}>"


# ── TABLE 3: demand_signals ───────────────────────────────────────────
class DemandSignal(Base):
    __tablename__ = "demand_signals"

    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String(255), nullable=False)
    trend_index = Column(Integer, nullable=False)
    region = Column(Text, nullable=False, default="US")
    week_date = Column(Date, nullable=False)
    created_at  = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("keyword", "region", "week_date", name="uq_demand_signal"),
        CheckConstraint("trend_index >= 0 AND trend_index <= 100", name="chk_trend_index_range"),
        Index("idx_demand_signals_keyword",   "keyword"),
        Index("idx_demand_signals_week_date", "week_date"),
    )

    def __repr__(self):
        return f"<DemandSignal keyword='{self.keyword}' index={self.trend_index} week={self.week_date}>"

# ── TABLE 4: competitor_prices ────────────────────────────────────────
class CompetitorPrice(Base):
    __tablename__ = "competitor_prices"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"))
    platform = Column(String(50), nullable=False)
    competitor_price = Column(Numeric(10, 2), nullable=False)
    listing_url = Column(Text)
    scraped_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship
    product = relationship("Product", back_populates="competitor_prices")

    __table_args__ = (
        Index("idx_competitor_prices_product",  "product_id"),
        Index("idx_competitor_prices_platform", "platform"),
    )

    def __repr__(self):
        return f"<CompetitorPrice product_id={self.product_id} platform={self.platform} price=${self.competitor_price}>"
    
# ── QUERY HELPERS ─────────────────────────────────────────────────────
def get_product_by_name(db, name: str):
    """Case-insensitive partial match — used by agent tools."""
    return (
        db.query(Product)
        .filter(Product.name.ilike(f"%{name}%"))
        .first()
    )

def get_products_by_category(db, category: str):
    return db.query(Product).filter(Product.category == category).all()

def get_price_band(db, product_id: int) -> dict:
    """
    Returns competitor price statistics for a product.
    Falls back to empty dict if no competitor data exists.
    """
    from sqlalchemy import func, text

    result = db.execute(
        text("""
            SELECT
                COUNT(*) AS comp_count,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY competitor_price) AS p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY competitor_price) AS median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY competitor_price) AS p75,
                AVG(competitor_price) AS avg_price
            FROM competitor_prices
            WHERE product_id = :pid
        """),
        {"pid": product_id}
    ).fetchone()

    if not result or result.comp_count == 0:
        return {}

    return {
        "comp_count":  int(result.comp_count),
        "p25":         round(float(result.p25), 2),
        "median":      round(float(result.median), 2),
        "p75":         round(float(result.p75), 2),
        "avg_price":   round(float(result.avg_price), 2),
    }


def get_latest_trend(db, keyword: str, region: str = "US") -> dict:
    """Returns the most recent demand signal for a keyword."""
    signal = (
        db.query(DemandSignal)
        .filter(DemandSignal.keyword == keyword, DemandSignal.region == region)
        .order_by(DemandSignal.week_date.desc())
        .first()
    )
    if not signal:
        return {}
    return {
        "keyword":     signal.keyword,
        "trend_index": signal.trend_index,
        "week_date":   str(signal.week_date),
    }


def upsert_demand_signal(db, keyword: str, trend_index: int, week_date, region: str = "US"):
    """Insert or update a demand signal row (ON CONFLICT DO UPDATE)."""
    from sqlalchemy.dialects.postgresql import insert

    stmt = insert(DemandSignal).values(
        keyword=keyword,
        trend_index=trend_index,
        region=region,
        week_date=week_date,
    ).on_conflict_do_update(
        constraint="uq_demand_signal",
        set_={"trend_index": trend_index}
    )
    db.execute(stmt)
    db.commit()


# ── INIT (create all tables if they don't exist) ──────────────────────
if __name__ == "__main__":
    print("Creating all tables...")
    Base.metadata.create_all(bind=engine)
    print("Done. Tables created:")
    for table in Base.metadata.sorted_tables:
        print(f"   {table.name}")

