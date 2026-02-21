-- Drop tables in reverse dependency order (safe to re-run)
DROP TABLE IF EXISTS competitor_prices CASCADE;
DROP TABLE IF EXISTS demand_signals CASCADE;
DROP TABLE IF EXISTS price_history CASCADE;
DROP TABLE IF EXISTS products CASCADE;

-- ============================================================
-- TABLE 1: products
-- Core product catalog seeded with test data
-- ============================================================
CREATE TABLE IF NOT EXISTS products(
    id               SERIAL PRIMARY KEY,
    name             VARCHAR(255) NOT NULL,
    category         VARCHAR(100) NOT NULL,
    base_price       NUMERIC(10, 2) NOT NULL,        -- Original / cost price
    current_price    NUMERIC(10, 2) NOT NULL,        -- Live selling price
    inventory_qty    INTEGER NOT NULL DEFAULT 0,
    created_at       TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMP NOT NULL DEFAULT NOW()

);

-- ============================================================
-- TABLE 2: price_history
-- Tracks every price change for a product over time
-- ============================================================
CREATE TABLE IF NOT EXISTS price_history(
    id           SERIAL PRIMARY KEY,
    product_id   INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    price        NUMERIC(10, 2) NOT NULL,
    recorded_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ============================================================
-- TABLE 3: demand_signals
-- Weekly Google Trends data per keyword/category
-- ============================================================
CREATE TABLE demand_signals(
    id           SERIAL PRIMARY KEY,
    keyword      VARCHAR(255) NOT NULL,
    trend_index  INTEGER NOT NULL CHECK (trend_index >= 0 AND trend_index <= 100),
    region       VARCHAR(10) NOT NULL DEFAULT 'US',
    week_date    DATE NOT NULL,
    created_at   TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (keyword, region, week_date)              -- Upsert-safe constraint
);

-- ============================================================
-- TABLE 4: competitor_prices
-- Competitor listings scraped from eBay & Etsy
-- ============================================================
CREATE TABLE IF NOT EXISTS competitor_prices(
    id                SERIAL PRIMARY KEY,
    product_id        INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    platform          VARCHAR(50) NOT NULL,          -- 'ebay' | 'etsy'
    competitor_price  NUMERIC(10, 2) NOT NULL,
    listing_url       TEXT,
    scraped_at        TIMESTAMP NOT NULL DEFAULT NOW()
);


-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX idx_price_history_product_id   ON price_history(product_id);
CREATE INDEX idx_price_history_recorded_at  ON price_history(recorded_at);
CREATE INDEX idx_demand_signals_keyword      ON demand_signals(keyword);
CREATE INDEX idx_demand_signals_week_date    ON demand_signals(week_date);
CREATE INDEX idx_competitor_prices_product   ON competitor_prices(product_id);
CREATE INDEX idx_competitor_prices_platform  ON competitor_prices(platform);


-- ============================================================
-- SEED DATA: 20 products across 4 categories
-- ============================================================
INSERT INTO products (name, category, base_price, current_price, inventory_qty) VALUES

-- Electronics (5 products)
('Wireless Noise-Cancelling Headphones',  'electronics',  65.00,  89.99,  120),
('Portable Bluetooth Speaker',            'electronics',  30.00,  49.99,  85),
('Smart Watch Fitness Tracker',           'electronics',  45.00,  74.99,  60),
('USB-C Hub 7-in-1',                      'electronics',  18.00,  34.99,  200),
('Mechanical Keyboard TKL',               'electronics',  55.00,  79.99,  45),

-- Fashion (5 products)
('Genuine Leather Bifold Wallet',         'fashion',      12.00,  29.99,  300),
('Canvas Tote Bag',                       'fashion',       8.00,  19.99,  250),
('Minimalist Watch Leather Strap',        'fashion',      22.00,  49.99,  80),
('Wool Beanie Hat',                       'fashion',       7.00,  18.99,  180),
('Polarised Sunglasses',                  'fashion',      15.00,  39.99,  150),

-- Home Goods (5 products)
('Bamboo Cutting Board Set',              'home_goods',   14.00,  34.99,  95),
('Stainless Steel Water Bottle 1L',       'home_goods',   10.00,  24.99,  220),
('Scented Soy Candle Set',               'home_goods',    9.00,  22.99,  160),
('Linen Throw Blanket',                   'home_goods',   18.00,  44.99,  70),
('Essential Oil Diffuser',               'home_goods',   20.00,  39.99,  110),

-- Sports (5 products)
('Yoga Mat Non-Slip 6mm',                 'sports',       14.00,  32.99,  140),
('Resistance Bands Set (5 levels)',       'sports',        8.00,  19.99,  310),
('Running Shoes Lightweight',             'sports',       40.00,  79.99,  200),
('Foam Roller Deep Tissue',              'sports',       12.00,  27.99,  130),
('Jump Rope Speed Cable',                'sports',        6.00,  14.99,  400);

-- Seed initial price_history from current product prices
INSERT INTO price_history (product_id, price, recorded_at)
SELECT id, current_price, NOW() FROM products;

-- ============================================================
-- HELPER VIEWS
-- ============================================================

-- Latest competitor price band per product

CREATE OR REPLACE VIEW competitor_price_bands AS

SELECT
    product_id,
    COUNT(*) AS comp_count,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY competitor_price):: NUMERIC, 2) AS p25_price,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY competitor_price):: NUMERIC, 2) AS median_price,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY competitor_price):: NUMERIC, 2) AS p75_price,
    ROUND(AVG(competitor_price):: NUMERIC, 2) AS avg_price,
    MAX(scraped_at) AS last_scraped

FROM competitor_prices
GROUP BY product_id;

-- Latest demand trend per keyword
CREATE OR REPLACE VIEW latest_demand_signals AS
SELECT DISTINCT ON (keyword, region)
    keyword,
    region,
    trend_index,
    week_date
FROM demand_signals
ORDER BY keyword, region, week_date DESC;

-- Full product dashboard view
CREATE OR REPLACE VIEW product_dashboard AS
SELECT
    p.id,
    p.name,
    p.category,
    p.current_price,
    p.inventory_qty,
    cpb.median_price,
    cpb.p25_price,
    cpb.p75_price,
    cpb.comp_count,
    ROUND(((p.current_price - cpb.median_price) / NULLIF(cpb.median_price, 0) * 100)::NUMERIC, 1) AS pct_vs_median
FROM products p
LEFT JOIN competitor_price_bands cpb ON cpb.product_id = p.id;

-- ============================================================
-- VERIFY
-- ============================================================
SELECT
    'products' AS table_name, COUNT(*) AS row_count FROM products
UNION ALL SELECT
    'price_history', COUNT(*) FROM price_history
UNION ALL SELECT
    'demand_signals', COUNT(*) FROM demand_signals
UNION ALL SELECT
    'competitor_prices', COUNT(*) FROM competitor_prices;



