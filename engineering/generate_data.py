"""
generate_data.py
----------------
1. Creates a realistic SQLite database with 3 tables and large row counts
2. Runs a diverse set of SQL queries (simple → very complex)
3. Times each query (average of 3 runs)
4. Extracts features using feature_extraction.py
5. Saves everything to data/query_features.csv
 
Run:
    python engineering/generate_data.py
"""
 
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
 
import sqlite3
import time
import random
import pandas as pd
from src.feature_extraction import extract_all_features
 
DB_PATH  = "data/benchmark.db"
CSV_PATH = "data/query_features.csv"
N_CUSTOMERS = 10_000
N_ORDERS    = 50_000
N_PRODUCTS  = 500
 
 
# ---------------------------------------------------------------------------
# 1. BUILD DATABASE
# ---------------------------------------------------------------------------
 
def build_database(conn: sqlite3.Connection):
    print("Building database tables...")
 
    conn.executescript("""
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
 
        CREATE TABLE customers (
            id          INTEGER PRIMARY KEY,
            name        TEXT,
            country     TEXT,
            signup_date TEXT,
            tier        TEXT,
            age         INTEGER
        );
 
        CREATE TABLE products (
            id       INTEGER PRIMARY KEY,
            name     TEXT,
            category TEXT,
            price    REAL
        );
 
        CREATE TABLE orders (
            id          INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id  INTEGER,
            amount      REAL,
            status      TEXT,
            order_date  TEXT
        );
    """)
 
    # --- customers ---
    countries = ["India", "US", "UK", "Germany", "France", "Brazil", "Japan"]
    tiers     = ["free", "basic", "premium", "enterprise"]
    years     = ["2020", "2021", "2022", "2023", "2024"]
 
    customers = [
        (i,
         f"Customer_{i}",
         random.choice(countries),
         f"{random.choice(years)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
         random.choice(tiers),
         random.randint(18, 70))
        for i in range(1, N_CUSTOMERS + 1)
    ]
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)
    print(f"  ✓ Inserted {N_CUSTOMERS:,} customers")
 
    # --- products ---
    categories = ["Electronics", "Clothing", "Food", "Sports", "Books", "Tools"]
    products = [
        (i, f"Product_{i}", random.choice(categories), round(random.uniform(5, 5000), 2))
        for i in range(1, N_PRODUCTS + 1)
    ]
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
    print(f"  ✓ Inserted {N_PRODUCTS:,} products")
 
    # --- orders ---
    statuses = ["pending", "completed", "cancelled", "refunded"]
    orders = [
        (i,
         random.randint(1, N_CUSTOMERS),
         random.randint(1, N_PRODUCTS),
         round(random.uniform(10, 10000), 2),
         random.choice(statuses),
         f"{random.choice(years)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}")
        for i in range(1, N_ORDERS + 1)
    ]
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", orders)
    print(f"  ✓ Inserted {N_ORDERS:,} orders")
 
    conn.commit()
    print("Database ready!\n")
 
 
# ---------------------------------------------------------------------------
# 2. QUERY LIBRARY  (simple → medium → complex → very complex)
# ---------------------------------------------------------------------------
 
QUERIES = [
    # ── SIMPLE (should be fast) ──────────────────────────────────────────
    "SELECT * FROM customers LIMIT 10",
    "SELECT COUNT(*) FROM orders",
    "SELECT COUNT(*) FROM customers",
    "SELECT name FROM customers WHERE id = 1",
    "SELECT * FROM products WHERE price < 100",
    "SELECT DISTINCT country FROM customers",
    "SELECT * FROM orders WHERE status = 'completed' LIMIT 50",
 
    # ── MEDIUM ───────────────────────────────────────────────────────────
    "SELECT country, COUNT(*) as cnt FROM customers GROUP BY country",
    "SELECT tier, AVG(age) FROM customers GROUP BY tier",
    "SELECT status, COUNT(*), SUM(amount) FROM orders GROUP BY status",
    "SELECT * FROM orders WHERE amount > 5000 ORDER BY amount DESC",
    "SELECT category, COUNT(*), AVG(price) FROM products GROUP BY category",
    "SELECT * FROM customers WHERE country = 'India' ORDER BY signup_date DESC LIMIT 100",
    "SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id ORDER BY order_count DESC LIMIT 20",
 
    # ── COMPLEX (1 join) ─────────────────────────────────────────────────
    """
    SELECT c.name, c.country, SUM(o.amount) as total_spent
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    GROUP BY c.id, c.name, c.country
    ORDER BY total_spent DESC
    LIMIT 50
    """,
 
    """
    SELECT c.tier, COUNT(o.id) as num_orders, AVG(o.amount) as avg_order
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.tier
    """,
 
    """
    SELECT p.category, SUM(o.amount) as revenue
    FROM products p
    JOIN orders o ON p.id = o.product_id
    GROUP BY p.category
    ORDER BY revenue DESC
    """,
 
    # ── COMPLEX (2 joins) ────────────────────────────────────────────────
    """
    SELECT c.name, p.category, SUM(o.amount) as total
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    JOIN products p ON p.id = o.product_id
    GROUP BY c.id, p.category
    ORDER BY total DESC
    LIMIT 100
    """,
 
    """
    SELECT c.country, p.category, COUNT(o.id) as purchases
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    JOIN products p ON p.id = o.product_id
    WHERE o.status = 'completed'
    GROUP BY c.country, p.category
    HAVING COUNT(o.id) > 10
    ORDER BY purchases DESC
    """,
 
    # ── VERY COMPLEX (subqueries) ─────────────────────────────────────────
    """
    SELECT c.name, c.country,
           (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) as order_count,
           (SELECT SUM(amount) FROM orders o WHERE o.customer_id = c.id) as lifetime_value
    FROM customers c
    WHERE c.tier = 'premium'
    ORDER BY lifetime_value DESC
    LIMIT 50
    """,
 
    """
    SELECT *
    FROM customers
    WHERE id IN (
        SELECT customer_id FROM orders
        WHERE amount > (SELECT AVG(amount) FROM orders)
        GROUP BY customer_id
        HAVING COUNT(*) > 5
    )
    LIMIT 100
    """,
 
    """
    SELECT c.country, AVG(lifetime) as avg_lifetime_value
    FROM (
        SELECT customer_id, SUM(amount) as lifetime
        FROM orders
        WHERE status = 'completed'
        GROUP BY customer_id
    ) lv
    JOIN customers c ON c.id = lv.customer_id
    GROUP BY c.country
    ORDER BY avg_lifetime_value DESC
    """,
 
    # ── DISTINCT + LIKE (index-unfriendly) ────────────────────────────────
    "SELECT DISTINCT country FROM customers WHERE name LIKE 'Customer_1%'",
    "SELECT * FROM products WHERE name LIKE '%_5' ORDER BY price DESC",
 
    # ── UNION ─────────────────────────────────────────────────────────────
    """
    SELECT id, name, 'customer' as type FROM customers WHERE tier = 'enterprise'
    UNION
    SELECT id, name, 'product' as type FROM products WHERE price > 4000
    """,
]
 
 
# ---------------------------------------------------------------------------
# 3. BENCHMARK & COLLECT DATA
# ---------------------------------------------------------------------------
 
def benchmark_queries(conn: sqlite3.Connection) -> pd.DataFrame:
    print(f"Benchmarking {len(QUERIES)} queries...\n")
    records = []
 
    for idx, query in enumerate(QUERIES, 1):
        # Run 3 times, take the average (removes OS scheduling noise)
        times = []
        try:
            for _ in range(3):
                start = time.perf_counter()
                conn.execute(query).fetchall()
                elapsed = (time.perf_counter() - start) * 1000  # → ms
                times.append(elapsed)
            avg_ms = sum(times) / len(times)
        except Exception as e:
            print(f"  ✗ Query {idx} failed: {e}")
            continue
 
        # Extract features
        feats = extract_all_features(query, conn)
        feats["execution_time_ms"] = round(avg_ms, 4)
        feats["query_id"] = idx
 
        records.append(feats)
        bar = "█" * int(avg_ms / 5)
        print(f"  [{idx:02d}/{len(QUERIES)}] {avg_ms:8.2f} ms  {bar}")
 
    df = pd.DataFrame(records)
    print(f"\n✓ Collected {len(df)} records")
    return df
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
 
    conn = sqlite3.connect(DB_PATH)
    build_database(conn)
    df = benchmark_queries(conn)
    conn.close()
 
    df.to_csv(CSV_PATH, index=False)
    print(f"\n✓ Saved to {CSV_PATH}")
    print(df[["query_id", "num_joins", "num_subqueries", "execution_time_ms"]].to_string())