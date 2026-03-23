"""
feature_extraction.py
---------------------
Converts a raw SQL string into a flat feature dictionary
that the ML model can consume.
 
Two levels of features:
  1. AST features   - structural properties of the query itself
  2. Schema features - row-count statistics about the tables referenced
"""
 
import re
import sqlite3
from typing import Optional

 
try:
    import sqlglot
    import sqlglot.expressions as exp
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False
    print("Warning: sqlglot not installed. Run: pip install sqlglot")
 
 
# ---------------------------------------------------------------------------
# 1. AST FEATURES
# ---------------------------------------------------------------------------
 
def extract_ast_features(query: str) -> dict:
    """
    Parse the SQL query into an AST and count structural elements.
    Returns a flat dict of integer/float features.
    """
    if not SQLGLOT_AVAILABLE:
        return _fallback_regex_features(query)
 
    try:
        ast = sqlglot.parse_one(query, error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception:
        return _fallback_regex_features(query)
 
    if ast is None:
        return _fallback_regex_features(query)
 
    feats = {}
 
    # --- JOIN complexity ---
    feats["num_joins"] = len(list(ast.find_all(exp.Join)))
 
    # --- Subquery complexity ---
    feats["num_subqueries"] = len(list(ast.find_all(exp.Subquery)))
 
    # --- WHERE conditions ---
    feats["num_where_clauses"] = len(list(ast.find_all(exp.Where)))
    feats["num_eq_conditions"]  = len(list(ast.find_all(exp.EQ)))
    feats["num_gt_conditions"]  = len(list(ast.find_all(exp.GT)))
    feats["num_lt_conditions"]  = len(list(ast.find_all(exp.LT)))
 
    # --- Aggregations ---
    feats["num_aggregations"] = len(list(ast.find_all(exp.AggFunc)))
 
    # --- Grouping / Sorting / Distinct ---
    feats["has_group_by"]  = int(ast.find(exp.Group)    is not None)
    feats["has_order_by"]  = int(ast.find(exp.Order)    is not None)
    feats["has_distinct"]  = int(ast.find(exp.Distinct) is not None)
    feats["has_limit"]     = int(ast.find(exp.Limit)    is not None)
    feats["has_union"]     = int(ast.find(exp.Union)    is not None)
    feats["has_having"]    = int(ast.find(exp.Having)   is not None)
 
    # --- Tables and columns ---
    feats["num_tables"]           = len(list(ast.find_all(exp.Table)))
    feats["num_columns_selected"] = len(list(ast.find_all(exp.Column)))
 
    # --- LIKE / pattern matching (index-unfriendly) ---
    feats["num_like_clauses"] = len(list(ast.find_all(exp.Like)))
 
    # --- Raw text proxies ---
    feats["query_char_length"] = len(query)
    feats["query_word_count"]  = len(query.split())
 
    return feats
 
 
def _fallback_regex_features(query: str) -> dict:
    """Regex-based fallback when sqlglot is unavailable."""
    q = query.upper()
    return {
        "num_joins":            len(re.findall(r'\bJOIN\b', q)),
        "num_subqueries":       len(re.findall(r'\bSELECT\b', q)) - 1,
        "num_where_clauses":    len(re.findall(r'\bWHERE\b', q)),
        "num_eq_conditions":    q.count('='),
        "num_gt_conditions":    q.count('>'),
        "num_lt_conditions":    q.count('<'),
        "num_aggregations":     len(re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN)\b', q)),
        "has_group_by":         int('GROUP BY' in q),
        "has_order_by":         int('ORDER BY' in q),
        "has_distinct":         int('DISTINCT' in q),
        "has_limit":            int('LIMIT' in q),
        "has_union":            int('UNION' in q),
        "has_having":           int('HAVING' in q),
        "num_tables":           len(re.findall(r'\bFROM\b|\bJOIN\b', q)),
        "num_columns_selected": 0,
        "num_like_clauses":     len(re.findall(r'\bLIKE\b', q)),
        "query_char_length":    len(query),
        "query_word_count":     len(query.split()),
    }
 
 
# ---------------------------------------------------------------------------
# 2. SCHEMA FEATURES
# ---------------------------------------------------------------------------
 
def extract_schema_features(query: str, conn: sqlite3.Connection) -> dict:
    """
    Query the SQLite DB for row-count statistics of tables referenced in
    the SQL query. These are crucial — the same query is slow on 10M rows
    and instant on 100 rows.
    """
    table_names = _extract_table_names(query)
    row_counts = []
 
    for table in table_names:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            row_counts.append(count)
        except Exception:
            pass
 
    if not row_counts:
        return {
            "total_rows_all_tables": 0,
            "max_single_table_rows": 0,
            "avg_table_rows":        0,
            "num_tables_found_in_db": 0,
        }
 
    return {
        "total_rows_all_tables":  sum(row_counts),
        "max_single_table_rows":  max(row_counts),
        "avg_table_rows":         sum(row_counts) / len(row_counts),
        "num_tables_found_in_db": len(row_counts),
    }
 
 
def _extract_table_names(query: str) -> list:
    """Extract table names using sqlglot or regex."""
    if SQLGLOT_AVAILABLE:
        try:
            ast = sqlglot.parse_one(query, error_level=sqlglot.ErrorLevel.IGNORE)
            return [t.name for t in ast.find_all(exp.Table)] if ast else []
        except Exception:
            pass
 
    # Regex fallback
    pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    return re.findall(pattern, query, re.IGNORECASE)
 
 
# ---------------------------------------------------------------------------
# 3. EXPLAIN PLAN FEATURES
# ---------------------------------------------------------------------------
 
def extract_explain_features(query: str, conn: sqlite3.Connection) -> dict:
    """
    Run EXPLAIN QUERY PLAN and inspect whether the DB will use an index
    or do a full table scan. Full scans are ~10-100x slower.
    """
    try:
        rows = conn.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
        plan_text = " ".join(str(r) for r in rows).upper()
        return {
            "plan_uses_index":     int("USING INDEX" in plan_text or "SEARCH" in plan_text),
            "plan_does_full_scan": int("SCAN TABLE" in plan_text or "SCAN" in plan_text),
            "plan_uses_temp":      int("TEMP" in plan_text),
            "plan_uses_sort":      int("ORDER BY" in plan_text or "SORT" in plan_text),
        }
    except Exception:
        return {
            "plan_uses_index":     0,
            "plan_does_full_scan": 1,
            "plan_uses_temp":      0,
            "plan_uses_sort":      0,
        }
 
 
# ---------------------------------------------------------------------------
# 4. COMBINED FEATURE EXTRACTOR
# ---------------------------------------------------------------------------
 
def extract_all_features(query: str, conn: Optional[sqlite3.Connection] = None) -> dict:
    """
    Master function — call this to get all features for a query.
    If conn is None, schema and explain features are set to defaults.
    """
    feats = extract_ast_features(query)
 
    if conn is not None:
        feats.update(extract_schema_features(query, conn))
        feats.update(extract_explain_features(query, conn))
    else:
        # Defaults when no DB connection available (e.g. demo/UI mode)
        feats.update({
            "total_rows_all_tables":  100_000,
            "max_single_table_rows":  100_000,
            "avg_table_rows":         100_000,
            "num_tables_found_in_db": feats.get("num_tables", 1),
            "plan_uses_index":        0,
            "plan_does_full_scan":    1,
            "plan_uses_temp":         0,
            "plan_uses_sort":         int(feats.get("has_order_by", 0)),
        })
 
    return feats