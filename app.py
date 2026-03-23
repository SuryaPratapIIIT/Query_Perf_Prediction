"""
app.py  —  Streamlit UI for SQL Query Performance Predictor
------------------------------------------------------------
Run:
    streamlit run app.py
"""
 
import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
 
sys.path.insert(0, os.path.dirname(__file__))
from src.feature_extraction import extract_all_features
 
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SQL Performance Predictor",
    page_icon="⚡",
    layout="wide",
)
 
# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;600;700&display=swap');
 
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    code, textarea, .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-val  { font-size: 2.4rem; font-weight: 700; color: #c9d1d9; }
    .metric-lbl  { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
    .badge-fast  { background:#0d4429; color:#3fb950; padding:4px 12px; border-radius:20px; font-size:.85rem; }
    .badge-ok    { background:#3d2c00; color:#e3b341; padding:4px 12px; border-radius:20px; font-size:.85rem; }
    .badge-slow  { background:#3d0000; color:#f85149; padding:4px 12px; border-radius:20px; font-size:.85rem; }
    .feat-pill {
        display:inline-block; background:#21262d; border:1px solid #30363d;
        border-radius:8px; padding:4px 10px; margin:4px;
        font-family:'JetBrains Mono',monospace; font-size:.78rem; color:#8b949e;
    }
    .feat-pill span { color:#58a6ff; font-weight:600; }
</style>
""", unsafe_allow_html=True)
 
# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        return None, None
    model    = joblib.load("model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features
 
model, FEATURE_NAMES = load_model()
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ SQL Perf Predictor")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("""
1. Paste any SQL query
2. The query is parsed into an AST
3. ~20 structural features are extracted
4. XGBoost predicts execution time
5. SHAP explains *why*
    """)
    st.markdown("---")
    st.markdown("**Tech Stack**")
    for item in ["sqlglot (AST parsing)", "XGBoost (model)", "SHAP (explainability)", "Streamlit (UI)"]:
        st.markdown(f"- `{item}`")
 
    st.markdown("---")
    table_rows = st.number_input(
        "Assume table size (rows)",
        min_value=1_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        help="Used for schema features when no live DB is connected"
    )
 
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ SQL Query Performance Predictor")
st.markdown("*Paste a SQL query and get an instant execution-time estimate powered by XGBoost + AST feature engineering*")
st.markdown("---")
 
# ── Example queries ───────────────────────────────────────────────────────────
EXAMPLES = {
    "🟢 Simple SELECT": "SELECT * FROM customers WHERE country = 'India' LIMIT 100",
    "🟡 GROUP BY + JOIN": """SELECT c.country, AVG(o.amount) as avg_order
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.country
ORDER BY avg_order DESC""",
    "🔴 Correlated Subquery": """SELECT c.name,
    (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) as orders,
    (SELECT SUM(amount) FROM orders o WHERE o.customer_id = c.id) as lifetime_value
FROM customers c
WHERE c.tier = 'premium'
ORDER BY lifetime_value DESC
LIMIT 50""",
    "🔴 Nested Subquery": """SELECT country, AVG(lifetime) as avg_ltv
FROM (
    SELECT customer_id, SUM(amount) as lifetime
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
) lv
JOIN customers c ON c.id = lv.customer_id
GROUP BY country""",
}
 
col_ex, _ = st.columns([3, 1])
with col_ex:
    chosen = st.selectbox("Load an example query:", ["— type your own —"] + list(EXAMPLES.keys()))
 
default_query = EXAMPLES.get(chosen, "")
query = st.text_area(
    "SQL Query",
    value=default_query,
    height=180,
    placeholder="SELECT c.name, SUM(o.amount)\nFROM customers c\nJOIN orders o ON c.id = o.customer_id\nGROUP BY c.name\nORDER BY 2 DESC",
)
 
predict_btn = st.button("⚡ Predict Execution Time", type="primary", use_container_width=False)
 
# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn and query.strip():
    if model is None:
        st.error("❌ No trained model found. Run `python engineering/generate_data.py` then `python benchmarking/train.py` first.")
    else:
        with st.spinner("Parsing AST and predicting..."):
            from src.feature_extraction import extract_ast_features, extract_explain_features
 
            ast_feats = extract_ast_features(query)
            schema_feats = {
                "total_rows_all_tables":  table_rows * max(ast_feats.get("num_tables", 1), 1),
                "max_single_table_rows":  table_rows,
                "avg_table_rows":         table_rows,
                "num_tables_found_in_db": ast_feats.get("num_tables", 1),
            }
            explain_feats = {
                "plan_uses_index":     0,
                "plan_does_full_scan": 1,
                "plan_uses_temp":      ast_feats.get("has_group_by", 0),
                "plan_uses_sort":      ast_feats.get("has_order_by", 0),
            }
 
            all_feats = {**ast_feats, **schema_feats, **explain_feats}
            feat_df = pd.DataFrame([all_feats]).reindex(columns=FEATURE_NAMES, fill_value=0)
 
            log_pred = model.predict(feat_df)[0]
            pred_ms  = float(np.expm1(log_pred))
 
        # ── Results ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Results")
 
        c1, c2, c3, c4 = st.columns(4)
 
        with c1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-val">{pred_ms:.1f}</div>
              <div class="metric-lbl">Predicted Time (ms)</div>
            </div>""", unsafe_allow_html=True)
 
        with c2:
            if pred_ms < 100:
                badge = '<span class="badge-fast">✅ FAST</span>'
            elif pred_ms < 1000:
                badge = '<span class="badge-ok">⚠️ MODERATE</span>'
            else:
                badge = '<span class="badge-slow">🐌 SLOW</span>'
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.6rem;margin-bottom:8px">{badge}</div>
              <div class="metric-lbl">Performance Rating</div>
            </div>""", unsafe_allow_html=True)
 
        with c3:
            complexity = (
                ast_feats.get("num_joins", 0) * 3 +
                ast_feats.get("num_subqueries", 0) * 4 +
                ast_feats.get("has_group_by", 0) +
                ast_feats.get("has_order_by", 0) +
                ast_feats.get("has_union", 0) * 2
            )
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-val">{complexity}</div>
              <div class="metric-lbl">Complexity Score</div>
            </div>""", unsafe_allow_html=True)
 
        with c4:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-val">{ast_feats.get('num_joins',0)}</div>
              <div class="metric-lbl">JOIN Count</div>
            </div>""", unsafe_allow_html=True)
 
        # ── Feature breakdown ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Extracted AST Features")
 
        pills_html = ""
        for k, v in ast_feats.items():
            if isinstance(v, (int, float)) and v != 0:
                pills_html += f'<div class="feat-pill">{k}: <span>{v}</span></div>'
        st.markdown(pills_html, unsafe_allow_html=True)
 
        # ── Optimisation tips ─────────────────────────────────────────────
        tips = []
        if ast_feats.get("num_joins", 0) >= 2:
            tips.append("🔗 Multiple JOINs detected — ensure all join columns are **indexed**.")
        if ast_feats.get("num_subqueries", 0) > 0:
            tips.append("📦 Correlated subqueries can be expensive — consider rewriting with **JOINs or CTEs**.")
        if ast_feats.get("num_like_clauses", 0) > 0:
            tips.append("🔎 `LIKE '%...'` prevents index usage — use **full-text search** if possible.")
        if ast_feats.get("has_distinct", 0):
            tips.append("⚠️ `DISTINCT` requires deduplication — check if it's really needed.")
        if ast_feats.get("has_union", 0):
            tips.append("🔀 `UNION` merges full result sets — use `UNION ALL` if duplicates aren't a concern.")
        if not ast_feats.get("has_limit", 0) and pred_ms > 500:
            tips.append("📄 No `LIMIT` clause — consider paginating large result sets.")
 
        if tips:
            st.markdown("---")
            st.markdown("### 💡 Optimisation Tips")
            for tip in tips:
                st.markdown(f"- {tip}")
 
elif predict_btn:
    st.warning("Please enter a SQL query first.")
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#555'>Built with XGBoost · sqlglot · Streamlit · SHAP</small>",
    unsafe_allow_html=True
)
 