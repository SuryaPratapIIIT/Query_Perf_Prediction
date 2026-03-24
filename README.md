SQL Query Performance Predictor

Predict SQL query execution time using AST feature engineering + XGBoost.

A portfolio-grade ML project demonstrating:

SQL parsing via Abstract Syntax Trees (sqlglot)
Feature engineering from query structure + schema stats
Regression modelling with XGBoost + log-transform target
Explainability via SHAP values
End-to-end deployment via Streamlit


🗂️ Project Structure
query-perf-predictor/
├── src/
│   └── feature_extraction.py   ← AST parsing + feature engineering
├── engineering/
│   └── generate_data.py        ← DB creation + query benchmarking
├── benchmarking/
│   └── train.py                ← Model training + SHAP analysis
├── data/
│   ├── benchmark.db            ← SQLite database (auto-generated)
│   └── query_features.csv      ← Engineered features + labels
├── outputs/
│   ├── eda_target_distribution_raw.png  ← Starting analysis
│   ├── eda_target_distribution_log.png  ← Starting analysis
│   ├── learning_curves.png              ← Ending analysis
│   ├── predicted_vs_actual.png          ← Ending analysis
│   ├── residual_histogram.png           ← Ending analysis
│   ├── xgb_feature_importance.png       ← Ending analysis
│   ├── shap_importance.png              ← Ending analysis
│   └── shap_beeswarm.png                ← Ending analysis
├── app.py                      ← Streamlit UI
├── model.pkl                   ← Trained XGBoost model
├── feature_names.pkl
└── requirements.txt

🚀 Quick Start
bash# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data (creates DB + benchmarks queries)
python engineering/generate_data.py

# 3. Train the model
python benchmarking/train.py

# 4. Launch the app
streamlit run app.py

🔧 Feature Engineering
Features are extracted at three levels:
AST Features (structural)
FeatureDescriptionnum_joinsNumber of JOIN clausesnum_subqueriesNested SELECT countnum_aggregationsCOUNT, SUM, AVG, etc.has_group_byGROUP BY present?has_order_byORDER BY present?has_distinctDISTINCT used?num_like_clausesLIKE pattern matcheshas_unionUNION used?
Schema Features (data-aware)
FeatureDescriptiontotal_rows_all_tablesSum of rows in all referenced tablesmax_single_table_rowsLargest single table
Explain Plan Features
FeatureDescriptionplan_uses_indexDB will use an indexplan_does_full_scanDB must scan entire tableplan_uses_tempTemp table required

🤖 Model

Algorithm: XGBoost Regressor
Target: log1p(execution_time_ms) — log-transformed to handle skew
Evaluation: MAE, RMSE, R²
Graphical Evaluation:
- Pre-training (EDA): Target distribution and log-transformation analysis.
- Post-training: Learning curves, prediction scatter plots, residual histograms, and XGBoost native feature importance.


📊 SHAP Explainability
SHAP (SHapley Additive exPlanations) reveals which features drive predictions.
Key findings:

num_joins is the strongest predictor of slow queries
plan_does_full_scan is a major negative signal
has_limit reduces predicted time significantly