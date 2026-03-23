"""
train.py
--------
Loads query_features.csv, trains an XGBoost regressor to predict
execution_time_ms, evaluates it, generates SHAP plots, and saves
the model + feature list.
 
Run:
    python benchmarking/train.py
"""
 
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
 
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
CSV_PATH   = "data/query_features.csv"
MODEL_PATH = "model.pkl"
FEATS_PATH = "feature_names.pkl"
 
 
# ---------------------------------------------------------------------------
# 1. LOAD & PREPARE DATA
# ---------------------------------------------------------------------------
 
def load_data():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
 
    # Drop non-feature columns
    drop_cols = ["execution_time_ms", "query_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
 
    X = df[feature_cols].fillna(0)
    y_raw = df["execution_time_ms"]
 
    # Log-transform target — execution times are right-skewed
    # log1p = log(1+x) — safe when x can be 0
    y = np.log1p(y_raw)
 
    print(f"\nTarget stats (raw ms):")
    print(f"  min={y_raw.min():.2f}  max={y_raw.max():.2f}  mean={y_raw.mean():.2f}")
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}\n")
 
    return X, y, feature_cols
 
 
# ---------------------------------------------------------------------------
# 2. TRAIN
# ---------------------------------------------------------------------------
 
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
 
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
 
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
 
    return model, X_train, X_test, y_train, y_test
 
 
# ---------------------------------------------------------------------------
# 3. EVALUATE
# ---------------------------------------------------------------------------
 
def evaluate(model, X_test, y_test):
    preds_log = model.predict(X_test)
 
    # Reverse the log transform
    preds  = np.expm1(preds_log)
    actual = np.expm1(y_test)
 
    mae  = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    r2   = r2_score(actual, preds)
 
    print("=" * 40)
    print("  MODEL EVALUATION")
    print("=" * 40)
    print(f"  MAE  : {mae:.2f} ms")
    print(f"  RMSE : {rmse:.2f} ms")
    print(f"  R²   : {r2:.4f}")
    print("=" * 40)
 
    # Show predictions vs actuals
    comparison = pd.DataFrame({
        "actual_ms":    actual.values,
        "predicted_ms": preds,
        "error_ms":     abs(actual.values - preds),
    }).round(2)
    print("\nSample predictions:")
    print(comparison.head(10).to_string(index=False))
 
    return mae, rmse, r2
 
 
# ---------------------------------------------------------------------------
# 4. SHAP EXPLAINABILITY
# ---------------------------------------------------------------------------
 
def shap_analysis(model, X_test, feature_cols):
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
 
        os.makedirs("outputs", exist_ok=True)
 
        explainer   = shap.Explainer(model)
        shap_values = explainer(X_test)
 
        # Summary bar plot — overall feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig("outputs/shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
 
        # Beeswarm plot — direction of effect
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig("outputs/shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
 
        print("\n✓ SHAP plots saved to outputs/")
 
    except ImportError:
        print("\nSHAP not installed — skipping plots. Run: pip install shap")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    X, y, feature_cols = load_data()
    model, X_train, X_test, y_train, y_test = train(X, y)
    evaluate(model, X_test, y_test)
    shap_analysis(model, X_test, feature_cols)
 
    # Save model artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATS_PATH)
    print(f"\n✓ Model saved → {MODEL_PATH}")
    print(f"✓ Features saved → {FEATS_PATH}")