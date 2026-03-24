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
 
    plot_eda(df, y_raw, y)
    return X, y, feature_cols

def plot_eda(df, y_raw, y):
    print("Generating EDA plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        pass
    import os
    os.makedirs("outputs", exist_ok=True)

    # 1. Target Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(y_raw, bins=30, color="crimson", alpha=0.7, edgecolor='k')
    plt.title("Distribution of Raw Execution Time (ms)")
    plt.xlabel("Execution Time (ms)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("outputs/eda_target_distribution_raw.png", dpi=150)
    plt.close()

    # 2. Target Distribution (Log Transformed)
    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=30, color="indigo", alpha=0.7, edgecolor='k')
    plt.title("Distribution of Log-Transformed Execution Time")
    plt.xlabel("log1p(Execution Time)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("outputs/eda_target_distribution_log.png", dpi=150)
    plt.close()

    # 3. Correlation Heatmap
    try:
        plt.figure(figsize=(10, 8))
        corrmat = df.corr()
        # Top 12 absolute correlated features
        top_corr_features = corrmat.corrwith(df["execution_time_ms"]).abs().nlargest(12).index
        import numpy as np
        cm = np.corrcoef(df[top_corr_features].values.T)
        sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 9}, 
                    yticklabels=top_corr_features.values, xticklabels=top_corr_features.values, cmap="coolwarm")
        plt.title("Top Correlated Features with Execution Time")
        plt.tight_layout()
        plt.savefig("outputs/eda_correlation_heatmap.png", dpi=150)
        plt.close()
    except Exception:
        pass

    print("✓ EDA plots saved to outputs/")
 
 
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
        eval_set=[(X_train, y_train), (X_test, y_test)],
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
 
    return mae, rmse, r2, actual.values, preds
 
 
# ---------------------------------------------------------------------------
# 4. PLOTTING
# ---------------------------------------------------------------------------
 
def plot_predictions(actual, preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    os.makedirs("outputs", exist_ok=True)
 
    # Actual vs Predicted
    plt.figure(figsize=(7, 7))
    plt.scatter(actual, preds, alpha=0.6, edgecolor="k", linewidth=0.4)
    lims = [0, max(actual.max(), preds.max()) * 1.05]
    plt.plot(lims, lims, "--", color="tab:gray", label="perfect")
    plt.xlabel("Actual execution_time_ms")
    plt.ylabel("Predicted execution_time_ms")
    plt.title("Actual vs Predicted execution time")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
 
    # Residual distribution
    residuals = actual - preds
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, color="tab:blue", alpha=0.65)
    plt.axvline(0, color="k", linestyle="--")
    plt.xlabel("Residual (actual - predicted) ms")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.savefig("outputs/residual_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
 
    print("\n✓ Prediction graphs saved to outputs/")

def plot_learning_curves(model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, results['validation_0']['rmse'], label='Train RMSE')
    plt.plot(x_axis, results['validation_1']['rmse'], label='Test RMSE')
    plt.legend()
    plt.ylabel('RMSE (log space)')
    plt.xlabel('Boosting Round')
    plt.title('XGBoost Learning Curves')
    plt.tight_layout()
    plt.savefig("outputs/learning_curves.png", dpi=150)
    plt.close()
    print("✓ Learning curves plot saved to outputs/")

def plot_xgb_importance(model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from xgboost import plot_importance
    
    plt.figure(figsize=(10, 8))
    plot_importance(model, max_num_features=15, importance_type='weight', title='XGBoost Feature Importance (Weight)')
    plt.tight_layout()
    plt.savefig("outputs/xgb_feature_importance.png", dpi=150)
    plt.close()
    print("✓ XGBoost feature importance plot saved to outputs/")
 
 
# ---------------------------------------------------------------------------
# 5. SHAP EXPLAINABILITY
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
    mae, rmse, r2, actual, preds = evaluate(model, X_test, y_test)
    
    # ── Post-Training Plots ──
    plot_predictions(actual, preds)
    plot_learning_curves(model)
    plot_xgb_importance(model)
    
    shap_analysis(model, X_test, feature_cols)
 
    # Save model artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATS_PATH)
    print(f"\n✓ Model saved → {MODEL_PATH}")
    print(f"✓ Features saved → {FEATS_PATH}")