import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Telemetry Data
df = pd.read_csv("../data/processed/database_performance_telemetry.csv")
print(f"Loaded {len(df)} Telemetry Records across {df['Shard_Count'].nunique()} Shard Topologies.")

# 2. STATISTICAL ANALYSIS FOR RQ1 (One-Way ANOVA + Tukey HSD on Query Latency)
print("\n=================== RQ1 STATISTICAL TESTS ===================")
hnsw_lat = df[df['Index_Type'] == 'HNSW_Graph']['Query_Latency_MS']
ivf_lat = df[df['Index_Type'] == 'IVF_PQ_Quantized']['Query_Latency_MS']
flat_lat = df[df['Index_Type'] == 'Flat']['Query_Latency_MS']

f_stat, p_val = stats.f_oneway(hnsw_lat, ivf_lat, flat_lat)
print(f"ANOVA F-Statistic (Query Latency): {f_stat:.4f}, p-value: {p_val:.4e}")

tukey = pairwise_tukeyhsd(endog=df['Query_Latency_MS'], groups=df['Index_Type'], alpha=0.05)
print("\n--- Tukey HSD Post-Hoc Test (Latency) ---")
print(tukey)

# 3. STATISTICAL ANALYSIS FOR RQ3 (Distance Metric Recall Comparison)
print("\n=================== RQ3 RECALL FIDELITY BY METRIC ===================")
recall_metric = df.groupby("Distance_Metric")["Recall_At_K"].agg(["mean", "std", "min", "max"])
print(recall_metric)

# 4. STATISTICAL ANALYSIS FOR RQ4 (QPS & Latency vs. Shard Count)
print("\n=================== RQ4 SHARD CONCURRENCY & INFLECTION ===================")
shard_summary = df.groupby("Shard_Count")[["QPS", "Query_Latency_MS", "Memory_Usage_MB"]].mean()
print(shard_summary)

# 5. ML MODELING WITH HYPERPARAMETER TUNING FOR RQ2
print("\n=================== RQ2 HYPERPARAMETER TUNING & ML ===================")
df_encoded = pd.get_dummies(df, columns=["Index_Type", "Distance_Metric"], drop_first=False)
X = df_encoded.drop(columns=["Query_Latency_MS", "Build_Time_Sec", "QPS"])
y = df_encoded["Query_Latency_MS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Tuning
rf_param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, None]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, scoring='r2')
rf_grid.fit(X_train, y_train)

# XGBoost Tuning
xgb_param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
xgb_grid = GridSearchCV(XGBRegressor(random_state=42), xgb_param_grid, cv=5, scoring='r2')
xgb_grid.fit(X_train, y_train)

def print_metrics(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name} -> RMSE: {rmse:.4f} ms, MAE: {mae:.4f} ms, R^2: {r2:.4f}")

print_metrics("Tuned Random Forest", rf_grid.best_estimator_, X_test, y_test)
print_metrics("Tuned XGBoost", xgb_grid.best_estimator_, X_test, y_test)