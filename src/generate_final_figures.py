import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor

# 1. Load Data
df = pd.read_csv("../data/processed/database_performance_telemetry.csv")
os.makedirs("../reports/figures", exist_ok=True)

# Set global styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_dpi = 300

# ----------------------------------------------------
# FIGURE 1: Latency vs. Memory Footprint Trade-off (RQ1)
# ----------------------------------------------------
plt.figure(figsize=(7, 4.5), dpi=fig_dpi)
colors = {'Flat': '#1f77b4', 'HNSW_Graph': '#ff7f0e', 'IVF_PQ_Quantized': '#2ca02c'}
for idx_type, group in df.groupby('Index_Type'):
    plt.scatter(group['Memory_Usage_MB'], group['Query_Latency_MS'], 
                label=idx_type, s=80, alpha=0.8, edgecolors='k', color=colors.get(idx_type))
plt.title("Figure 1: Query Execution Latency vs. Memory Footprint by Index Type", fontsize=11, fontweight='bold')
plt.xlabel("RAM Allocation (MB)", fontsize=10)
plt.ylabel("Query Search Latency (ms)", fontsize=10)
plt.legend(title="Indexing Methodology", frameon=True)
plt.tight_layout()
plt.savefig("../reports/figures/fig1_latency_vs_memory.png")
plt.close()
print("Generated Figure 1: fig1_latency_vs_memory.png")

# ----------------------------------------------------
# FIGURE 2: QPS & Latency vs. Shard Count Inflection (RQ4)
# ----------------------------------------------------
shard_agg = df.groupby('Shard_Count')[['QPS', 'Query_Latency_MS']].mean().reset_index()

fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=fig_dpi)
ax2 = ax1.twinx()

ax1.plot(shard_agg['Shard_Count'], shard_agg['QPS'], color='#1f77b4', marker='o', linewidth=2, label='Throughput (QPS)')
ax2.plot(shard_agg['Shard_Count'], shard_agg['Query_Latency_MS'], color='#d62728', marker='s', linestyle='--', linewidth=2, label='Latency (ms)')

ax1.set_xlabel("Database Cluster Shard Count", fontsize=10)
ax1.set_ylabel("Throughput (Queries Per Second)", color='#1f77b4', fontsize=10)
ax2.set_ylabel("Mean Query Latency (ms)", color='#d62728', fontsize=10)
plt.title("Figure 2: QPS Throughput Scaling and Tail Latency Inflection across Shards", fontsize=11, fontweight='bold')
ax1.set_xticks([1, 2, 4, 8, 16])
fig.tight_layout()
plt.savefig("../reports/figures/fig2_shard_qps_latency.png")
plt.close()
print("Generated Figure 2: fig2_shard_qps_latency.png")

# ----------------------------------------------------
# FIGURE 3: XGBoost Latency Regressor Residual Plot (RQ2)
# ----------------------------------------------------
df_encoded = pd.get_dummies(df, columns=["Index_Type", "Distance_Metric"], drop_first=False)
X = df_encoded.drop(columns=["Query_Latency_MS", "Build_Time_Sec", "QPS"])
y = df_encoded["Query_Latency_MS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(7, 4.5), dpi=fig_dpi)
plt.scatter(y_pred, residuals, color='#9467bd', s=70, edgecolors='k', alpha=0.85)
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.title("Figure 3: Residual Distribution of Tuned XGBoost Latency Regressor", fontsize=11, fontweight='bold')
plt.xlabel("Predicted Query Latency (ms)", fontsize=10)
plt.ylabel("Residuals (Actual - Predicted, ms)", fontsize=10)
plt.tight_layout()
plt.savefig("../reports/figures/fig3_xgb_residuals.png")
plt.close()
print("Generated Figure 3: fig3_xgb_residuals.png")

# ----------------------------------------------------
# FIGURE 4: Feature Importance for Latency Prediction (RQ2)
# ----------------------------------------------------
importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(7.5, 4.5), dpi=fig_dpi)
importances.plot(kind='barh', color='#2ca02c', edgecolor='black')
plt.title("Figure 4: Relative Feature Importances for Query Latency Prediction", fontsize=11, fontweight='bold')
plt.xlabel("Relative Feature Importance Weight", fontsize=10)
plt.tight_layout()
plt.savefig("../reports/figures/fig4_feature_importance.png")
plt.close()
print("Generated Figure 4: fig4_feature_importance.png")

print("All 4 figures successfully generated and saved to reports/figures/!")