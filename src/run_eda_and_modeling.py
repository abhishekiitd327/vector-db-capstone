import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Telemetry Data
df = pd.read_csv("../data/processed/database_performance_telemetry.csv")
print("--- Raw Telemetry Summary ---")
print(df.head())
print("\nShape:", df.shape)

# 2. Data Cleaning Log Generation
print("\n--- Running Data Cleaning Integrity Checks ---")
null_count = df.isnull().sum().sum()
invalid_latency = (df['Query_Latency_MS'] <= 0).sum()
print(f"Null Values Found: {null_count}")
print(f"Invalid Latency Records: {invalid_latency}")

# 3. EDA Insights
print("\n--- EDA Insights by Index Type ---")
eda_summary = df.groupby("Index_Type")[["Query_Latency_MS", "Memory_Usage_MB", "Recall_At_K", "QPS"]].mean()
print(eda_summary)

# 4. Feature Engineering & One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=["Index_Type", "Distance_Metric"], drop_first=False)

X = df_encoded.drop(columns=["Query_Latency_MS", "Build_Time_Sec", "QPS"])
y = df_encoded["Query_Latency_MS"]

# 5. Train Baseline Random Forest Latency Regressor (RQ2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 6. Evaluation Metrics Calculation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Baseline Model Performance (RQ2 Latency Prediction) ---")
print(f"RMSE: {rmse:.4f} ms")
print(f"MAE:  {mae:.4f} ms")
print(f"R^2 Score: {r2:.4f}")