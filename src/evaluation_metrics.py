
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_recall_at_k(retrieved_indices, ground_truth_indices, k=10):
    """
    Calculates Recall@K by comparing retrieved neighbor indices against brute-force ground truth.
    Formula: Recall@K = |Retrieved ∩ True| / |True|
    """
    recalls = []
    for ret, gt in zip(retrieved_indices, ground_truth_indices):
        ret_set = set(ret[:k])
        gt_set = set(gt[:k])
        intersection = ret_set.intersection(gt_set)
        recalls.append(len(intersection) / float(len(gt_set)))
    return np.mean(recalls)

def calculate_qps(total_queries, total_time_seconds):
    """
    Calculates Queries-Per-Second (QPS) throughput capacity.
    """
    if total_time_seconds <= 0:
        return 0.0
    return total_queries / total_time_seconds

def evaluate_regression_models(y_true, y_pred):
    """
    Computes RMSE, MAE, and R^2 evaluation metrics for latency prediction.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def clean_telemetry_data(df):
    """
    Cleans raw benchmark output and logs issues for the Interim Data Cleaning Log.
    """
    initial_count = len(df)
    cleaning_log = []
    
    # Check 1: Remove invalid latency or memory values
    invalid_mask = (df['Query_Latency_MS'] <= 0) | (df['Memory_Usage_MB'] <= 0)
    df_clean = df[~invalid_mask].copy()
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        cleaning_log.append({
            "Issue": "Non-Positive Latency / Memory",
            "Variables Affected": "Query_Latency_MS, Memory_Usage_MB",
            "Detection Method": "Range filtering (<= 0)",
            "Treatment Applied": f"Removed {invalid_count} records",
            "Rationale": "Thread-lock or timer artifact corruption"
        })
        
    # Check 2: Check for missing values
    null_count = df_clean.isnull().sum().sum()
    if null_count > 0:
        df_clean = df_clean.dropna()
        cleaning_log.append({
            "Issue": "Missing Values",
            "Variables Affected": "Multiple Telemetry Attributes",
            "Detection Method": "Null scan",
            "Treatment Applied": f"Dropped null rows",
            "Rationale": "Incomplete execution logs"
        })
        
    return df_clean, pd.DataFrame(cleaning_log)