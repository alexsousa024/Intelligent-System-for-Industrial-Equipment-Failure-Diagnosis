'''
Bayesian Network Data Preparation Functions, including:

    - Discretization thresholds calculation
    - Dataframe discretization
    - Spindle overheat condition-based assignment

'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib


def discretization_thresholds(df, num_cols, percentiles=(0.25, 0.75)):
    rows = []
    for col in num_cols:
        n_unique = df[col].nunique()
        if n_unique < 3:
            print(f"Ignore columns '{col}' (only has {n_unique} unique values).")
            continue

        q1 = df[col].quantile(percentiles[0])   # limite between low and normal
        q2 = df[col].quantile(percentiles[1])   # limite between normal and high

        rows.append({
            "feature": col,
            "low  (<= q1)": q1,
            "high  (> q2)": q2
        })
    
    return pd.DataFrame(rows)

def plot_discretization_thresholds(df, percentiles=(0.25, 0.75)):
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        
        data = df[col].dropna()
        q1 = data.quantile(percentiles[0])
        q3 = data.quantile(percentiles[1])

        axes[i].hist(data, bins=40, density=True, alpha=0.75, color="#93CADA")
        
        axes[i].axvline(q1, linestyle='--', linewidth=2, label='25th percentile', color="#8A2828")
        axes[i].axvline(q3, linestyle='--', linewidth=2, label='75th percentile', color="#DB2E2E")

        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    # Remove unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Discretization Thresholds (25% / 75%)",
        fontsize=16,
        fontweight="bold"
    )
    
    plt.tight_layout()
    plt.show()

def fit_discretizer(df, q_low=0.25, q_high=0.75):
    thresholds = {}
    df_copy = df.drop(columns=["spindle_overheat"]).copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        q1 = df_copy[col].quantile(q_low)
        q2 = df_copy[col].quantile(q_high)
        thresholds[col] = (q1, q2)
        
    return thresholds

def apply_discretizer(df, thresholds, labels=("low","normal","high")):
    
    df_new = df.copy()
    for col, (q1,q2) in thresholds.items():
        if q1 == q2:
            df_new[col] = pd.cut(
                df_new[col],
                bins=[-np.inf, q1, np.inf],
                labels=[labels[0], labels[2]]
            )
        else:
            df_new[col] = pd.cut(
                df_new[col],
                bins=[-np.inf, q1, q2, np.inf],
                labels=labels
            )
        
    return df_new

def save_discretizer(thresholds, output_dir):
    joblib.dump(thresholds, output_dir)
    print(f"Discretizer saved to: {output_dir}")

def load_discretizer(filename):
    thresholds = joblib.load(filename)
    print(f"Discretizer loaded from: {filename}")
    return thresholds

def spindle_overheat_from_condition(df, condition, seed=42):
    np.random.seed(seed) 
    mask = condition & (np.random.rand(len(df)) < 0.5)
    df = df.copy()
    df.loc[mask, "spindle_overheat"] = 'Yes'
    df.loc[~mask, "spindle_overheat"] = 'No'
    return df