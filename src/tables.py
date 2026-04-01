import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


# 4) Tables and Results Organization

### 4.1) Table for Optimal k Selection

def create_k_results_table(
    k_values: range,
    results_k: Dict[str, List[float]]
) -> pd.DataFrame:

    df_k = pd.DataFrame({
        "k": list(k_values),
        "euclidean_f1": results_k["euclidean"],
        "cosine_f1": results_k["cosine"]
    })


    df_k["best_f1"] = df_k[["euclidean_f1", "cosine_f1"]].max(axis=1)


    df_k = df_k.sort_values(by="best_f1", ascending=False).reset_index(drop=True)

    df_k = df_k.round(3)

    return df_k


### 4.2) Table for Final Model Metrics

def create_final_metrics_table(
    results: Dict[str, Dict[str, List[float]]]
) -> pd.DataFrame:

    summary = []

    for metric_name in ["f1", "auc", "top3"]:

        euclidean_mean = np.mean(results["euclidean"][metric_name])
        cosine_mean = np.mean(results["cosine"][metric_name])

        summary.append({
            "metric": metric_name,
            "euclidean": euclidean_mean,
            "cosine": cosine_mean
        })

    df_metrics = pd.DataFrame(summary)

    df_metrics = df_metrics.round(3)

    return df_metrics


### 4.3) Save Table as Image

def save_table_as_image(
    df: pd.DataFrame,
    filename: str
) -> None:

    fig, ax = plt.subplots(figsize=(8, len(df) * 0.6 + 1))

    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    plt.savefig(filename, bbox_inches="tight")
    plt.close()