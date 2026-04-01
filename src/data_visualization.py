import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE


# 2) Data Visualization

### 2.1) Data Preparation

def data_preparation(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Any]:
    X = np.vstack(df["embeddings"].values)

    le = LabelEncoder()
    y = le.fit_transform(df["syndrome_id"])

    return X, y, le

### 2.2) t-SNE Visualization

def tsne_visualization(X: np.ndarray, y: np.ndarray, save_path: str = None, show: bool = False) -> None:
    tsne = TSNE(n_components=2, random_state=42)
    X_2D = tsne.fit_transform(X)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()


### 2.3) ROC AUC Curves Plots

def roc_curves_visualization(
    mean_fpr: np.ndarray,
    tprs_euclidean: list,
    tprs_cosine: list,
    aucs_euclidean: list,
    aucs_cosine: list,
    save_path: str = None,
    show: bool = False
) -> None:

    mean_tpr_euclidean = np.mean(tprs_euclidean, axis=0)
    mean_tpr_cosine = np.mean(tprs_cosine, axis=0)

    mean_auc_euclidean = np.mean(aucs_euclidean)
    mean_auc_cosine = np.mean(aucs_cosine)

    plt.figure(figsize=(8, 6))

    plt.plot(mean_fpr, mean_tpr_euclidean, label=f"Euclidean (AUC = {mean_auc_euclidean:.3f})")
    plt.plot(mean_fpr, mean_tpr_cosine, label=f"Cosine (AUC = {mean_auc_cosine:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

