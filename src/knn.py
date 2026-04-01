import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score, top_k_accuracy_score, roc_curve


# 3) Classification Task

### 3.1) Optimal Value of kk

def optimal_k_value(X: np.ndarray, y: np.ndarray) -> Tuple[range, Dict[str, List[float]]]:
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    k_values = range(1,16)

    results = {"euclidean": [], "cosine": []}

    for k in k_values:
        f1_euclidean_score = []
        f1_cosine_score = []

        for train_idx, test_idx in kf.split(X,y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            knn_euclidean = KNeighborsClassifier(n_neighbors=k,metric="euclidean")
            knn_euclidean.fit(X_train, y_train)
            y_pred_euclidean = knn_euclidean.predict(X_test)
            f1_euclidean_score.append(f1_score(y_test, y_pred_euclidean,average="weighted"))

            knn_cosine = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            knn_cosine.fit(X_train, y_train)
            y_pred_cosine = knn_cosine.predict(X_test)
            f1_cosine_score.append(f1_score(y_test, y_pred_cosine, average="weighted"))

        results["euclidean"].append(np.mean(f1_euclidean_score))
        results["cosine"].append(np.mean(f1_cosine_score))

    return k_values, results



## 3.2) Model Evaluation: F1, AUC and Top-k (k = 3)

def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    k_optimal: int
) -> Tuple[
    Dict[str, Dict[str, List[float]]],
    np.ndarray,
    List[np.ndarray],
    List[np.ndarray],
    List[float],
    List[float]
]:
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results = {"euclidean": {"f1": [], "auc": [], "top3": []},
               "cosine": {"f1": [], "auc": [], "top3": []}}

    mean_fpr = np.linspace(0, 1, 100)

    tprs_euclidean, tprs_cosine = [], []
    aucs_euclidean, aucs_cosine = [], []

    for train_idx, test_idx in kf.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        knn_euclidean = KNeighborsClassifier(n_neighbors=k_optimal, metric="euclidean")
        knn_euclidean.fit(X_train, y_train)
        y_pred_euclidean = knn_euclidean.predict(X_test)
        y_prob_euclidean = knn_euclidean.predict_proba(X_test)

        results["euclidean"]["f1"].append(f1_score(y_test,y_pred_euclidean, average="weighted"))
        results["euclidean"]["auc"].append(roc_auc_score(y_test,y_prob_euclidean,multi_class="ovr"))
        results["euclidean"]["top3"].append(top_k_accuracy_score(y_test,y_prob_euclidean,k=3))

        y_bin = label_binarize(y_test, classes=np.unique(y))

        fpr_euclidean, tpr_euclidean, _ = roc_curve(y_bin.ravel(), y_prob_euclidean.ravel())
        tprs_euclidean.append(np.interp(mean_fpr, fpr_euclidean, tpr_euclidean))
        aucs_euclidean.append(results["euclidean"]["auc"][-1])

        knn_cosine = KNeighborsClassifier(n_neighbors=k_optimal, metric="cosine")
        knn_cosine.fit(X_train, y_train)
        y_pred_cosine = knn_cosine.predict(X_test)
        y_prob_cosine = knn_cosine.predict_proba(X_test)

        results["cosine"]["f1"].append(f1_score(y_test, y_pred_cosine, average="weighted"))
        results["cosine"]["auc"].append(roc_auc_score(y_test, y_prob_cosine, multi_class="ovr"))
        results["cosine"]["top3"].append(top_k_accuracy_score(y_test, y_prob_cosine, k=3))

        fpr_cosine, tpr_cosine, _ = roc_curve(y_bin.ravel(), y_prob_cosine.ravel())
        tprs_cosine.append(np.interp(mean_fpr, fpr_cosine, tpr_cosine))
        aucs_cosine.append(results["cosine"]["auc"][-1])

    return results, mean_fpr, tprs_euclidean, tprs_cosine, aucs_euclidean, aucs_cosine