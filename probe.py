from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler


class HallucinationProbe:
    _global_fold_id: int = 0
    _val_probs_all: list[np.ndarray] = []
    _val_y_all: list[np.ndarray] = []
    _val_preds_all: list[np.ndarray] = []

    def __init__(self) -> None:
        self._threshold: float = 0.5
        self._scaler = StandardScaler()
        self._models: list[LogisticRegression] = []
        self._C: float = 0.01
        self._random_state: int = 42

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        HallucinationProbe._global_fold_id += 1

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        X_scaled = self._scaler.fit_transform(X)

        rng = np.random.RandomState(self._random_state)
        self._models = []
        for i in range(10):
            boot = rng.choice(len(y), size=len(y), replace=True)
            if len(np.unique(y[boot])) < 2:
                boot = np.arange(len(y))
            clf = LogisticRegression(
                C=self._C,
                penalty="l2",
                class_weight=None,
                max_iter=5000,
                solver="lbfgs",
                random_state=self._random_state + i,
            )
            clf.fit(X_scaled[boot], y[boot])
            self._models.append(clf)

        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        best_t = 0.5
        best_acc = -1.0
        for t in np.linspace(0.01, 0.99, 99):
            pred = (probs >= t).astype(int)
            acc = accuracy_score(y_val, pred)
            if acc > best_acc:
                best_acc = acc
                best_t = t
        self._threshold = float(best_t)
        preds = (probs >= self._threshold).astype(int)

        HallucinationProbe._val_probs_all.append(probs)
        HallucinationProbe._val_y_all.append(y_val)
        HallucinationProbe._val_preds_all.append(preds)

        if HallucinationProbe._global_fold_id == 5:
            self._save_summary_plots()

        return self

    @classmethod
    def _save_summary_plots(cls):
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)

        all_y = np.concatenate(cls._val_y_all)
        all_probs = np.concatenate(cls._val_probs_all)
        all_preds = np.concatenate(cls._val_preds_all)

        plt.figure(figsize=(6, 6))
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for yt, pt in zip(cls._val_y_all, cls._val_probs_all):
            fpr, tpr, _ = roc_curve(yt, pt)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(auc(fpr, tpr))
            plt.plot(fpr, tpr, alpha=0.3, lw=1)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b', label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})", lw=2)
        tprs_upper = np.minimum(mean_tpr + np.std(tprs, axis=0), 1)
        tprs_lower = np.maximum(mean_tpr - np.std(tprs, axis=0), 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label="± 1 std")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC curves on validation folds")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plot_dir / "summary_roc_val.png", dpi=120)
        plt.grid(alpha=0.5)
        plt.close()

        plt.figure(figsize=(10, 4))
        n_folds = 5
        cmap_truth = plt.get_cmap("Blues")
        cmap_halluc = plt.get_cmap("Oranges")

        for i, (y_f, p_f) in enumerate(zip(cls._val_y_all, cls._val_probs_all)):
            color_val = 0.4 + (i / (n_folds - 1)) * 0.5
            
            plt.hist(p_f[y_f == 0], bins=20, alpha=0.3, density=True, 
                     label=f"truthful {i+1}", color=cmap_truth(color_val))
            plt.hist(p_f[y_f == 1], bins=20, alpha=0.3, density=True, 
                     label=f"hallucinated {i+1}", color=cmap_halluc(color_val))

        plt.xlabel("Predicted probability")
        plt.ylabel("Density")
        plt.title("Probability distributions")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), ncol=3, fontsize='small')
        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.savefig(plot_dir / "prob_dist_folds.png", dpi=120)
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.hist(all_probs[all_y == 0], bins=30, alpha=0.6, density=True, label="truthful")
        plt.hist(all_probs[all_y == 1], bins=30, alpha=0.6, density=True, label="hallucinated")
        plt.xlabel("Predicted probability")
        plt.ylabel("Density")
        plt.title("Probability distribution aggregated")
        plt.legend()
        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.savefig(plot_dir / "prob_dist_combined.png", dpi=120)
        plt.close()

        cm = confusion_matrix(all_y, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["truthful", "hallucinated"])
        disp.plot()
        plt.title("Aggregated confusion matrix on validation folds")
        plt.tight_layout()
        plt.savefig(plot_dir / "confusion_matrix_val.png", dpi=100)
        plt.close()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_scaled = self._scaler.transform(X)
        prob_pos = np.mean(
            [clf.predict_proba(X_scaled)[:, 1] for clf in self._models],
            axis=0,
        )
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)