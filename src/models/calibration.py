import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score
from typing import Tuple, Dict, Optional
import joblib

class ProbabilityCalibrator:
    """
    Calibrates raw predictive probabilities using either Platt Scaling (Logistic sigmoid)
    or Isotonic Regression to ensure predicted probabilities match true empirical win frequencies.
    """
    def __init__(self, method: str = "platt"):
        self.method = method.lower()
        self.calibrator = None
        self.is_fitted = False

    def fit(self, y_prob_raw: np.ndarray, y_true: np.ndarray):
        y_prob_raw = np.clip(np.asarray(y_prob_raw).reshape(-1, 1), 1e-6, 1.0 - 1e-6)
        y_true = np.asarray(y_true).ravel()

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
            self.calibrator.fit(y_prob_raw.ravel(), y_true)
        else: # Platt Scaling via Logistic Regression with strong L2 regularization
            logits = np.log(y_prob_raw / (1.0 - y_prob_raw))
            self.calibrator = LogisticRegression(solver="lbfgs", C=0.1, max_iter=100)
            self.calibrator.fit(logits, y_true)

        self.is_fitted = True
        return self

    def transform(self, y_prob_raw: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.asarray(y_prob_raw)

        y_prob_raw = np.clip(np.asarray(y_prob_raw).reshape(-1, 1), 1e-6, 1.0 - 1e-6)

        if self.method == "isotonic":
            calibrated = self.calibrator.predict(y_prob_raw.ravel())
        else:
            logits = np.log(y_prob_raw / (1.0 - y_prob_raw))
            calibrated = self.calibrator.predict_proba(logits)[:, 1]

        return np.clip(calibrated, 0.01, 0.99)

    def fit_transform(self, y_prob_raw: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.fit(y_prob_raw, y_true)
        return self.transform(y_prob_raw)

def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Computes Log Loss, Brier Score, ECE (Expected Calibration Error), and reliability curve bins.
    """
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob), 1e-6, 1.0 - 1e-6)

    ll = float(log_loss(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    acc = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5

    # Compute reliability curve and ECE
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_data = []
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        if count > 0:
            mean_prob = float(np.mean(y_prob[mask]))
            mean_true = float(np.mean(y_true[mask]))
            ece += (count / total_samples) * abs(mean_prob - mean_true)
            bin_data.append({
                "bin": i,
                "bin_start": float(bins[i]),
                "bin_end": float(bins[i + 1]),
                "count": count,
                "confidence": round(mean_prob, 4),
                "accuracy": round(mean_true, 4)
            })

    return {
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "ece": round(float(ece), 4),
        "reliability_bins": bin_data
    }