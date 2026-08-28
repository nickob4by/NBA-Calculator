import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from typing import Dict, Optional, Tuple
import joblib
import config
from src.models.calibration import ProbabilityCalibrator, compute_calibration_metrics

class WinProbabilityModel:
    """
    Computes calibrated win probabilities either via normal distribution CDF mapping
    on predicted point margin Phi(margin / sigma), or via a direct calibrated classifier.
    """
    def __init__(self, mode: str = "cdf_calibrated"):
        self.mode = mode.lower() # 'cdf', 'cdf_calibrated', or 'classifier'
        self.calibrator = ProbabilityCalibrator(method="platt")
        self.classifier = None
        self.feature_names = []

    def margin_to_probability_cdf(self, predicted_margins: np.ndarray, sigma: float = 13.5) -> np.ndarray:
        """
        Maps predicted point margin (Home - Away) to home win probability using the Gaussian CDF.
        """
        margins = np.asarray(predicted_margins)
        if sigma <= 0:
            sigma = 13.5
        probs = norm.cdf(margins / sigma)
        return np.clip(probs, 0.01, 0.99)

    def fit_with_margins(self, predicted_margins: np.ndarray, y_true_wins: np.ndarray, sigma: float = 13.5):
        """
        Fits probability calibrator (Platt scaling) on raw CDF win probabilities.
        """
        raw_probs = self.margin_to_probability_cdf(predicted_margins, sigma)
        self.calibrator.fit(raw_probs, y_true_wins)
        return self

    def fit_classifier(self, X: pd.DataFrame, y_true_wins: np.ndarray):
        """
        Fits a direct gradient boosted classifier with probability calibration.
        """
        self.feature_names = list(X.columns)
        self.classifier = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.03,
                num_leaves=15,
                subsample=0.8,
                random_state=42,
                verbosity=-1
            ))
        ])
        self.classifier.fit(X, y_true_wins)
        raw_probs = self.classifier.predict_proba(X)[:, 1]
        self.calibrator.fit(raw_probs, y_true_wins)
        return self

    def predict_proba(self, X: Optional[pd.DataFrame] = None, predicted_margins: Optional[np.ndarray] = None, sigma: float = 13.5) -> np.ndarray:
        """
        Returns calibrated Home Win Probabilities.
        """
        if self.mode in ["cdf", "cdf_calibrated"] and predicted_margins is not None:
            if sigma <= 1.0:
                sigma = 13.5
            raw_probs = self.margin_to_probability_cdf(predicted_margins, sigma)
            if self.mode == "cdf_calibrated" and self.calibrator.is_fitted:
                calibrated = self.calibrator.transform(raw_probs)
                final_probs = 0.5 * raw_probs + 0.5 * calibrated
                return np.clip(final_probs, 0.15, 0.85)
            return np.clip(raw_probs, 0.15, 0.85)
        
        elif self.classifier is not None and X is not None:
            raw_probs = self.classifier.predict_proba(X[self.feature_names])[:, 1]
            return self.calibrator.transform(raw_probs)
        else:
            raise ValueError(f"Unknown mode {self.mode} or missing required inputs")

    def evaluate(self, y_true_wins: np.ndarray, win_probs: np.ndarray) -> Dict:
        return compute_calibration_metrics(y_true_wins, win_probs)

    def save(self, filepath: Optional[str] = None):
        path = filepath or (config.MODELS_DIR / "win_prob_model.joblib")
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Optional[str] = None):
        path = filepath or (config.MODELS_DIR / "win_prob_model.joblib")
        return joblib.load(path)