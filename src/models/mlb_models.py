import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from typing import Dict, Optional
import joblib
import config
from src.models.calibration import ProbabilityCalibrator, compute_calibration_metrics

class MLBRunMarginPredictor:
    """
    Predicts MLB Run Margin (Home Runs - Away Runs) using an ensemble of Ridge and LightGBM.
    """
    def __init__(self):
        self.feature_names = []
        self.ridge = None
        self.lgb = None
        self.residual_std = 3.2 # Empirical MLB standard deviation of run margin

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        valid_cols = [c for c in X.columns if not X[c].isna().all()]
        self.feature_names = valid_cols
        X_clean = X[valid_cols]
        y = np.asarray(y).ravel()

        self.ridge = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=50.0, random_state=42))
        ]).fit(X_clean, y)

        self.lgb = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMRegressor(
                n_estimators=120,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=30,
                subsample=0.8,
                random_state=42,
                verbosity=-1
            ))
        ]).fit(X_clean, y)

        preds = self.predict(X_clean)
        self.residual_std = float(np.std(y - preds))
        if self.residual_std < 0.5:
            self.residual_std = 3.2
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = np.nan
            X_eval = X[self.feature_names]
        else:
            X_eval = X
        p_ridge = self.ridge.predict(X_eval)
        p_lgb = self.lgb.predict(X_eval)
        return 0.5 * p_ridge + 0.5 * p_lgb

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        preds = self.predict(X)
        y = np.asarray(y).ravel()
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        return {
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "residual_std": round(float(self.residual_std), 3)
        }

    def save(self, filepath: Optional[str] = None):
        path = filepath or (config.MODELS_DIR / "mlb_margin_model.joblib")
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Optional[str] = None):
        path = filepath or (config.MODELS_DIR / "mlb_margin_model.joblib")
        return joblib.load(path)

class MLBTotalsPredictor:
    """
    Predicts combined Total Runs (Home Runs + Away Runs) for MLB Over/Under markets.
    """
    def __init__(self):
        self.feature_names = []
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.03,
                num_leaves=12,
                min_child_samples=30,
                subsample=0.8,
                random_state=42,
                verbosity=-1
            ))
        ])
        self.total_residual_std = 3.8

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        valid_cols = [c for c in X.columns if not X[c].isna().all()]
        self.feature_names = valid_cols
        X_clean = X[valid_cols]
        y = np.asarray(y).ravel()
        self.pipeline.fit(X_clean, y)
        preds = self.pipeline.predict(X_clean)
        self.total_residual_std = float(np.std(y - preds))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = np.nan
            X_eval = X[self.feature_names]
        else:
            X_eval = X
        return self.pipeline.predict(X_eval)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        preds = self.predict(X)
        y = np.asarray(y).ravel()
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        return {
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "total_residual_std": round(float(self.total_residual_std), 3)
        }

    def save(self, filepath: Optional[str] = None):
        path = filepath or (config.MODELS_DIR / "mlb_totals_model.joblib")
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Optional[str] = None):
        path = filepath or (config.MODELS_DIR / "mlb_totals_model.joblib")
        return joblib.load(path)

class MLBWinProbabilityModel:
    """
    Computes calibrated Win Probabilities and Run Line (-1.5 / +1.5) cover probabilities for MLB.
    """
    def __init__(self):
        self.calibrator = ProbabilityCalibrator(method="platt")

    def fit_with_margins(self, predicted_margins: np.ndarray, y_true_wins: np.ndarray, sigma: float = 3.2):
        margins = np.asarray(predicted_margins)
        raw_probs = norm.cdf(margins / sigma)
        self.calibrator.fit(raw_probs, y_true_wins)
        return self

    def predict_proba(self, predicted_margins: np.ndarray, sigma: float = 3.2) -> np.ndarray:
        margins = np.asarray(predicted_margins)
        raw_probs = norm.cdf(margins / sigma)
        if self.calibrator.is_fitted:
            return self.calibrator.transform(raw_probs)
        return raw_probs

    def predict_run_line_proba(self, predicted_margins: np.ndarray, run_line: float = -1.5, sigma: float = 3.2) -> np.ndarray:
        """
        Calculates probability of Home covering Run Line (e.g. -1.5 => winning by 2+ runs).
        P(Margin > -run_line) = Phi((Margin + run_line) / sigma)
        """
        margins = np.asarray(predicted_margins)
        # Home covers -1.5 if margin >= 2 runs (margin + (-1.5) > 0)
        z = (margins + run_line) / sigma
        return np.clip(norm.cdf(z), 0.01, 0.99)

    def evaluate(self, y_true_wins: np.ndarray, win_probs: np.ndarray) -> Dict:
        return compute_calibration_metrics(y_true_wins, win_probs)

    def save(self, filepath: Optional[str] = None):
        path = filepath or (config.MODELS_DIR / "mlb_win_prob_model.joblib")
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Optional[str] = None):
        path = filepath or (config.MODELS_DIR / "mlb_win_prob_model.joblib")
        return joblib.load(path)