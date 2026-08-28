import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from typing import Dict, Optional
import joblib
import config

class TotalsPredictor:
    """
    Predicts combined total game score (Home PTS + Away PTS) for Over/Under markets.
    """
    def __init__(self):
        self.feature_names = []
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMRegressor(
                n_estimators=120,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=25,
                subsample=0.8,
                random_state=42,
                verbosity=-1
            ))
        ])
        self.total_residual_std = 18.0

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names = list(X.columns)
        y = np.asarray(y).ravel()
        self.pipeline.fit(X, y)
        preds = self.pipeline.predict(X)
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
        path = filepath or (config.MODELS_DIR / "totals_model.joblib")
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Optional[str] = None):
        path = filepath or (config.MODELS_DIR / "totals_model.joblib")
        return joblib.load(path)