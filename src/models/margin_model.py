import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, Tuple, List, Optional
import joblib
import config

class MarginPredictor:
    """
    Point Margin (Home PTS - Away PTS) prediction model using an ensemble of
    regularized linear models (Ridge) and gradient boosted decision trees (LightGBM & XGBoost).
    """
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type.lower()
        self.feature_names = []
        self.pipeline = None
        self.residual_std = 13.5 # Initial empirical NBA point margin standard deviation

    def _build_model(self):
        if self.model_type == "ridge":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=100.0, random_state=42))
            ])
        elif self.model_type == "lightgbm":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMRegressor(
                    n_estimators=150,
                    learning_rate=0.03,
                    num_leaves=15,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=-1
                ))
            ])
        elif self.model_type == "xgboost":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", xgb.XGBRegressor(
                    n_estimators=120,
                    learning_rate=0.03,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0
                ))
            ])
        else: # Ensemble Blend
            return None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names = list(X.columns)
        y = np.asarray(y).ravel()

        if self.model_type == "ensemble":
            # Train Ridge, LightGBM, and XGBoost components
            self.ridge = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=100.0, random_state=42))
            ]).fit(X, y)

            self.lgb = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMRegressor(
                    n_estimators=150, learning_rate=0.03, num_leaves=15,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=-1
                ))
            ]).fit(X, y)

            self.xgb = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", xgb.XGBRegressor(
                    n_estimators=120, learning_rate=0.03, max_depth=3,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
                ))
            ]).fit(X, y)

            preds = self.predict(X)
        else:
            self.pipeline = self._build_model()
            self.pipeline.fit(X, y)
            preds = self.pipeline.predict(X)

        # Estimate residual standard deviation for Normal CDF probability calculation
        residuals = y - preds
        self.residual_std = float(np.std(residuals))
        if self.residual_std < 1.0:
            self.residual_std = 13.5

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Align features
        if isinstance(X, pd.DataFrame):
            # Ensure missing feature columns are present
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = np.nan
            X_eval = X[self.feature_names]
        else:
            X_eval = X

        if self.model_type == "ensemble":
            p_ridge = self.ridge.predict(X_eval)
            p_lgb = self.lgb.predict(X_eval)
            p_xgb = self.xgb.predict(X_eval)
            # Weighted blend: 40% Ridge, 30% LightGBM, 30% XGBoost
            return 0.40 * p_ridge + 0.30 * p_lgb + 0.30 * p_xgb
        else:
            return self.pipeline.predict(X_eval)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        preds = self.predict(X)
        y = np.asarray(y).ravel()
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        return {
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "r2": round(float(r2), 4),
            "residual_std": round(float(self.residual_std), 3)
        }

    def save(self, filepath: Optional[str] = None):
        path = filepath or (config.MODELS_DIR / "margin_model.joblib")
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Optional[str] = None):
        path = filepath or (config.MODELS_DIR / "margin_model.joblib")
        return joblib.load(path)