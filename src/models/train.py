import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.db.database import db
from src.features.matchup_builder import build_full_feature_dataset, get_feature_columns
from src.models.margin_model import MarginPredictor
from src.models.totals_model import TotalsPredictor
from src.models.win_prob_model import WinProbabilityModel
from src.models.calibration import compute_calibration_metrics
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_all_models(n_splits: int = 5) -> dict:
    """
    Executes Time-Series Cross Validation and trains final production models
    for point margin, totals, and calibrated win probabilities.
    """
    logger.info("Loading game logs from SQLite for model training...")
    logs_df = db.fetch_df("SELECT * FROM team_game_logs ORDER BY game_date, game_id")
    
    if logs_df.empty:
        raise ValueError("No game logs found in database. Run ingestion first!")

    logger.info(f"Loaded {len(logs_df)} team game logs. Building zero-lookahead features...")
    matchups_df = build_full_feature_dataset(logs_df)

    # Filter out games with missing targets (unplayed games)
    train_df = matchups_df.dropna(subset=["point_margin", "total_points", "home_win"]).copy()
    feature_cols = get_feature_columns(train_df)

    logger.info(f"Prepared {len(train_df)} completed matchups with {len(feature_cols)} predictor features.")

    X = train_df[feature_cols]
    y_margin = train_df["point_margin"].values
    y_total = train_df["total_points"].values
    y_win = train_df["home_win"].values

    # TimeSeriesSplit cross validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_oof_margins = np.zeros(len(train_df))
    cv_oof_totals = np.zeros(len(train_df))
    cv_oof_win_probs = np.zeros(len(train_df))
    cv_indices = []

    logger.info(f"Running {n_splits}-fold TimeSeriesSplit cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_m_tr, y_m_val = y_margin[train_idx], y_margin[val_idx]
        y_t_tr, y_t_val = y_total[train_idx], y_total[val_idx]
        y_w_tr, y_w_val = y_win[train_idx], y_win[val_idx]

        # 1. Train fold margin model
        m_model = MarginPredictor(model_type="ensemble").fit(X_tr, y_m_tr)
        val_pred_margin = m_model.predict(X_val)
        cv_oof_margins[val_idx] = val_pred_margin

        # 2. Train fold totals model
        t_model = TotalsPredictor().fit(X_tr, y_t_tr)
        val_pred_total = t_model.predict(X_val)
        cv_oof_totals[val_idx] = val_pred_total

        # 3. Train fold win probability calibrator
        w_model = WinProbabilityModel(mode="cdf_calibrated")
        w_model.fit_with_margins(m_model.predict(X_tr), y_w_tr, sigma=m_model.residual_std)
        val_pred_win_prob = w_model.predict_proba(predicted_margins=val_pred_margin, sigma=m_model.residual_std)
        cv_oof_win_probs[val_idx] = val_pred_win_prob

        cv_indices.extend(val_idx)

    # Evaluate out-of-fold performance on evaluated validation splits
    eval_idx = np.array(cv_indices)
    oof_margin_eval = {
        "mae": round(float(np.mean(np.abs(y_margin[eval_idx] - cv_oof_margins[eval_idx]))), 3),
        "rmse": round(float(np.sqrt(np.mean((y_margin[eval_idx] - cv_oof_margins[eval_idx]) ** 2))), 3)
    }

    oof_total_eval = {
        "mae": round(float(np.mean(np.abs(y_total[eval_idx] - cv_oof_totals[eval_idx]))), 3),
        "rmse": round(float(np.sqrt(np.mean((y_total[eval_idx] - cv_oof_totals[eval_idx]) ** 2))), 3)
    }

    oof_win_eval = compute_calibration_metrics(y_win[eval_idx], cv_oof_win_probs[eval_idx])

    logger.info(f"OOF Margin MAE: {oof_margin_eval['mae']} pts | RMSE: {oof_margin_eval['rmse']} pts")
    logger.info(f"OOF Totals MAE: {oof_total_eval['mae']} pts | RMSE: {oof_total_eval['rmse']} pts")
    logger.info(f"OOF Win Prob Log Loss: {oof_win_eval['log_loss']} | Brier Score: {oof_win_eval['brier_score']} | Accuracy: {oof_win_eval['accuracy'] * 100:.1f}%")

    # Fit final production models on full history
    logger.info("Training final production models on full dataset...")
    final_margin_model = MarginPredictor(model_type="ensemble").fit(X, y_margin)
    final_totals_model = TotalsPredictor().fit(X, y_total)
    
    full_pred_margins = final_margin_model.predict(X)
    final_win_model = WinProbabilityModel(mode="cdf_calibrated").fit_with_margins(
        full_pred_margins, y_win, sigma=final_margin_model.residual_std
    )

    # Save models
    final_margin_model.save()
    final_totals_model.save()
    final_win_model.save()

    metrics_summary = {
        "num_games": len(train_df),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "margin_metrics": oof_margin_eval,
        "totals_metrics": oof_total_eval,
        "win_probability_metrics": oof_win_eval,
        "residual_std": final_margin_model.residual_std,
        "total_residual_std": final_totals_model.total_residual_std
    }

    with open(config.MODELS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("Saved production models and metrics to data/models/")
    return metrics_summary

if __name__ == "__main__":
    train_all_models()