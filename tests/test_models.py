import pytest
import numpy as np
import pandas as pd
from src.models.margin_model import MarginPredictor
from src.models.totals_model import TotalsPredictor
from src.models.win_prob_model import WinProbabilityModel
from src.models.calibration import compute_calibration_metrics, ProbabilityCalibrator

def test_margin_and_win_prob_models():
    # Construct small synthetic dataset
    rng = np.random.RandomState(42)
    n_samples = 100
    X = pd.DataFrame({
        "diff_roll_net_rating_w10": rng.normal(0, 5, n_samples),
        "diff_roll_efg_pct_w10": rng.normal(0, 0.05, n_samples),
        "diff_rest_days": rng.choice([-1, 0, 1, 2], n_samples)
    })
    y_margin = 2.5 * X["diff_roll_net_rating_w10"] + 3.0 + rng.normal(0, 5, n_samples)
    y_win = (y_margin > 0).astype(int)

    # Train margin model
    m_model = MarginPredictor(model_type="ridge").fit(X, y_margin)
    preds = m_model.predict(X)
    assert len(preds) == n_samples
    assert m_model.residual_std > 0

    # Test Win Probability Model
    w_model = WinProbabilityModel(mode="cdf_calibrated").fit_with_margins(preds, y_win, sigma=m_model.residual_std)
    probs = w_model.predict_proba(predicted_margins=preds, sigma=m_model.residual_std)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Test calibration metrics
    metrics = compute_calibration_metrics(y_win, probs)
    assert "log_loss" in metrics
    assert "brier_score" in metrics
    assert "accuracy" in metrics