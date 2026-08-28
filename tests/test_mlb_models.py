import pytest
import numpy as np
import pandas as pd
from src.models.mlb_models import MLBRunMarginPredictor, MLBTotalsPredictor, MLBWinProbabilityModel

def test_mlb_models_and_run_line():
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame({
        "diff_roll_pythag_win_pct_w10": rng.normal(0, 0.1, n),
        "diff_roll_ops_w10": rng.normal(0, 0.05, n),
        "diff_roll_fip_proxy_w10": rng.normal(0, 0.5, n),
        "diff_rest_days": rng.choice([-1, 0, 1], n)
    })
    y_margin = 1.5 * X["diff_roll_pythag_win_pct_w10"] + 0.35 + rng.normal(0, 1.5, n)
    y_win = (y_margin > 0).astype(int)

    m_model = MLBRunMarginPredictor().fit(X, y_margin)
    preds = m_model.predict(X)
    assert len(preds) == n
    assert m_model.residual_std > 0

    w_model = MLBWinProbabilityModel().fit_with_margins(preds, y_win, sigma=m_model.residual_std)
    probs = w_model.predict_proba(preds, sigma=m_model.residual_std)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Test Run Line (-1.5)
    run_line_probs = w_model.predict_run_line_proba(preds, run_line=-1.5, sigma=m_model.residual_std)
    assert np.all(run_line_probs >= 0.0) and np.all(run_line_probs <= 1.0)
    # Covering -1.5 is harder than outright winning, so P(Cover -1.5) <= P(Win)
    assert np.all(run_line_probs <= probs + 0.05)