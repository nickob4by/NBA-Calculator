import pytest
from src.betting.backtester import HistoricalBacktester

def test_backtester_run():
    bt = HistoricalBacktester(starting_bankroll=10000.0, kelly_fraction=0.15, min_edge=0.025, compound_bankroll=False)
    results = bt.run_backtest(season_filter="2024-25")
    
    assert "total_bets" in results
    assert "win_rate" in results
    assert "pnl" in results
    assert "roi_pct" in results
    assert "max_drawdown_pct" in results
    assert "avg_clv_pct" in results
    assert results["total_bets"] > 0
    assert len(results["equity_curve"]) > 0