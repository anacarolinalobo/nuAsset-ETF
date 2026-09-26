import pandas as pd

from src.selection import select_portfolio
from src.weighting import compute_weights
from src.config import MomentumConfig


def _config(**overrides):
    base = dict(
        entry_percentile=0.70, hold_percentile=0.50, min_names=2, max_names=4, weight_cap=0.40
    )
    base.update(overrides)
    return MomentumConfig(**base)


def test_new_entrant_needs_tighter_cutoff():
    signal = pd.Series({"A": 2.0, "B": 1.5, "C": 0.5, "D": -0.5, "E": -1.0})
    cfg = _config(min_names=1, max_names=5)
    selected = select_portfolio(signal, current_holdings=set(), config=cfg)
    assert "A" in selected
    assert "E" not in selected


def test_buffer_rule_keeps_existing_holding_in_wider_band():
    signal = pd.Series({"A": 2.0, "B": 1.5, "C": 0.3, "D": -0.5, "E": -1.0})
    cfg = _config(min_names=1, max_names=5)
    # C está abaixo do corte de entrada mas seria mantido se já estivesse na carteira
    selected_new = select_portfolio(signal, current_holdings=set(), config=cfg)
    selected_hold = select_portfolio(signal, current_holdings={"C"}, config=cfg)
    assert "C" not in selected_new
    assert "C" in selected_hold


def test_size_limits_respected():
    signal = pd.Series({f"T{i}": float(10 - i) for i in range(10)})
    cfg = _config(min_names=3, max_names=5)
    selected = select_portfolio(signal, current_holdings=set(), config=cfg)
    assert 3 <= len(selected) <= 5


def test_weights_sum_to_one_and_respect_cap():
    signal = pd.Series({"A": 5.0, "B": 1.0, "C": 1.0, "D": 1.0})
    cfg = _config(weight_cap=0.40)
    weights = compute_weights(signal, {"A", "B", "C", "D"}, cfg)

    assert abs(weights.sum() - 1.0) < 1e-9
    assert (weights <= cfg.weight_cap + 1e-9).all()


def test_negative_scores_get_zero_weight():
    signal = pd.Series({"A": 2.0, "B": -1.0, "C": 0.5})
    cfg = _config(weight_cap=0.9)
    weights = compute_weights(signal, {"A", "B", "C"}, cfg)
    assert weights["B"] == 0.0
