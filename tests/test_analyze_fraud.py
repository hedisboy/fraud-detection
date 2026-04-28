import pandas as pd
import pytest
from analyze_fraud import score_transactions, summarize_results


def make_scored(*rows):
    return pd.DataFrame(rows, columns=["transaction_id", "account_id", "amount_usd",
                                        "risk_score", "risk_label", "device_risk_score",
                                        "is_international", "velocity_24h", "failed_logins_24h",
                                        "prior_chargebacks", "is_large_amount", "login_pressure",
                                        "country"])


def simple_scored():
    return pd.DataFrame({
        "transaction_id": [1, 2, 3, 4, 5],
        "amount_usd":     [500.0, 300.0, 200.0, 100.0, 150.0],
        "risk_label":     ["high", "high", "medium", "low", "low"],
    })


def simple_chargebacks(*ids):
    return pd.DataFrame({"transaction_id": list(ids)})


# --- summarize_results: transaction counts ---

def test_transaction_counts_per_label():
    summary = summarize_results(simple_scored(), simple_chargebacks())
    row = summary.set_index("risk_label")["transactions"]
    assert row["high"] == 2
    assert row["medium"] == 1
    assert row["low"] == 2


# --- summarize_results: amounts ---

def test_total_amount_per_label():
    summary = summarize_results(simple_scored(), simple_chargebacks())
    row = summary.set_index("risk_label")["total_amount_usd"]
    assert row["high"] == pytest.approx(800.0)
    assert row["low"] == pytest.approx(250.0)


def test_avg_amount_per_label():
    summary = summarize_results(simple_scored(), simple_chargebacks())
    row = summary.set_index("risk_label")["avg_amount_usd"]
    assert row["high"] == pytest.approx(400.0)
    assert row["low"] == pytest.approx(125.0)


# --- summarize_results: chargeback rate ---

def test_chargeback_rate_all_fraud():
    # Both high-risk transactions are chargebacks → rate = 1.0
    summary = summarize_results(simple_scored(), simple_chargebacks(1, 2))
    row = summary.set_index("risk_label")["chargeback_rate"]
    assert row["high"] == pytest.approx(1.0)


def test_chargeback_rate_no_fraud():
    summary = summarize_results(simple_scored(), simple_chargebacks())
    row = summary.set_index("risk_label")["chargeback_rate"]
    assert row["low"] == pytest.approx(0.0)


def test_chargeback_rate_partial():
    # One of two high-risk transactions is a chargeback → rate = 0.5
    summary = summarize_results(simple_scored(), simple_chargebacks(1))
    row = summary.set_index("risk_label")["chargeback_rate"]
    assert row["high"] == pytest.approx(0.5)


def test_chargeback_not_double_counted():
    # Chargeback on a low-risk transaction must not inflate the high count.
    summary = summarize_results(simple_scored(), simple_chargebacks(4))
    row = summary.set_index("risk_label")["chargeback_rate"]
    assert row["high"] == pytest.approx(0.0)
    assert row["low"] == pytest.approx(0.5)


# --- score_transactions pipeline ---

def make_pipeline_inputs():
    transactions = pd.DataFrame({
        "transaction_id":   [101, 102],
        "account_id":       [1, 2],
        "amount_usd":       [50.0, 1500.0],
        "device_risk_score":[10,   85],
        "is_international": [0,    1],
        "velocity_24h":     [1,    8],
        "failed_logins_24h":[0,    6],
    })
    accounts = pd.DataFrame({
        "account_id":        [1, 2],
        "prior_chargebacks": [0, 2],
        "country":           ["US", "RU"],
    })
    return transactions, accounts


def test_score_transactions_adds_risk_score_column():
    txns, accts = make_pipeline_inputs()
    result = score_transactions(txns, accts)
    assert "risk_score" in result.columns
    assert result["risk_score"].between(0, 100).all()


def test_score_transactions_adds_risk_label_column():
    txns, accts = make_pipeline_inputs()
    result = score_transactions(txns, accts)
    assert "risk_label" in result.columns
    assert set(result["risk_label"]).issubset({"low", "medium", "high"})


def test_score_transactions_clean_scores_high_risk():
    # The high-signal row (tx 102) should score high.
    txns, accts = make_pipeline_inputs()
    result = score_transactions(txns, accts).set_index("transaction_id")
    assert result.loc[102, "risk_label"] == "high"


def test_score_transactions_clean_scores_low_risk():
    # The low-signal row (tx 101) should score low.
    txns, accts = make_pipeline_inputs()
    result = score_transactions(txns, accts).set_index("transaction_id")
    assert result.loc[101, "risk_label"] == "low"
