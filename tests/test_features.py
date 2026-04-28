import pandas as pd
import pytest
from features import build_model_frame


def make_transactions(**overrides):
    row = {
        "transaction_id": 1,
        "account_id": 100,
        "amount_usd": 50.0,
        "failed_logins_24h": 0,
        "device_risk_score": 10,
        "is_international": 0,
        "velocity_24h": 1,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def make_accounts(**overrides):
    row = {
        "account_id": 100,
        "prior_chargebacks": 0,
        "country": "US",
    }
    row.update(overrides)
    return pd.DataFrame([row])


# --- merge ---

def test_account_columns_joined():
    txns = make_transactions()
    accts = make_accounts(prior_chargebacks=2)
    result = build_model_frame(txns, accts)
    assert "prior_chargebacks" in result.columns
    assert result["prior_chargebacks"].iloc[0] == 2


def test_unmatched_account_produces_nan():
    txns = make_transactions(account_id=999)
    accts = make_accounts(account_id=100)
    result = build_model_frame(txns, accts)
    assert pd.isna(result["prior_chargebacks"].iloc[0])


# --- is_large_amount ---

def test_is_large_amount_true_at_threshold():
    result = build_model_frame(make_transactions(amount_usd=1000), make_accounts())
    assert result["is_large_amount"].iloc[0] == 1


def test_is_large_amount_true_above_threshold():
    result = build_model_frame(make_transactions(amount_usd=2500), make_accounts())
    assert result["is_large_amount"].iloc[0] == 1


def test_is_large_amount_false_below_threshold():
    result = build_model_frame(make_transactions(amount_usd=999.99), make_accounts())
    assert result["is_large_amount"].iloc[0] == 0


# --- login_pressure ---

def test_login_pressure_none_for_zero_logins():
    result = build_model_frame(make_transactions(failed_logins_24h=0), make_accounts())
    assert result["login_pressure"].iloc[0] == "none"


def test_login_pressure_low_for_one_or_two_logins():
    for n in (1, 2):
        result = build_model_frame(make_transactions(failed_logins_24h=n), make_accounts())
        assert result["login_pressure"].iloc[0] == "low", f"failed for failed_logins_24h={n}"


def test_login_pressure_high_for_three_or_more_logins():
    for n in (3, 7):
        result = build_model_frame(make_transactions(failed_logins_24h=n), make_accounts())
        assert result["login_pressure"].iloc[0] == "high", f"failed for failed_logins_24h={n}"
