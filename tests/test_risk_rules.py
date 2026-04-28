from risk_rules import label_risk, score_transaction


BASE_TX = {
    "device_risk_score": 10,
    "is_international": 0,
    "amount_usd": 50,
    "velocity_24h": 1,
    "failed_logins_24h": 0,
    "prior_chargebacks": 0,
}


# --- label_risk ---

def test_label_risk_low():
    assert label_risk(0) == "low"
    assert label_risk(10) == "low"
    assert label_risk(29) == "low"


def test_label_risk_medium():
    assert label_risk(30) == "medium"
    assert label_risk(35) == "medium"
    assert label_risk(59) == "medium"


def test_label_risk_high():
    assert label_risk(60) == "high"
    assert label_risk(75) == "high"
    assert label_risk(100) == "high"


# --- score bounds ---

def test_score_minimum_is_zero():
    assert score_transaction(BASE_TX) == 0


def test_score_clamped_to_100():
    # All signals at maximum: 25+15+25+20+20+20 = 125, must clamp to 100.
    tx = {
        "device_risk_score": 85,
        "is_international": 1,
        "amount_usd": 1500,
        "velocity_24h": 8,
        "failed_logins_24h": 6,
        "prior_chargebacks": 2,
    }
    assert score_transaction(tx) == 100


# --- device risk score ---

def test_device_risk_high_tier_adds_25():
    assert score_transaction({**BASE_TX, "device_risk_score": 70}) == 25
    assert score_transaction({**BASE_TX, "device_risk_score": 85}) == 25


def test_device_risk_medium_tier_adds_10():
    assert score_transaction({**BASE_TX, "device_risk_score": 40}) == 10
    assert score_transaction({**BASE_TX, "device_risk_score": 65}) == 10


def test_device_risk_low_tier_adds_nothing():
    assert score_transaction({**BASE_TX, "device_risk_score": 39}) == 0


# --- international ---

def test_international_adds_15():
    assert score_transaction({**BASE_TX, "is_international": 1}) == 15


def test_domestic_adds_nothing():
    assert score_transaction({**BASE_TX, "is_international": 0}) == 0


# --- amount ---

def test_amount_large_tier_adds_25():
    assert score_transaction({**BASE_TX, "amount_usd": 1000}) == 25
    assert score_transaction({**BASE_TX, "amount_usd": 2000}) == 25


def test_amount_medium_tier_adds_10():
    assert score_transaction({**BASE_TX, "amount_usd": 500}) == 10
    assert score_transaction({**BASE_TX, "amount_usd": 999}) == 10


def test_amount_low_tier_adds_nothing():
    assert score_transaction({**BASE_TX, "amount_usd": 499}) == 0


# --- velocity ---

def test_velocity_high_tier_adds_20():
    assert score_transaction({**BASE_TX, "velocity_24h": 6}) == 20
    assert score_transaction({**BASE_TX, "velocity_24h": 10}) == 20


def test_velocity_medium_tier_adds_5():
    assert score_transaction({**BASE_TX, "velocity_24h": 3}) == 5
    assert score_transaction({**BASE_TX, "velocity_24h": 5}) == 5


def test_velocity_low_tier_adds_nothing():
    assert score_transaction({**BASE_TX, "velocity_24h": 2}) == 0


# --- failed logins ---

def test_failed_logins_high_tier_adds_20():
    assert score_transaction({**BASE_TX, "failed_logins_24h": 5}) == 20
    assert score_transaction({**BASE_TX, "failed_logins_24h": 9}) == 20


def test_failed_logins_medium_tier_adds_10():
    assert score_transaction({**BASE_TX, "failed_logins_24h": 2}) == 10
    assert score_transaction({**BASE_TX, "failed_logins_24h": 4}) == 10


def test_failed_logins_none_adds_nothing():
    assert score_transaction({**BASE_TX, "failed_logins_24h": 1}) == 0


# --- prior chargebacks ---

def test_prior_chargebacks_two_or_more_adds_20():
    assert score_transaction({**BASE_TX, "prior_chargebacks": 2}) == 20
    assert score_transaction({**BASE_TX, "prior_chargebacks": 5}) == 20


def test_prior_chargebacks_one_adds_5():
    assert score_transaction({**BASE_TX, "prior_chargebacks": 1}) == 5


def test_prior_chargebacks_none_adds_nothing():
    assert score_transaction({**BASE_TX, "prior_chargebacks": 0}) == 0


# --- additive combination ---

def test_scores_are_additive():
    # device(25) + international(15) = 40
    tx = {**BASE_TX, "device_risk_score": 75, "is_international": 1}
    assert score_transaction(tx) == 40
