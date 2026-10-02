RULES = {
    "long_after_drop_1pct": {
        "description": "Buy at the close after a >1% daily drop, sell n days later.",
        "signal": lambda df: df["return_1d"] < -0.01,
        "direction": 1,
        "already_own": False,
    },
    "long_after_rise_1pct": {
        "description": "Buy at the close after a >1% daily rise, sell n days later.",
        "signal": lambda df: df["return_1d"] > 0.01,
        "direction": 1,
        "already_own": False,
    },
    "short_after_rise_1pct": {
        "description": "Short at the close after a >1% daily rise, buy back n days later.",
        "signal": lambda df: df["return_1d"] > 0.01,
        "direction": -1,
        "already_own": False,
    },
    "own_sell_after_rise_1pct": {
        "description": "Already own shares. After a >1% rise, sell n days later (no entry cost).",
        "signal": lambda df: df["return_1d"] > 0.01,
        "direction": 1,
        "already_own": True,
    },
    "baseline_long": {
        "description": "Benchmark: buy on any day, sell n days later.",
        "signal": lambda df: df["return_1d"].notna(),
        "direction": 1,
        "already_own": False,
    },
    "long_below_200ma_5pct": {
        "description": "Buy when price is >5% below its 200-day average, sell n days later.",
        "signal": lambda df: df["distance_200ma"] < -0.05,
        "direction": 1,
        "already_own": False,
    },
}


def get_rule(name):
    if name not in RULES:
        raise KeyError(f"Unknown rule '{name}'. Available: {list(RULES)}")
    return RULES[name]