def streak(df, days, up=True): # consecutive up or down days
    """True on days that complete a run of `days` consecutive up (or down) days."""
    r = df["return_1d"]
    cond = (r > 0) if up else (r < 0)
    return cond.astype(int).rolling(days).sum() == days

def avg_move(df, days, positive=True, threshold=0.0): # average daily return over last `days` days
    """True when the average daily return over the last `days` days is
    above +threshold (positive=True) or below -threshold (positive=False)."""
    avg = df["return_1d"].rolling(days).mean()
    return (avg > threshold) if positive else (avg < -threshold)



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
    "long_after_3_down_days": {
        "description": "Buy at the close after 3 consecutive down days, sell n days later.",
        "signal": lambda df: streak(df, 3, up=False),
        "direction": 1,
        "already_own": False,
    },
    "own_sell_after_3_up_days": {
        "description": "Already own shares. After 3 consecutive up days, sell n days later (no entry cost).",
        "signal": lambda df: streak(df, 3, up=True),
        "direction": 1,
        "already_own": True,
    },
    "short_after_3_up_days": {
        "description": "Short at the close after 3 consecutive up days, buy back n days later.",
        "signal": lambda df: streak(df, 3, up=True),
        "direction": -1,
        "already_own": False,
    },
    "long_after_3d_avg_negative": {
        "description": "Buy at the close when the 3-day average return is negative, sell n days later.",
        "signal": lambda df: avg_move(df, 3, positive=False),
        "direction": 1,
        "already_own": False,
    },
    "own_sell_after_3d_avg_positive": {
        "description": "Already own shares. When the 3-day average return is positive, sell n days later (no entry cost).",
        "signal": lambda df: avg_move(df, 3, positive=True),
        "direction": 1,
        "already_own": True,
    },
    "short_after_3d_avg_positive": {
        "description": "Short at the close when the 3-day average return is positive, buy back n days later.",
        "signal": lambda df: avg_move(df, 3, positive=True),
        "direction": -1,
        "already_own": False,
    },
    "buy_after_n_day_avg_negative": {
        "description": "Buy when the n-day average return is negative. Sharpe averaged over holds, for each n.",
        "signal_n": lambda df, n: avg_move(df, n, positive=False),
        "direction": 1,
        "already_own": False,
    },
    "own_sell_after_n_day_avg_positive": {
        "description": "Already own shares. Sell n days later after a positive n-day average. Sharpe averaged over holds, for each n.",
        "signal_n": lambda df, n: avg_move(df, n, positive=True),
        "direction": 1,
        "already_own": True,
    },
    "short_after_n_day_avg_positive": {
        "description": "Short when the n-day average return is positive. Sharpe averaged over holds, for each n.",
        "signal_n": lambda df, n: avg_move(df, n, positive=True),
        "direction": -1,
        "already_own": False,
    },
    "long_dip_in_uptrend": {
        "description": "Buy after a >1% drop, but only when price is above its 200-day average.",
        "signal": lambda df: (df["return_1d"] < -0.01) & (df["distance_200ma"] > 0),
        "direction": 1,
        "already_own": False,
    },
    "long_pullback_in_uptrend": {
        "description": "Buy when price is >3% below its 20-day average, but above its 200-day average.",
        "signal": lambda df: (df["distance_20ma"] < -0.03) & (df["distance_200ma"] > 0),
        "direction": 1,
        "already_own": False,
    },
    "long_when_above_200ma": {
        "description": "Buy on any day price is above its 200-day average (classic trend filter).",
        "signal": lambda df: df["distance_200ma"] > 0,
        "direction": 1,
        "already_own": False,
    },
    "long_positive_60d_momentum": {
        "description": "Buy when the past 60-day return is positive.",
        "signal": lambda df: df["return_60d"] > 0,
        "direction": 1,
        "already_own": False,
    },
    "long_5d_drop_in_uptrend": {
        "description": "Buy after a 5-day return below -2%, only when above the 200-day average.",
        "signal": lambda df: (df["return_5d"] < -0.02) & (df["distance_200ma"] > 0),
        "direction": 1,
        "already_own": False,
    },
    "long_dip_in_calm_market": {
        "description": "Buy after a >1% drop, only when 20-day volatility is below its own 1-year median.",
        "signal": lambda df: (df["return_1d"] < -0.01)
                             & (df["vol_20"] < df["vol_20"].rolling(250).median()),
        "direction": 1,
        "already_own": False,
    },
}


def get_rule(name):
    if name not in RULES:
        raise KeyError(f"Unknown rule '{name}'. Available: {list(RULES)}")
    return RULES[name]