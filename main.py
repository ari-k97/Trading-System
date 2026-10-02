from data import load_prices, get_single
from features import add_features
from backtest import sweep_hold
from plots import plot_sweep
from rules import get_rule

# -------------------- settings ------------------------
RULE_NAME = "long_after_drop_1pct"
TICKER = "IVV.AX"
HOLDS = range(1, 61)
COST = 0.0005
# ------------------------------------------------------

if __name__ == "__main__":
    rule = get_rule(RULE_NAME)
    print(f"Rule: {RULE_NAME}\n{rule['description']}\n")

    prices = load_prices(TICKER)
    df = get_single(prices, TICKER)
    df = add_features(df)

    event = rule["signal"](df)
    print(f"events: {event.sum()}")

    sweep = sweep_hold(
        df, event,
        holds=HOLDS,
        cost_per_side=COST,
        direction=rule["direction"],
        already_own=rule["already_own"],
    )
    print(sweep.round(4))

    plot_sweep(sweep, "avg_trade", title=f"{RULE_NAME} ({TICKER}): avg trade")
    plot_sweep(sweep, "sharpe", title=f"{RULE_NAME} ({TICKER}): Sharpe")