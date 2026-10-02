import matplotlib.pyplot as plt
import pandas as pd

from data import load_prices, get_single
from features import add_features
from backtest import sweep_hold
from plots import plot_sweep
from rules import get_rule

# ---------- settings: change these, then run ----------
RULE_NAME = "long_pullback_in_uptrend"
TICKER = "AAPL"
HOLDS = range(1, 501)        # holding periods to test
LOOKBACKS = range(1, 1000)    # only used by the "n_day" rules
COST = 0.0005
# ------------------------------------------------------

if __name__ == "__main__":
    rule = get_rule(RULE_NAME)
    print(f"Rule: {RULE_NAME}\n{rule['description']}\n")

    prices = load_prices(TICKER)
    df = get_single(prices, TICKER)
    df = add_features(df)

    common = dict(cost_per_side=COST,
                  direction=rule["direction"],
                  already_own=rule["already_own"])

    if "signal_n" in rule:
        # one number per lookback n: Sharpe averaged across all the holds
        rows = []
        for n in LOOKBACKS:
            event = rule["signal_n"](df, n)
            sweep = sweep_hold(df, event, holds=HOLDS, **common)
            if sweep.empty:
                continue
            rows.append({
                "lookback": n,
                "avg_sharpe": sweep["sharpe"].mean(),
                "avg_n_trades": sweep["n_trades"].mean(),
            })
        summary = pd.DataFrame(rows).set_index("lookback")

        # baseline: same trade type, no signal (fires every day)
        base = sweep_hold(df, df["return_1d"].notna(), holds=HOLDS, **common)
        baseline = base["sharpe"].mean()

        print(summary.round(3))
        print(f"\nbaseline avg Sharpe: {baseline:.3f}")

        plt.figure(figsize=(12, 5))
        plt.plot(summary.index, summary["avg_sharpe"], marker="o", label="rule")
        plt.axhline(baseline, color="grey", linestyle="--", label="baseline (no signal)")
        plt.title(f"{RULE_NAME} ({TICKER})")
        plt.xlabel("Lookback n (days)")
        plt.ylabel(f"Average Sharpe over holds {HOLDS.start}-{HOLDS.stop - 1}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    else:
        # normal rule: one sweep across the holding periods
        event = rule["signal"](df)
        print(f"events: {event.sum()}")

        sweep = sweep_hold(df, event, holds=HOLDS, **common)
        # print(sweep.round(4))

        # plot_sweep(sweep, "avg_trade", title=f"{RULE_NAME} ({TICKER}): avg trade")
        plot_sweep(sweep, "sharpe", title=f"{RULE_NAME} ({TICKER}): Sharpe")