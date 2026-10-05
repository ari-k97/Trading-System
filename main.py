import matplotlib.pyplot as plt
import pandas as pd

from data import load_prices, get_single
from features import add_features
from backtest import sweep_hold
from plots import plot_sweep
from rules import get_rule
from evaluation import evaluate

# ---------- settings: change these, then run ----------
RULE_NAME = "long_after_drop_1pct"
TICKER = "IVV.AX"
COST = 0.0005

ACTION = "evaluate"          # "sweep" (all holding periods) or "evaluate" (one hold, detailed)

# used when ACTION = "sweep"
HOLDS = range(1, 31)
LOOKBACKS = range(1, 21)     # only for the "n_day" rules

# used when ACTION = "evaluate"
EVALUATIONS = ["summary", "trades", "significance"]   # any of: summary, trades, significance, yearly, split, risk
EVAL_HOLD = 5
EVAL_LOOKBACK = 3            # only for the "n_day" rules
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

    if ACTION == "evaluate":
        if "signal_n" in rule:
            event = rule["signal_n"](df, EVAL_LOOKBACK)
        else:
            event = rule["signal"](df)
        print(f"events: {event.sum()}")

        evaluate(EVALUATIONS, df, event, hold=EVAL_HOLD, cost=COST,
                 direction=rule["direction"], already_own=rule["already_own"])

    elif "signal_n" in rule:
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
        event = rule["signal"](df)
        print(f"events: {event.sum()}")

        sweep = sweep_hold(df, event, holds=HOLDS, **common)
        print(sweep.round(4))

        plot_sweep(sweep, "avg_trade", title=f"{RULE_NAME} ({TICKER}): avg trade")
        plot_sweep(sweep, "sharpe", title=f"{RULE_NAME} ({TICKER}): Sharpe")