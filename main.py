import matplotlib.pyplot as plt
import pandas as pd

from backtest import sweep_hold
from evaluation import evaluate
from pipeline import load_ticker, make_event, compare_tickers
from plots import plot_sweep
from rules import get_rule

# ---------- settings: change these, then run ----------
RULE_NAME = "long_after_drop_1pct"
TICKER = "IVV.AX"            # used by "sweep" and "evaluate"
COST = 0.0005

START = None                 # e.g. "2008-01-01". None = start of the data
END = None                   # e.g. "2018-12-31". None = up to today
DELAY = 0                    # 0 = trade at the signal day's close, 1 = at the next day's close

ACTION = "evaluate"          # "sweep", "evaluate" or "multi"

# "sweep": all holding periods
HOLDS = range(1, 31)
LOOKBACKS = range(1, 21)     # only for the "n_day" rules

# "evaluate" and "multi": one holding period
EVALUATIONS = ["summary", "trades", "significance"]   # any of: summary, trades, significance, yearly, split, risk
EVAL_HOLD = 5
EVAL_LOOKBACK = 3            # only for the "n_day" rules

# "multi": tickers to compare
TICKERS = ["IVV.AX", "A200.AX", "SPY", "VAS.AX"]
# ------------------------------------------------------

if __name__ == "__main__":
    if ACTION not in ("sweep", "evaluate", "multi"):
        raise ValueError("ACTION must be 'sweep', 'evaluate' or 'multi'")

    rule = get_rule(RULE_NAME)
    print(f"Rule: {RULE_NAME}\n{rule['description']}")
    print(f"Period: {START or 'start of data'} to {END or 'today'} | "
          f"delay: {DELAY} day(s) | cost: {COST:.2%} per side\n")

    common = dict(cost_per_side=COST,
                  direction=rule["direction"],
                  already_own=rule["already_own"],
                  delay=DELAY)

    if ACTION == "multi":
        table = compare_tickers(rule, TICKERS, lookback=EVAL_LOOKBACK, hold=EVAL_HOLD,
                                cost=COST, start=START, end=END, delay=DELAY)
        if table.empty:
            print("No results.")
        else:
            print(table.round(4))
            ok = table.dropna(subset=["excess_avg_trade"])
            print(f"\nBeat baseline on average trade: {(ok['excess_avg_trade'] > 0).sum()} of {len(ok)} tickers")
            print(f"Beat baseline on Sharpe:        {(ok['excess_sharpe'] > 0).sum()} of {len(ok)} tickers")

    else:
        full = load_ticker(TICKER)
        df = full.loc[START:END]
        print(f"{TICKER}: {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}\n")

        if ACTION == "evaluate":
            event = make_event(rule, full, df, EVAL_LOOKBACK)
            print(f"events: {event.sum()}")
            evaluate(EVALUATIONS, df, event, hold=EVAL_HOLD, cost=COST,
                     direction=rule["direction"], already_own=rule["already_own"],
                     delay=DELAY)

        elif "signal_n" in rule:
            rows = []
            for n in LOOKBACKS:
                event = make_event(rule, full, df, n)
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
            event = make_event(rule, full, df)
            print(f"events: {event.sum()}")

            sweep = sweep_hold(df, event, holds=HOLDS, **common)
            print(sweep.round(4))

            plot_sweep(sweep, "avg_trade", title=f"{RULE_NAME} ({TICKER}): avg trade")
            plot_sweep(sweep, "sharpe", title=f"{RULE_NAME} ({TICKER}): Sharpe")