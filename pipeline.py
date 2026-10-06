import numpy as np
import pandas as pd

from backtest import run_backtest, performance
from data import load_prices, get_single
from features import add_features


def load_ticker(ticker, data_start="2000-01-01"):
    """Download the full history and add all feature columns."""
    prices = load_prices(ticker, start=data_start)
    df = get_single(prices, ticker)
    return add_features(df)


def make_event(rule, full, period, lookback=None):
    """Compute the signal on the full history (so rolling windows are warmed up),
    then keep only the days in `period`."""
    if "signal_n" in rule:
        event = rule["signal_n"](full, lookback)
    else:
        event = rule["signal"](full)
    return event.loc[period.index]


COLS = ["start", "n_trades", "avg_trade", "base_avg_trade", "excess_avg_trade",
        "sharpe", "base_sharpe", "excess_sharpe", "time_in_market"]


def compare_tickers(rule, tickers, lookback=None, hold=5, cost=0.0005,
                    start=None, end=None, delay=0):
    """Run one rule on many tickers at one holding period, next to a no-signal baseline."""
    rows = []
    for t in tickers:
        try:
            full = load_ticker(t)
            df = full.loc[start:end]
            if df.empty:
                raise ValueError("no data in the chosen period")
            event = make_event(rule, full, df, lookback)
            kw = dict(hold=hold, cost_per_side=cost, direction=rule["direction"],
                      already_own=rule["already_own"], delay=delay)
            res, trades = run_backtest(df, event, **kw)
            base, base_trades = run_backtest(df, df["return_1d"].notna(), **kw)
        except Exception as err:
            print(f"skipping {t}: {err}")
            continue

        row = dict.fromkeys(COLS, np.nan)
        row.update({"ticker": t, "start": str(df.index[0].date()), "n_trades": len(trades)})
        if len(trades) and len(base_trades):
            perf, bperf = performance(res["strategy"]), performance(base["strategy"])
            avg, bavg = trades["return"].mean(), base_trades["return"].mean()
            row.update({
                "avg_trade": avg,
                "base_avg_trade": bavg,
                "excess_avg_trade": avg - bavg,
                "sharpe": perf["sharpe"],
                "base_sharpe": bperf["sharpe"],
                "excess_sharpe": perf["sharpe"] - bperf["sharpe"],
                "time_in_market": res["position"].abs().mean(),
            })
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("ticker")[COLS]