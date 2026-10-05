import numpy as np
import pandas as pd
from scipy import stats

from backtest import run_backtest, performance, TRADING_DAYS


def _baseline_event(df):
    """Fires every day: the same trade as the rule, but with no signal."""
    return df["return_1d"].notna()


def summary_vs_baseline(df, event, hold=5, cost=0.0005, direction=1, already_own=False):
    res, trades = run_backtest(df, event, hold=hold, cost_per_side=cost,
                               direction=direction, already_own=already_own)
    base, base_trades = run_backtest(df, _baseline_event(df), hold=hold, cost_per_side=cost,
                                     direction=direction, already_own=already_own)

    table = pd.DataFrame({
        "rule": performance(res["strategy"]),
        "baseline": performance(base["strategy"]),
        "buy_hold": performance(res["buy_hold"]),
    })
    extra = pd.DataFrame({
        "rule": {"n_trades": len(trades), "time_in_market": res["position"].abs().mean()},
        "baseline": {"n_trades": len(base_trades), "time_in_market": base["position"].abs().mean()},
        "buy_hold": {"n_trades": 1, "time_in_market": 1.0},
    })
    return pd.concat([table, extra])


def trade_stats(df, event, hold=5, cost=0.0005, direction=1, already_own=False):
    _, trades = run_backtest(df, event, hold=hold, cost_per_side=cost,
                             direction=direction, already_own=already_own)
    _, base_trades = run_backtest(df, _baseline_event(df), hold=hold, cost_per_side=cost,
                                  direction=direction, already_own=already_own)

    def stats_for(t):
        if t.empty:
            return {"n_trades": 0}
        r = t["return"]
        wins, losses = r[r > 0], r[r <= 0]
        gross_loss = -losses.sum()
        return {
            "n_trades": len(r),
            "win_rate": (r > 0).mean(),
            "avg_return": r.mean(),
            "median_return": r.median(),
            "avg_win": wins.mean() if len(wins) else np.nan,
            "avg_loss": losses.mean() if len(losses) else np.nan,
            "profit_factor": wins.sum() / gross_loss if gross_loss > 0 else np.nan,
            "best_trade": r.max(),
            "worst_trade": r.min(),
        }

    out = pd.DataFrame({"rule": stats_for(trades), "baseline": stats_for(base_trades)})
    if "avg_return" in out.index:
        out.loc["excess_avg_return", "rule"] = (
            out.loc["avg_return", "rule"] - out.loc["avg_return", "baseline"]
        )
    return out


def significance(df, event, hold=5, cost=0.0005, direction=1, already_own=False):
    _, trades = run_backtest(df, event, hold=hold, cost_per_side=cost,
                             direction=direction, already_own=already_own)
    _, base_trades = run_backtest(df, _baseline_event(df), hold=hold, cost_per_side=cost,
                                  direction=direction, already_own=already_own)
    if len(trades) < 5 or len(base_trades) < 5:
        return pd.Series({"note": "too few trades to test"})

    r = trades["return"].to_numpy()
    b = base_trades["return"].to_numpy()

    _, p_zero = stats.ttest_1samp(r, 0.0)
    _, p_base = stats.ttest_ind(r, b, equal_var=False)

    rng = np.random.default_rng(0)
    boots = rng.choice(r, size=(5000, len(r)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return pd.Series({
        "n_trades": len(r),
        "mean_trade_return": r.mean(),
        "p_value_vs_zero": p_zero,
        "mean_vs_baseline": r.mean() - b.mean(),
        "p_value_vs_baseline": p_base,
        "bootstrap_95pct_low": lo,
        "bootstrap_95pct_high": hi,
    })


def yearly_breakdown(df, event, hold=5, cost=0.0005, direction=1, already_own=False):
    res, trades = run_backtest(df, event, hold=hold, cost_per_side=cost,
                               direction=direction, already_own=already_own)
    year = res.index.year
    yearly = pd.DataFrame({
        "rule": (1 + res["strategy"]).groupby(year).prod() - 1,
        "buy_hold": (1 + res["buy_hold"]).groupby(year).prod() - 1,
    })
    yearly["rule_minus_bh"] = yearly["rule"] - yearly["buy_hold"]
    if trades.empty:
        yearly["n_trades"] = 0
    else:
        counts = trades.groupby(trades["entry_date"].dt.year).size()
        yearly["n_trades"] = counts.reindex(yearly.index).fillna(0).astype(int)
    yearly.index.name = "year"
    return yearly


def split_test(df, event, hold=5, cost=0.0005, direction=1, already_own=False, split=0.7):
    k = int(len(df) * split)
    cols = {}
    for part, sl in [("first", slice(0, k)), ("last", slice(k, None))]:
        d, e = df.iloc[sl], event.iloc[sl]
        res, trades = run_backtest(d, e, hold=hold, cost_per_side=cost,
                                   direction=direction, already_own=already_own)
        perf = performance(res["strategy"])
        label = f"{part} {d.index[0].date()} to {d.index[-1].date()}"
        cols[label] = {
            "n_trades": len(trades),
            "avg_trade": trades["return"].mean() if len(trades) else np.nan,
            "sharpe": perf["sharpe"],
            "total_return": perf["total_return"],
            "buy_hold_sharpe": performance(res["buy_hold"])["sharpe"],
        }
    return pd.DataFrame(cols)


def risk_metrics(df, event, hold=5, cost=0.0005, direction=1, already_own=False):
    res, _ = run_backtest(df, event, hold=hold, cost_per_side=cost,
                          direction=direction, already_own=already_own)
    out = {}
    for name, col in [("rule", "strategy"), ("buy_hold", "buy_hold")]:
        r = res[col].fillna(0)
        perf = performance(r)
        downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
        out[name] = {
            "sortino": r.mean() / downside * np.sqrt(TRADING_DAYS) if downside > 0 else np.nan,
            "calmar": perf["cagr"] / abs(perf["max_drawdown"]) if perf["max_drawdown"] < 0 else np.nan,
            "max_drawdown": perf["max_drawdown"],
            "worst_day": r.min(),
            "daily_5pct_quantile": r.quantile(0.05),
        }
    return pd.DataFrame(out)


# the names you use in main.py
EVALUATIONS = {
    "summary": summary_vs_baseline,
    "trades": trade_stats,
    "significance": significance,
    "yearly": yearly_breakdown,
    "split": split_test,
    "risk": risk_metrics,
}


def evaluate(names, df, event, **kwargs):
    """Run the named evaluations in order and print each result."""
    for name in names:
        if name not in EVALUATIONS:
            raise KeyError(f"Unknown evaluation '{name}'. Available: {list(EVALUATIONS)}")
    for name in names:
        print(f"\n===== {name} =====")
        print(EVALUATIONS[name](df, event, **kwargs).round(4))