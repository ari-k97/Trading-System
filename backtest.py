import numpy as np
import pandas as pd

TRADING_DAYS = 252


def find_trades(event, hold=5):
    """Return (entry_idx, exit_idx) pairs. One position at a time:
    events that fire while you're already in a trade are ignored."""
    sig = event.to_numpy()
    n = len(sig)
    trades = []
    i = 0
    while i < n - 1:
        if sig[i]:
            if i + hold > n - 1:      # not enough data left to finish the trade
                break
            trades.append((i, i + hold))
            i += hold                 # next entry can happen on the exit day
        else:
            i += 1
    return trades


def run_backtest(df, event, hold=5, cost_per_side=0.0005, direction=1, already_own=False):
    """direction = 1 for long, -1 for short.
    already_own=True: you hold the shares already, so only the exit cost is paid."""
    trades = find_trades(event, hold)
    n_sides = 1 if already_own else 2

    position = pd.Series(0.0, index=df.index)
    cost = pd.Series(0.0, index=df.index)
    trade_rows = []

    for entry, exit_ in trades:
        position.iloc[entry + 1 : exit_ + 1] = float(direction)
        if not already_own:
            cost.iloc[entry] += cost_per_side
        cost.iloc[exit_] += cost_per_side

        gross = direction * (df["price"].iloc[exit_] / df["price"].iloc[entry] - 1)
        trade_rows.append({
            "entry_date": df.index[entry],
            "exit_date": df.index[exit_],
            "return": gross - n_sides * cost_per_side,
        })

    daily = df["return_1d"].fillna(0)
    out = pd.DataFrame({
        "position": position,
        "strategy": position * daily - cost,
        "buy_hold": daily,
    })
    return out, pd.DataFrame(trade_rows)


def performance(returns):
    returns = returns.fillna(0)
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    std = returns.std()
    return {
        "total_return": equity.iloc[-1] - 1,
        "cagr": equity.iloc[-1] ** (1 / years) - 1,
        "ann_vol": std * np.sqrt(TRADING_DAYS),
        "sharpe": returns.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
    }


def sweep_hold(df, event, holds=range(1, 61), cost_per_side=0.0005,
               direction=1, already_own=False):
    rows = []
    for h in holds:
        results, trades = run_backtest(df, event, hold=h,
                                       cost_per_side=cost_per_side,
                                       direction=direction,
                                       already_own=already_own)
        if trades.empty:
            continue
        perf = performance(results["strategy"])
        rows.append({
            "hold": h,
            "n_trades": len(trades),
            "hit_rate": (trades["return"] > 0).mean(),
            "avg_trade": trades["return"].mean(),
            "total_return": perf["total_return"],
            "sharpe": perf["sharpe"],
            "max_drawdown": perf["max_drawdown"],
            "time_in_market": results["position"].abs().mean(),
        })
    return pd.DataFrame(rows).set_index("hold")