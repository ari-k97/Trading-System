import pandas as pd

from data import load_prices, get_single
from features import add_features
from backtest import run_backtest, performance, sweep_hold
from plots import plot_sweep

if __name__ == "__main__":
    # 1. load data and build features
    prices = load_prices()
    df = get_single(prices, "IVV.AX")
    print(df.head(10))        # raw data, before features are added
    print(df.tail(3))         # most recent dates, to check the data is current
    df = add_features(df)

    # Test 1: run a backtest with a variable-day holding period
    event = df["return_1d"] < -0.01
    print(f"events: {event.sum()}")

    sweep = sweep_hold(df, event, holds=range(1, 100), cost_per_side=0.0005)
    print(sweep.round(4))

    plot_sweep(sweep, "avg_trade")
    plot_sweep(sweep, "sharpe")