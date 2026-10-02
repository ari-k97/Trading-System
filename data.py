import pandas as pd
import yfinance as yf
from datetime import date


def load_prices(tickers, start="2008-01-01", end=None):
    if isinstance(tickers, str):
        tickers = [tickers]
    end = end or date.today()

    data = yf.download(tickers, start=start, end=end, auto_adjust=False)
    prices = data["Adj Close"].copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return prices


def get_single(prices, ticker):
    df = prices[[ticker]].copy()
    df = df.rename(columns={ticker: "price"})
    return df.dropna()