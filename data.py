import yfinance as yf
from datetime import date

TICKERS = ["SPY", "IVV.AX", "^GSPC", "A200.AX", "IOZ.AX", "STW.AX"]


def load_prices(tickers=TICKERS, start="2008-01-01", end=None):
    end = end or date.today()
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)
    return data["Adj Close"].copy()


def get_single(prices, ticker="IVV.AX"):
    df = prices[[ticker]].copy()
    df = df.rename(columns={ticker: "price"})
    return df.dropna()