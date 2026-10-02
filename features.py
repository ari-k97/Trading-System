def add_features(df):
    df = df.copy()  # don't mutate the caller's DataFrame

    # past returns
    for n in [1, 3, 5, 10, 20, 60]:
        df[f"return_{n}d"] = df["price"].pct_change(n)

    # future returns
    for n in [1, 3, 5, 10, 20]:
        df[f"future_{n}d"] = df["price"].shift(-n) / df["price"] - 1

    # moving averages
    for n in [5, 20, 50, 100, 200]:
        df[f"ma_{n}"] = df["price"].rolling(n).mean()

    # distance from moving averages
    for n in [20, 50, 200]:
        df[f"distance_{n}ma"] = df["price"] / df[f"ma_{n}"] - 1

    # volatility
    for n in [5, 20, 60]:
        df[f"vol_{n}"] = df["return_1d"].rolling(n).std()

    return df