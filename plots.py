import matplotlib.pyplot as plt

def plot_price(df, title="A200 Adjusted Price"):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["price"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

def plot_sweep(sweep, column, title=None):
    plt.figure(figsize=(12, 5))
    plt.plot(sweep.index, sweep[column], marker="o")
    plt.title(title or f"{column} by holding period")
    plt.xlabel("Holding period (days)")
    plt.ylabel(column)
    plt.grid(True, alpha=0.3)
    plt.show()