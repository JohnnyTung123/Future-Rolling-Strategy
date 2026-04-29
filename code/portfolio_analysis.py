import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format

# ---------------------------
# Configuration
# ---------------------------
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "JNJ"
]
MARKET_TICKER = "^GSPC"  # S&P 500 Index
START_DATE = "2020-09-20"
END_DATE = "2026-03-21"
ROLLING_WINDOW = 60
INITIAL_CAPITAL = 100_000_000


# ---------------------------
# Data Download
# ---------------------------
def download_prices(tickers, start, end, save_path="../data/processed/long_port_px.csv"):
    """Download adjusted close prices from Yahoo Finance and save to CSV."""
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    data = data.dropna(how="all")
    data.to_csv(save_path)
    return data


def load_prices(file_path="../data/process/long_port_px.csv"):
    """Load saved price data from CSV and clean it."""
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    data = data.sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(how="all")
    return data

# ---------------------------
# Portfolio Construction
# ---------------------------
def compute_returns(prices):
    """Compute daily simple returns."""
    return prices.pct_change().dropna()


def compute_rolling_beta(portfolio_returns, market_returns, window):
    """Compute rolling CAPM beta of the portfolio versus the market."""
    covariance = portfolio_returns.rolling(window).cov(market_returns)
    market_variance = market_returns.rolling(window).var()
    rolling_beta = covariance / market_variance
    rolling_beta.to_csv("../data/processed/rolling_beta.csv")
    return rolling_beta


def compute_equal_weights(index, columns):
    """Construct an equal-weight portfolio."""
    weights = pd.DataFrame(
        1 / len(columns),
        index=index,
        columns=columns,
    )
    return weights


def backtest_portfolio(asset_returns, weights, initial_capital):
    """Backtest portfolio using one-day lagged weights."""
    aligned_returns = asset_returns.loc[weights.index]
    portfolio_returns = (weights.shift(1) * aligned_returns).sum(axis=1)

    portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
    return portfolio_returns, portfolio_value


def benchmark_market(market_returns, initial_capital):
    """Compute benchmark market portfolio value."""
    market_returns.iloc[0] = 0
    return initial_capital * (1 + market_returns).cumprod()


def create_nav_dataframe(portfolio_value, market_value):
    """Combine daily NAV series for portfolio and benchmark."""
    nav = pd.DataFrame({
        "EqualWeight_NAV": portfolio_value,
        "Market_NAV": market_value,
    })
    return nav.dropna()


def create_combined_returns(portfolio_returns, market_returns):
    """Combine portfolio and benchmark returns into one DataFrame."""
    combined = pd.DataFrame({
        "EqualWeight": portfolio_returns,
        "^GSPC": market_returns,
    })
    # combined.round(5).to_csv("../data/processed/combined_returns.csv")
    return combined.dropna()


def performance_summary(returns, annualization_factor=252):
    """Compute annualized return, volatility, and Sharpe ratio."""
    ann_return = returns.mean() * annualization_factor
    ann_vol = returns.std() * np.sqrt(annualization_factor)
    sharpe = ann_return / ann_vol

    return pd.DataFrame({
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
    })


# ---------------------------
# Visualization
# ---------------------------
def plot_performance(portfolio_value, market_value):
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_value.index, portfolio_value, label="Equal-Weighted Portfolio")
    plt.plot(market_value.index, market_value, label="S&P 500 Benchmark", linestyle="--")

    plt.title(f"Growth of ${INITIAL_CAPITAL:,.0f} Investment")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main
# ---------------------------
def main():
    # print("Downloading price data...")
    #
    stock_prices = download_prices(TICKERS, START_DATE, END_DATE, "../data/processed/long_port_px.csv")
    market_prices = download_prices(MARKET_TICKER, START_DATE, END_DATE, "../data/processed/s&p500_px.csv")

    # stock_prices = load_prices(file_path="../data/processed/long_port_px.csv")
    # market_prices = load_prices(file_path="../data/processed/s&p500_px.csv")

    stock_returns = compute_returns(stock_prices)
    market_returns = compute_returns(market_prices.squeeze())

    common_index = stock_returns.index.intersection(market_returns.index)
    stock_returns = stock_returns.loc[common_index]
    market_returns = market_returns.loc[common_index]

    portfolio_weights = compute_equal_weights(
        index=stock_returns.index,
        columns=stock_returns.columns,
    )

    portfolio_returns, portfolio_value = backtest_portfolio(
        asset_returns=stock_returns,
        weights=portfolio_weights,
        initial_capital=INITIAL_CAPITAL,
    )

    rolling_beta = compute_rolling_beta(
        portfolio_returns=portfolio_returns,
        market_returns=market_returns.loc[portfolio_returns.index],
        window=ROLLING_WINDOW,
    )

    market_value = benchmark_market(
        market_returns.loc[portfolio_returns.index],
        INITIAL_CAPITAL,
    )

    combined_returns = create_combined_returns(
        portfolio_returns=portfolio_returns,
        market_returns=market_returns.loc[portfolio_returns.index],
    )

    nav_df = create_nav_dataframe(
        portfolio_value=portfolio_value,
        market_value=market_value,
    )

    plot_performance(portfolio_value, market_value)

    print("\nFinal Portfolio Value:")
    print(f"Low-Beta Portfolio: ${portfolio_value.iloc[-1]:,.2f}")
    print(f"S&P 500 Benchmark:  ${market_value.iloc[-1]:,.2f}")

    print("Latest Rolling Portfolio Beta:")
    print(f"{rolling_beta.iloc[-1]:.4f}")

    print("Latest Portfolio Weights:")
    print(portfolio_weights)

    print("Combined Returns:")
    print(combined_returns.tail())

    print("Performance Summary:")
    print(performance_summary(combined_returns).round(4))

    print("Daily NAV:")
    print(nav_df)

if __name__ == "__main__":
    main()
