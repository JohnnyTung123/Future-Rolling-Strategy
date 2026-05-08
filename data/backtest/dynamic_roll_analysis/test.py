def compute_performance_metrics(
    combined_nav,
    returns,
    transaction_cost,
    trading_days=252
):
    """
    Parameters
    ----------
    combined_nav : pd.Series
        Strategy NAV series

    returns : pd.Series
        Daily strategy returns

    transaction_cost : pd.Series
        Daily transaction costs

    trading_days : int
        Annualization factor
    """

    # =========================
    # Basic stats
    # =========================
    final_nav = combined_nav.iloc[-1]

    total_return = (
        combined_nav.iloc[-1] / combined_nav.iloc[0]
    ) - 1

    n_years = len(returns) / trading_days

    annualized_return = (
        (1 + total_return) ** (1 / n_years)
    ) - 1

    annualized_volatility = (
        returns.std() * np.sqrt(trading_days)
    )

    sharpe_ratio = (
        returns.mean() / (returns.std() + 1e-12)
    ) * np.sqrt(trading_days)

    # =========================
    # Drawdown
    # =========================
    rolling_max = combined_nav.cummax()

    drawdown = (
        combined_nav - rolling_max
    ) / rolling_max

    max_drawdown = drawdown.min()

    # =========================
    # Daily return stats
    # =========================
    max_return = returns.max()

    min_return = returns.min()

    win_rate = (
        (returns > 0).sum() / len(returns)
    )

    # =========================
    # Higher moments
    # =========================
    skewness = skew(returns.dropna())

    kurt = kurtosis(returns.dropna())

    # =========================
    # Calmar Ratio
    # =========================
    calmar_ratio = (
        annualized_return / abs(max_drawdown + 1e-12)
    )

    # =========================
    # Transaction cost
    # =========================
    total_transaction_cost = transaction_cost.sum()

    # =========================
    # Output
    # =========================
    metrics = pd.Series({
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Calmar Ratio": calmar_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Max Daily Return": max_return,
        "Min Daily Return": min_return,
        "Skewness": skewness,
        "Kurtosis": kurt,
        "Final NAV": final_nav,
        "Total Transaction Cost": total_transaction_cost
    })

    return metrics