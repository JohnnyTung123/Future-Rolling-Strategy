import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import os

pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_columns', None)

# =========================
# LOAD DATA
# =========================
analysis_df = pd.read_csv(
    '../data/processed/analysis_multi.csv',
    header=[0, 1],
    index_col=0
)

analysis_df.index = pd.to_datetime(analysis_df.index)

combined_returns = pd.read_csv(
    '../data/processed/combined_returns.csv',
    index_col=0,
)

combined_returns.index = pd.to_datetime(combined_returns.index)

# =========================
# Performance Metrics
# =========================
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
    skewness = returns.skew()

    kurt = returns.kurtosis()

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

def run_dynamic_roll_strategy_with_static_hedge(
    analysis_df,
    combined_returns,   # <-- your returns dataframe (^GSPC, EqualWeight)
    futures,
    initial_nav=100_000_000,
    multiplier=50,
    t_cost=2.5,
    lookback=20,
    z_threshold=2.0,
    beta_window=60,
    hedge_ratio=1.0,
    strategy_name="strategy",
    plot=False,
):
    # =========================
    # CREATE OUTPUT FOLDER
    # =========================
    output_dir = f'../data/backtest/strats_log/{strategy_name}'
    os.makedirs(output_dir, exist_ok=True)

    start_date = analysis_df.index[0]
    end_date = analysis_df.index[-1]

    initial_cost = 0.002 * 1 * initial_nav# assume 20 bps

    # =========================
    # Extract data
    # =========================
    fut_price = analysis_df['PX'][futures]
    fair_price = analysis_df['Fair'][futures]
    T = analysis_df['T']

    # =========================
    # Compute rolling beta
    # =========================
    cov = combined_returns ['EqualWeight'].rolling(beta_window).cov(combined_returns ['^GSPC'])
    var = combined_returns ['^GSPC'].rolling(beta_window).var()
    beta = (cov / (var + 1e-12)).shift(1)  # avoid look-ahead
    beta = beta.loc[start_date:end_date]

    combined_returns = combined_returns.loc[start_date:end_date]
    combined_returns.iloc[0] = 0 # return accrue after the start day

    # Portfolio NAV (for sizing)
    port_nav = initial_nav * (1 + combined_returns ['EqualWeight']).cumprod()
    lagged_port_nav = port_nav.shift(1)
    # =========================
    # Init
    # =========================
    front_idx = 0
    F = pd.Series(index=fut_price.index, dtype=object)
    positions = pd.DataFrame(0.0, index=fut_price.index, columns=futures)

    current_contracts = 0.0
    hedge_initialized = False

    signal_series = pd.Series(index=fut_price.index, dtype=float)

    rolling_spread = pd.DataFrame(
        index=fut_price.index,
        columns=["near", "far", "fair", "market", "difference", "zscore"]
    )

    total_market_roll = 0.0
    total_fair_roll = 0.0
    roll_count = 0

    # =========================
    # MAIN LOOP
    # =========================
    for i, date in enumerate(fut_price.index):

        if front_idx >= len(futures) - 1: # final contract
            F.iloc[i] = futures[front_idx]
            positions.loc[date, F.iloc[i]] = -current_contracts
            continue

        near = futures[front_idx]
        far = futures[front_idx + 1]

        # =========================
        # Initial hedge setup
        # =========================
        if (not hedge_initialized) and (not pd.isna(beta.loc[date])):
            beta_t = beta.loc[date] # beta is already lagged
            Ip = initial_nav # initial portfolio value: 100M USD
            F_price = fut_price.loc[date, near] # no need lagged right? buy today -> realize the return on tomorrow
            notional = F_price * multiplier
            current_contracts = (
                    hedge_ratio * beta_t * Ip / (notional + 1e-12)
            )
            hedge_initialized = True

        # =========================
        # Spread calc
        # =========================
        fair = fair_price.loc[date, far] - fair_price.loc[date, near]
        market = fut_price.loc[date, far] - fut_price.loc[date, near]
        diff = market - fair

        signal_series.loc[date] = diff

        rolling_spread.loc[date, ["near", "far"]] = [near, far]
        rolling_spread.loc[date, "fair"] = fair
        rolling_spread.loc[date, "market"] = market
        rolling_spread.loc[date, "difference"] = diff

        # =========================
        # Z-score
        # =========================
        if i >= lookback:
            window = signal_series.iloc[i - lookback:i] # exclude today's diff
            z = (diff - window.mean()) / (window.std() + 1e-8)
        else:
            z = np.nan

        rolling_spread.loc[date, "zscore"] = z
        z_lag = rolling_spread["zscore"].shift(1) # using yesterday z-score

        T_now = T.loc[date, near]

        rolled = False

        # =========================
        # SIGNAL ROLL
        # =========================
        if (T_now <= 14/365) and (not pd.isna(z)):
            if z_lag.iloc[i] > z_threshold:
                rolled = True

        # =========================
        # FORCED ROLL
        # =========================
        if (not rolled) and (T_now <= 3/365):
            rolled = True

        # =========================
        # IF ROLL → UPDATE CONTRACTS
        # =========================
        if rolled:
            # close old position implicitly via diff()

            # compute hedge size at roll
            beta_t = beta.loc[date]
            Ip = lagged_port_nav.loc[date] # yesterday portfolio value
            if pd.isna(Ip):
                Ip = initial_nav # 2021-09-20: 100M USD
            F_price = fut_price.loc[date, near] # should I lag?

            notional = F_price * multiplier # multiplier: points PnL -> dollar PnL

            if not np.isnan(beta_t):
                current_contracts = hedge_ratio * beta_t * Ip / (notional + 1e-12)
            else:
                current_contracts = 0.0

            # move to next contract
            front_idx += 1

            total_market_roll += market * current_contracts
            total_fair_roll += fair * current_contracts
            roll_count += 1

        # =========================
        # RECORD FRONT
        # =========================
        F.iloc[i] = futures[front_idx]

        # =========================
        # APPLY STATIC HEDGE
        # =========================
        positions.loc[date, F.loc[date]] = -current_contracts

    # =========================
    # TRANSACTION COST
    # =========================
    trades = positions.diff().abs().sum(axis=1).fillna(0)
    trades.iloc[0] = positions.iloc[0].abs().sum()
    transaction_cost = trades * t_cost
    # not the best way but I have to account for the initial t-cost of long-portfolio
    transaction_cost.iloc[0] += initial_cost
    # trades[trades != 0].to_csv('../data/backtest/strats_log/rolling_history.csv')

    # =========================
    # FUTURES PnL
    # =========================
    price_diff = fut_price.diff().fillna(0)

    fut_pnl = (
        positions.shift(1).fillna(0) * price_diff
    ).sum(axis=1) * multiplier # multiplier: points PnL -> dollar PnL

    fut_net_pnl = fut_pnl - transaction_cost

    # =========================
    # PORTFOLIO PnL
    # =========================
    port_pnl = port_nav.diff().fillna(0)

    # =========================
    # COMBINED NAV
    # =========================
    combined_nav = initial_nav + (port_pnl + fut_net_pnl).cumsum()
    combined_nav_norm = combined_nav / combined_nav.iloc[0]

    # =========================
    # METRICS
    # =========================
    # returns = (port_pnl + fut_net_pnl) / initial_nav
    returns = combined_nav.pct_change().fillna(0)

    metrics = compute_performance_metrics(
        combined_nav=combined_nav,
        returns=returns,
        transaction_cost=transaction_cost
    )

    # =========================
    # Save Logs
    # =========================
    positions.to_csv(f'{output_dir}/futures_positions.csv')

    fut_pnl.to_csv(
        f'{output_dir}/futures_pnl.csv',
        header=['fut_pnl']
    )

    port_pnl.to_csv(
        f'{output_dir}/portfolio_pnl.csv',
        header=['port_pnl']
    )

    combined_nav.to_csv(
        f'{output_dir}/daily_nav.csv',
        header=['nav']
    )

    rolling_spread.to_csv(
        f'{output_dir}/rolling_spread.csv'
    )

    transaction_cost.to_csv(
        f'{output_dir}/transaction_cost.csv'
    )

    metrics.to_csv(
        f'{output_dir}/performance_metrics.csv',
        header=['value']
    )

    # =========================
    # Plot
    # =========================
    if plot:
        plt.figure(figsize=(10, 5))

        plt.plot(
            combined_nav_norm,
            label=strategy_name,
            linewidth=2
        )

        plt.title(f'{strategy_name} NAV')
        plt.xlabel('Date')
        plt.ylabel('Normalized NAV')

        plt.grid(True)
        plt.legend()

        # save figure
        plt.savefig(
            f'{output_dir}/nav_plot.png',
            dpi=300,
            bbox_inches='tight'
        )

        plt.show()
        plt.close()

    # display
    # print("\n========== PERFORMANCE METRICS ==========")
    # print(metrics.apply(lambda x: f"{x:.4f}"))

    return {
        "metrics": metrics,
        "combined_nav": combined_nav,
        "combined_nav_norm": combined_nav_norm
    }


# =========================
# RUN BACKTEST
# =========================
unique_cols = analysis_df.columns.get_level_values(1).unique()

exclude = ['ESH1', 'ESM1', 'ESU1']

futures = [
    c for c in unique_cols
    if re.fullmatch(r'ES[HMUZ]\d+', c)
    and c not in exclude
]


# =========================
# RUN MULTIPLE STRATEGIES
# =========================
start_date = analysis_df.index[0]
end_date = analysis_df.index[-1]

print("\n========== Backtesting Period ==========")
print(
    f"Start invest on {start_date} "
    f"with initial NAV of ${100_000_000:,.0f}"
)
print(f"End date: {end_date}")

strategy_configs = {
    "no_hedge": 0.0,
    "quarter_hedge": 0.25,
    "half_hedge": 0.5,
    "full_hedge": 1.0
}

all_metrics = []

all_navs = pd.DataFrame()

for strategy_name, hedge_ratio in strategy_configs.items():

    # print("\n")
    # print("=" * 50)
    # print(f"Running: {strategy_name}")
    # print("=" * 50)

    res = run_dynamic_roll_strategy_with_static_hedge(
        analysis_df=analysis_df,
        combined_returns=combined_returns,
        futures=futures,
        initial_nav=100_000_000,
        multiplier=50,
        t_cost=2.5,
        lookback=20,
        z_threshold=2.0,
        beta_window=60,
        hedge_ratio=hedge_ratio,
        strategy_name=strategy_name,
        plot=False,
    )

    metrics = res["metrics"]
    nav = res["combined_nav_norm"]

    metrics.name = strategy_name

    all_metrics.append(metrics)

    all_navs[strategy_name] = nav

# =========================
# COMBINE RESULTS
# =========================
results_df = pd.concat(
    all_metrics,
    axis=1
).T

print("\n========== ALL STRATEGY RESULTS ==========")
print(results_df)

# save
results_df.round(4).to_csv(
    "../data/backtest/all_strategy_metrics.csv"
)

# =========================
# COMPARISON PLOT
# =========================
plt.figure(figsize=(12, 6))

for col in all_navs.columns:
    plt.plot(all_navs.index, all_navs[col], label=col)

plt.title('Strategy Comparison')
plt.xlabel('Date')
plt.ylabel('Normalized NAV')

plt.grid(True)
plt.legend()

plt.savefig(
    "../data/backtest/all_strategy_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =========================
# DRAWDOWN COMPARISON PLOT
# =========================
drawdowns = pd.DataFrame(index=all_navs.index)

for col in all_navs.columns:

    rolling_max = all_navs[col].cummax()

    drawdown = (
        all_navs[col] - rolling_max
    ) / rolling_max

    drawdowns[col] = drawdown

# plot
plt.figure(figsize=(12, 6))

for col in drawdowns.columns:
    plt.plot(
        drawdowns.index,
        drawdowns[col],
        label=col
    )

plt.title('Strategy Drawdown Comparison')
plt.xlabel('Date')
plt.ylabel('Drawdown')

plt.grid(True)
plt.legend()

plt.savefig(
    "../data/backtest/all_strategy_drawdown.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()
