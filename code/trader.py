import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

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
    plot=True
):
    # =========================
    # Extract data
    # =========================
    fut_price = analysis_df['PX'][futures]
    fair_price = analysis_df['Fair'][futures]
    T = analysis_df['T']

    # =========================
    # Compute rolling beta
    # =========================
    cov = combined_returns ['EqualWeight'].rolling(beta_window, min_periods=1).cov(combined_returns ['^GSPC'])
    var = combined_returns ['^GSPC'].rolling(beta_window, min_periods=1).var()
    beta = (cov / (var + 1e-12)).shift(1)  # avoid look-ahead

    print(beta)

    # Portfolio NAV (for sizing)
    port_nav = initial_nav * (1 + combined_returns ['EqualWeight']).cumprod()

    # =========================
    # Init
    # =========================
    front_idx = 0
    F = pd.Series(index=fut_price.index, dtype=object)

    positions = pd.DataFrame(0.0, index=fut_price.index, columns=futures)

    current_contracts = 0.0

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

        if front_idx >= len(futures) - 1:
            F.iloc[i] = futures[front_idx]
            continue

        near = futures[front_idx]
        far = futures[front_idx + 1]

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
            window = signal_series.iloc[i - lookback:i]
            z = (diff - window.mean()) / (window.std() + 1e-8)
        else:
            z = np.nan

        rolling_spread.loc[date, "zscore"] = z
        z_lag = rolling_spread["zscore"].shift(1)

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
            Ip = port_nav.loc[date]
            F_price = fut_price.loc[date, near]

            notional = F_price * multiplier

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
    transaction_cost = trades * t_cost

    # =========================
    # FUTURES PnL
    # =========================
    price_diff = fut_price.diff().fillna(0)

    fut_pnl = (
        positions.shift(1).fillna(0) * price_diff
    ).sum(axis=1) * multiplier

    fut_pnl -= transaction_cost
    fut_pnl.iloc[0] = 0

    # =========================
    # PORTFOLIO PnL
    # =========================
    port_pnl = port_nav.diff().fillna(0)

    # =========================
    # COMBINED NAV
    # =========================
    combined_nav = initial_nav + (port_pnl + fut_pnl).cumsum()
    combined_nav_norm = combined_nav / combined_nav.iloc[0]

    # =========================
    # METRICS
    # =========================
    returns = (port_pnl + fut_pnl) / initial_nav

    sharpe = (
        returns.mean() / (returns.std() + 1e-12)
    ) * np.sqrt(252)

    print("\n========== Combined Strategy ==========")
    print(f"Final NAV: {combined_nav.iloc[-1]:,.1f}")
    print(f"Sharpe: {sharpe:.4f}")
    print(f"Total Rolls: {roll_count}")

    # =========================
    # PLOT
    # =========================
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(combined_nav_norm, label='Combined NAV')
        plt.grid(True)
        plt.legend()
        plt.show()

    return {
        "combined_nav": combined_nav,
        "combined_nav_norm": combined_nav_norm,
        "fut_pnl": fut_pnl,
        "port_pnl": port_pnl,
        "positions": positions,
        "contracts": positions.abs().sum(axis=1),
        "beta": beta,
        "front_contract": F,
        "rolling_spread": rolling_spread,
        "sharpe": sharpe
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

res = run_dynamic_roll_strategy_with_static_hedge(
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
    plot=True
)
