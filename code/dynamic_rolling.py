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


# =========================
# BACKTEST FUNCTION
# =========================
def run_dynamic_roll_strategy(
    analysis_df,
    futures,
    initial_nav=100_000_000,
    multiplier=50,
    t_cost=2.5,
    lookback=20,
    z_threshold=2.0,
    # direction=-1,
    # hedge_contracts=1,
    plot=True
):
    total_market_roll = 0.0
    total_fair_roll = 0.0
    roll_count = 0
    signal_roll_count = 0
    forced_roll_count = 0

    # =========================
    # Extract data
    # =========================
    fut_price = analysis_df['PX'][futures]
    fair_price = analysis_df['Fair'][futures]
    T = analysis_df['T']

    # =========================
    # 1. Rolling + Signal Loop
    # =========================
    front_idx = 0

    F = pd.Series(index=fut_price.index, dtype=object)

    rolling_spread = pd.DataFrame(
        index=fut_price.index,
        columns=["near", "far", "fair", "market", "difference", "zscore"]
    )

    signal_series = pd.Series(index=fut_price.index, dtype=float)

    for i, date in enumerate(fut_price.index):

        # -------------------------
        # If last contract
        # -------------------------
        if front_idx >= len(futures) - 1:
            F.iloc[i] = futures[front_idx]
            continue

        near = futures[front_idx]
        far = futures[front_idx + 1]

        # -------------------------
        # Compute spreads
        # -------------------------
        fair = fair_price.loc[date, far] - fair_price.loc[date, near]
        market = fut_price.loc[date, far] - fut_price.loc[date, near]
        diff = market - fair

        rolling_spread.loc[date, ["near", "far"]] = [near, far]
        rolling_spread.loc[date, "fair"] = fair
        rolling_spread.loc[date, "market"] = market
        rolling_spread.loc[date, "difference"] = diff

        signal_series.loc[date] = diff

        # -------------------------
        # Z-score
        # -------------------------
        if i >= lookback:
            window = signal_series.iloc[i - lookback:i]

            mean = window.mean()
            std = window.std()

            z = (diff - mean) / (std + 1e-8)
        else:
            z = np.nan

        rolling_spread.loc[date, "zscore"] = z

        # -------------------------
        # Rolling decision
        # -------------------------
        T_now = T.loc[date, near]

        rolled = False
        roll_type = None

        # SIGNAL ROLL (T-10)
        if (T_now <= 14/365) and (not pd.isna(z)):
            if z > z_threshold:
                front_idx += 1
                rolled = True
                roll_type = "SIGNAL"
                signal_roll_count += 1

        # FORCED ROLL (T-3)
        if (not rolled) and (T_now <= 3/365):
            front_idx += 1
            rolled = True
            roll_type = "FORCED"
            forced_roll_count += 1

        # -------------------------
        # PRINT LOG
        # -------------------------
        if rolled:
            T_days = int(round(T_now * 365))

            total_market_roll += market
            total_fair_roll += fair
            roll_count += 1

            # Adjust for weekends
            if T_days >= 14:
                T_days -= 4  # ~2 weekends
            elif T_days >= 7:
                T_days -= 2  # ~1 weekend

            print(
                f"{date.date()} | Roll ({roll_type}) | "
                f"T-{T_days} | {near} → {far} | "
                f"market={market:.2f} | fair={fair:.2f} | diff={diff:.2f} | z={z:.2f}"
            )

        # -------------------------
        # Record front contract
        # -------------------------
        F.iloc[i] = futures[front_idx]

    # =========================
    # 2. Positions (always short front)
    # =========================
    positions = pd.DataFrame(0, index=fut_price.index, columns=futures)

    for date in positions.index:
        positions.loc[date, F.loc[date]] = -1

    # =========================
    # 3. Transaction costs
    # =========================
    trades = positions.diff().abs().sum(axis=1).fillna(0)
    transaction_cost = trades * t_cost

    # =========================
    # 4. PnL
    # =========================
    price_diff = fut_price.diff().fillna(0)

    raw_pnl = (
        positions.shift(1).fillna(0) * price_diff
    ).sum(axis=1) * multiplier

    net_pnl = raw_pnl - transaction_cost
    net_pnl.iloc[0] = 0

    # =========================
    # 5. NAV
    # =========================
    NAV = initial_nav + net_pnl.cumsum()
    NAV_norm = NAV / NAV.iloc[0]

    # =========================
    # 6. Metrics
    # =========================
    returns = net_pnl / initial_nav

    sharpe = (
        returns.mean() / (returns.std() + 1e-12)
    ) * np.sqrt(252)

    print("\n========== Backtest Summary ==========")
    print(f"Final NAV: {NAV.iloc[-1]:,.1f}")
    print(f"Total Return: {(NAV.iloc[-1]/initial_nav - 1)*100:.6f}%")
    print(f"Sharpe Ratio: {sharpe:.6f}")

    print("\n========== Roll Cost Summary ==========")
    print(f"Total Rolls: {roll_count}")
    print(f"Signal Rolls: {signal_roll_count}")
    print(f"Forced Rolls: {forced_roll_count}")

    print(f"Total Market Roll: {total_market_roll:.4f}")
    print(f"Total Fair Roll: {total_fair_roll:.4f}")
    print(f"(Market - Fair): {(total_market_roll - total_fair_roll):.4f}")

    # =========================
    # Plot
    # =========================
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(NAV_norm, label='Normalized NAV')
        plt.title("Dynamic Roll Strategy (Z-score Driven)")
        plt.grid(True)
        plt.legend()
        plt.show()

    return {
        "NAV": NAV,
        "NAV_norm": NAV_norm,
        "PnL": net_pnl,
        "positions": positions,
        "front_contract": F,
        "rolling_spread": rolling_spread,
        "trades": trades
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

print(futures)

results = run_dynamic_roll_strategy(
    analysis_df=analysis_df,
    futures=futures,
    initial_nav=100_000_000,
    multiplier=50,
    t_cost=2.5,
    lookback=30,
    z_threshold=2.0,
    # direction=-1,
    # hedge_contracts=1,
    plot=True,
)