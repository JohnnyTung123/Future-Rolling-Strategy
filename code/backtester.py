import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# =========================
# BACKTEST FUNCTION
# =========================
def run_futures_backtest(
    analysis_df,
    futures,
    initial_nav=100_000_000,
    multiplier=50,
    t_cost=2.5,
    roll_window=1/365,
    direction=-1,          # +1 = long, -1 = short
    hedge_contracts=1,     # number of contracts per signal
    save_path=None,
    plot=True
):

    # =========================
    # Extract data
    # =========================
    fut_price = analysis_df['PX'][futures]
    fair_price = analysis_df['Fair'][futures]
    T = analysis_df['T']

    # =========================
    # 1. Roll logic (front contract)
    # =========================
    front_idx = 0
    F = pd.Series(index=fut_price.index, dtype=object)

    rolling_spread = pd.DataFrame(
        index=fut_price.index,
        columns=["fair", "market", "difference"]
    )

    for i, date in enumerate(fut_price.index):
        front = futures[front_idx]

        if T.loc[date, front] <= roll_window:
            if front_idx < len(futures) - 1:
                near = futures[front_idx]
                far = futures[front_idx + 1]

                fair = fair_price.loc[date, far] - fair_price.loc[date, near]
                market = fut_price.loc[date, far] - fut_price.loc[date, near]

                rolling_spread.loc[date, "fair"] = fair *  hedge_contracts
                rolling_spread.loc[date, "market"] = market * hedge_contracts
                rolling_spread["difference"] = (
                        rolling_spread["market"] - rolling_spread["fair"]
                )

                front_idx += 1

        F.iloc[i] = futures[front_idx]

    # =========================
    # 2. Positions
    # =========================
    positions = pd.DataFrame(0, index=fut_price.index, columns=futures)

    for date in positions.index:
        positions.loc[date, F.loc[date]] = direction * hedge_contracts

    # =========================
    # 3. Transaction costs
    # =========================
    trades = positions.diff().abs().sum(axis=1).fillna(0)
    transaction_cost = trades * t_cost
    roll_dates = trades[trades == 2]

    # =========================
    # 4. PnL
    # =========================
    price_diff = fut_price.diff().fillna(0)

    raw_pnl = (
        positions.shift(1).fillna(0) * price_diff
    ).sum(axis=1) * multiplier

    net_pnl = raw_pnl - transaction_cost

    # avoid first-step distortion
    net_pnl.iloc[0] = 0

    # =========================
    # 5. NAV
    # =========================
    NAV = initial_nav + net_pnl.cumsum()
    NAV_norm = NAV / NAV.iloc[0]

    # =========================
    # 6. Save outputs
    # =========================
    if save_path:
        # convert roll window back to days
        roll_days = int(round(roll_window * 365))

        roll_path = f"{save_path}/roll_{roll_days}d"

        os.makedirs(roll_path, exist_ok=True)

        F.to_csv(f"{roll_path}/F.csv")
        positions.to_csv(f"{roll_path}/positions.csv")
        NAV.to_csv(f"{roll_path}/NAV.csv")
        rolling_spread.to_csv(f"{roll_path}/rolling_spread.csv")

    # =========================
    # 7. Plots
    # =========================
    # if plot:
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(NAV_norm, label='Normalized NAV', linewidth=2)
    #     plt.title("Strategy NAV")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()

    # =========================
    # 8. Metrics
    # =========================
    # returns = net_pnl / initial_nav
    #
    # sharpe = (
    #     returns.mean() / (returns.std() + 1e-12)
    # ) * np.sqrt(252)

    # print("\n========== Backtest Summary ==========")
    # print(f"Final NAV: {NAV.iloc[-1]:,.0f}")
    # print(f"Total Return: {(NAV.iloc[-1]/initial_nav - 1)*100:.2f}%")
    # print(f"Sharpe Ratio: {sharpe:.2f}")
    # print(f"Total Rolling: {trades.sum():.0f}")

    return {
        "NAV": NAV,
        "NAV_norm": NAV_norm,
        "PnL": net_pnl,
        "positions": positions,
        "front_contract": F,
        "trades": trades,
        "rolling_spread": rolling_spread
    }

# =========================
# LOAD DATA
# =========================
analysis_df = pd.read_csv(
    '../data/processed/analysis_multi.csv',
    header=[0, 1],
    index_col=0
)

analysis_df.index = pd.to_datetime(analysis_df.index)

unique_cols = analysis_df.columns.get_level_values(1).unique()
exclude = ['ESH1', 'ESM1', 'ESU1']
futures = [
    c for c in unique_cols
    if re.fullmatch(r'ES[HMUZ]\d+', c)
    and c not in exclude
]

print(futures)

# E-mini S&P 500 (ES) futures expire quarterly on the third Friday of March, June, September, and December
# ISSUE: I don't have access to close price at weekend. T-5(Sunday) is the same as T-4(Monday)

# =========================
# RUN BACKTEST
# =========================

results_list = []
roll_list = list(range(1, 5)) + [7, 8, 9, 10, 11, 14]
print("\n========== Data Range ==========")
print(f"Start: {analysis_df.index.min()}")
print(f"End:   {analysis_df.index.max()}")

for d in roll_list:
    roll_window = d / 365
    # convert business days → approx calendar days
    cal_days = d
    if cal_days >= 14:
        cal_days = cal_days - 4
    elif cal_days >= 7:
        cal_days = cal_days - 2
    print(f"\nRunning backtest: roll_days = {d} (calendar days) ≈ T-{cal_days} (business days)")

    res = run_futures_backtest(
        analysis_df=analysis_df,
        futures=futures,
        initial_nav=100_000_000,
        multiplier=50,
        t_cost=2.5,
        roll_window=roll_window,
        direction=-1,  # +1 = long, -1 = short
        hedge_contracts=1,  # number of contracts per signal (assume we roll the same amount)
        save_path='../data/backtest/static_roll_analysis',
        plot=False
    )

    NAV = res["NAV"]
    PnL = res["PnL"]

    roll_spread = res["rolling_spread"].dropna()
    print("\n========== Rolling Spread ==========")
    print(roll_spread)
    net_market = roll_spread["market"].sum()
    net_fair = roll_spread["fair"].sum()
    net_diff = roll_spread["difference"].sum()

    print(f"Total Market Roll: {net_market:.4f}")
    print(f"Total Fair Roll:   {net_fair:.4f}")

    returns = PnL / 100_000_000
    sharpe = (returns.mean() / (returns.std())) * np.sqrt(252)

    results_list.append({
        "roll_window_days": cal_days,
        "final_nav": NAV.iloc[-1],
        # "total_return_pct": (NAV.iloc[-1] / 100_000_000 - 1) * 100,
        # "sharpe": sharpe,
        "market_roll_spread": net_market,
        "fair_roll_spread": net_fair,
        "market-fair": net_diff
    })

results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values("roll_window_days", ascending=False)
pd.set_option('display.width', None)
print(results_df)

