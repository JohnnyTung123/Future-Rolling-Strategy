from scipy.stats import ttest_1samp

# =========================
# RUN BACKTEST
# =========================

lookbacks = [10, 20, 30]
z_thresholds = [2.0, 2.5, 3.0]

results_list = []

# store full outputs
full_results = {}

for lb in lookbacks:
    for z in z_thresholds:

        print(f"\nRunning: lookback={lb}, z_threshold={z}")

        res = run_dynamic_roll_strategy(
            analysis_df=analysis_df,
            futures=futures,
            initial_nav=100_000_000,
            multiplier=50,
            t_cost=2.5,
            lookback=lb,
            z_threshold=z,
            hedge_contracts=1,
            plot=False
        )

        # store full result
        full_results[(lb, z)] = res

        results_list.append({
            "lookback": lb,
            "z_threshold": z,
            "final_nav": res["final_nav"],
            "Total Market Roll": res["total_market_roll"],
            "Total Fair Roll": res["total_fair_roll"],
            "Market - Fair": (
                res["total_market_roll"]
                - res["total_fair_roll"]
            ),
        })

# =========================
# BUILD RESULT TABLES
# =========================

results_df = pd.DataFrame(results_list)

pivot_market = results_df.pivot(
    index="lookback",
    columns="z_threshold",
    values="Total Market Roll"
)

pivot_fair = results_df.pivot(
    index="lookback",
    columns="z_threshold",
    values="Total Fair Roll"
)

pivot_diff = results_df.pivot(
    index="lookback",
    columns="z_threshold",
    values="Market - Fair"
)

# =========================
# PRINT TABLES
# =========================

print("\n=== Total Market Roll ===")
print(pivot_market)

print("\n=== Total Fair Roll ===")
print(pivot_fair)

print("\n=== Market - Fair ===")
print(pivot_diff)

# =========================
# FIND OPTIMAL PARAMETER SET
# =========================

best_row = results_df.loc[
    results_df["Market - Fair"].idxmax()
]

best_lb = best_row["lookback"]
best_z = best_row["z_threshold"]

print("\n========== Optimal Parameter ==========")
print(f"Best Lookback: {best_lb}")
print(f"Best Z-threshold: {best_z}")

# =========================
# T-TEST
# Compare:
# baseline = (20, 2.0)
# vs optimal parameter set
# =========================

baseline_res = full_results[(20, 2.0)]
optimal_res = full_results[(best_lb, best_z)]

# convert to numeric
baseline_diff = pd.to_numeric(
    baseline_res["rolling_spread"]["difference"],
    errors="coerce"
).dropna()

optimal_diff = pd.to_numeric(
    optimal_res["rolling_spread"]["difference"],
    errors="coerce"
).dropna()

# align dates
common_dates = baseline_diff.index.intersection(
    optimal_diff.index
)

baseline_diff = baseline_diff.loc[common_dates]
optimal_diff = optimal_diff.loc[common_dates]

# improvement series
spread_improvement = optimal_diff - baseline_diff

# ensure numeric dtype
spread_improvement = spread_improvement.astype(float)

# =========================
# ONE-SAMPLE T-TEST
# H0: mean improvement = 0
# =========================

t_stat, p_value = ttest_1samp(
    spread_improvement,
    0
)

# =========================
# PRINT T-TEST RESULTS
# =========================

print("\n========== T-Test ==========")

print(
    f"Baseline Mean Spread : "
    f"{baseline_diff.mean():.6f}"
)

print(
    f"Optimal Mean Spread  : "
    f"{optimal_diff.mean():.6f}"
)

print(
    f"Mean Improvement     : "
    f"{spread_improvement.mean():.6f}"
)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print(
        "Reject H0: statistically significant difference"
    )
else:
    print(
        "Fail to reject H0: spreads are statistically similar"
    )