# ==========================================================
# Descriptive statistics across the 63-pair panel
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# index_col=0 -> use the first column (dates) as the row index
# parse_dates=True -> read them as real dates, not strings
panel = pd.read_csv("data/processed/ah_premiums.csv",
                    index_col=0, parse_dates=True)
summary = pd.read_csv("data/processed/ah_summary.csv")

print(f"Panel: {panel.shape[0]} dates x {panel.shape[1]} pairs\n")

# ---- 1. Higher moments ------------------------------------
# We have mean/median/std already. Skew and kurtosis tell us
# about SHAPE: is the distribution symmetric? fat-tailed?
print("HIGHER MOMENTS OF THE MEAN PREMIUM (cross-section)")
m = summary["mean"]
print(f"  mean     {m.mean():+.1%}")
print(f"  median   {m.median():+.1%}")
print(f"  std      {m.std():.1%}")
print(f"  skew     {m.skew():+.2f}")
print(f"  kurtosis {m.kurtosis():+.2f}")
print(f"  IQR      {m.quantile(0.25):+.1%} to {m.quantile(0.75):+.1%}")

# ---- 2. The equal-weighted index --------------------------
# Average across all pairs on each DATE. This is the closest
# thing to "the A-H premium" as a single time series.
# axis=1 means average across columns (pairs), not down rows.
panel["EW_INDEX"] = panel.drop(columns=["EW_INDEX"], errors="ignore").mean(axis=1)
ew = panel["EW_INDEX"]

print("\nEQUAL-WEIGHTED A-H PREMIUM INDEX")
print(f"  mean   {ew.mean():+.1%}")
print(f"  min    {ew.min():+.1%}  on {ew.idxmin().date()}")
print(f"  max    {ew.max():+.1%}  on {ew.idxmax().date()}")
print("\n  by year:")
print((ew.groupby(ew.index.year).mean() * 100).round(1))

# ---- 3. Is the tier gap real, or driven by stale prices? --
# Core vs extended differ in premium AND in staleness.
# Check whether staleness alone explains the gap.
summary["stale_pct"] = (summary["stale_a"] + summary["stale_h"]) / (2 * summary["n"])

print("\nTIER COMPARISON")
tier = summary.groupby("tier").agg(
    pairs=("name", "count"),
    mean_premium=("mean", "mean"),
    mean_std=("std", "mean"),
    mean_stale=("stale_pct", "mean"),
)
print(tier.round(3))

# Correlation between staleness and premium across all pairs.
# If illiquidity drives the premium, this should be positive.
r = summary["stale_pct"].corr(summary["mean"])
print(f"\n  corr(staleness, mean premium) = {r:+.3f}")

# And within core only — if it survives here, it isn't just
# a core-vs-extended labelling artefact.
core = summary[summary.tier == "core"]
print(f"  corr within core only         = "
      f"{core['stale_pct'].corr(core['mean']):+.3f}")

# ---- 4. Do premiums move together? ------------------------
# If all 63 rise and fall together, there's one common factor
# (capital controls). If not, they're idiosyncratic.
corr_matrix = panel.drop(columns=["EW_INDEX"]).corr()
# Take the upper triangle only, excluding the diagonal of 1s.
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
pairwise = upper.stack().dropna()

print("\nCO-MOVEMENT ACROSS PAIRS")
print(f"  mean pairwise correlation   {pairwise.mean():+.3f}")
print(f"  median                      {pairwise.median():+.3f}")
print(f"  share of pairs r > 0.5      {(pairwise > 0.5).mean():.1%}")
print(f"  share of pairs r < 0        {(pairwise < 0).mean():.1%}")

# ---- 5. Charts --------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) distribution of mean premiums
axes[0, 0].hist(summary["mean"] * 100, bins=25, color="steelblue",
                edgecolor="white")
axes[0, 0].axvline(0, color="black", ls="--", lw=1)
axes[0, 0].set_xlabel("Mean A-H premium (%)")
axes[0, 0].set_ylabel("Number of pairs")
axes[0, 0].set_title(f"All {len(summary)} pairs trade at a premium")

# (b) the equal-weighted index over time
axes[0, 1].plot(ew.index, ew * 100, color="firebrick", lw=1)
axes[0, 1].axhline(0, color="black", ls="--", lw=0.8)
axes[0, 1].set_ylabel("Premium (%)")
axes[0, 1].set_title("Equal-weighted A-H premium index")

# (c) staleness vs premium — the liquidity question
axes[1, 0].scatter(summary["stale_pct"] * 100, summary["mean"] * 100,
                   c=(summary["tier"] == "extended").map({True: "orange",
                                                          False: "steelblue"}),
                   alpha=0.7)
axes[1, 0].set_xlabel("Stale days (% of sample)")
axes[1, 0].set_ylabel("Mean premium (%)")
axes[1, 0].set_title("Illiquidity vs premium (orange = extended)")

# (d) volatility vs level
axes[1, 1].scatter(summary["mean"] * 100, summary["std"] * 100,
                   alpha=0.7, color="seagreen")
axes[1, 1].set_xlabel("Mean premium (%)")
axes[1, 1].set_ylabel("Std dev of premium (%)")
axes[1, 1].set_title("Higher premiums are more volatile?")

for ax in axes.flat:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/descriptives.png", dpi=150)
plt.show()

# The log spread — the object everything else is built on
log_panel = np.log(1 + panel.drop(columns=["EW_INDEX"], errors="ignore"))
log_panel.to_csv("data/processed/ah_log_spreads.csv")

# Sanity check: does the mean-vs-std relationship flatten out?
ls = log_panel.agg(["mean", "std"]).T
print(f"corr(mean, std) in raw premium : {summary['mean'].corr(summary['std']):+.3f}")
print(f"corr(mean, std) in log spread  : {ls['mean'].corr(ls['std']):+.3f}")