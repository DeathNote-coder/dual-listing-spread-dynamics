import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ah = pd.read_csv("data/processed/ah_premiums.csv", index_col=0, parse_dates=True)
ah_ew = ah.drop(columns=["EW_INDEX"], errors="ignore").mean(axis=1)

adr = pd.read_csv("data/processed/adr_premiums.csv", index_col=0, parse_dates=True)
summ = pd.read_csv("data/processed/adr_summary.csv", index_col=0)
verified = summ[(summ["ratio_error"] < 0.01) & (summ["drift"] < 0.01)].index
verified = [c for c in verified if c in adr.columns]
adr_ew = adr[verified].mean(axis=1)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ah_ew.index, ah_ew * 100, label="A-H (equal-weighted, 63 pairs)",
       color="firebrick", lw=1.2)
ax.plot(adr_ew.index, adr_ew * 100, label="H-ADR (equal-weighted, 20 verified pairs)",
       color="steelblue", lw=1.2)
ax.axhline(0, color="black", ls="--", lw=0.8)
ax.set_ylabel("Premium (%)")
ax.set_xlabel("Date")
ax.set_title("A-H vs H-ADR: same estimator, same window, one structural difference")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/ah_vs_adr_comparison.png", dpi=150)
plt.show()
print("Saved -> results/figures/ah_vs_adr_comparison.png")