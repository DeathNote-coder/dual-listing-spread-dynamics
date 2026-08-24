# ==========================================================
# Ornstein-Uhlenbeck half-lives for both groups
# ==========================================================
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")


def half_life(series):
    """
    Fit S_t = alpha + phi * S_{t-1} + eps, return (phi, half-life, se).

    half-life = ln(2) / -ln(phi)   -- days for a shock to decay by half.
    Returns inf if phi >= 1 (no mean reversion).
    """
    s = series.dropna()
    lag = s.shift(1).dropna()
    cur = s.loc[lag.index]

    ols = sm.OLS(cur, sm.add_constant(lag)).fit()
    phi = ols.params.iloc[1]
    se = ols.bse.iloc[1]

    if phi >= 1 or phi <= 0:
        return phi, np.inf, se
    return phi, np.log(2) / (-np.log(phi)), se


# ---- A-H group --------------------------------------------
ah = pd.read_csv("data/processed/ah_premiums.csv", index_col=0, parse_dates=True)
ah = ah.drop(columns=["EW_INDEX"], errors="ignore")
ah_log = np.log(1 + ah)

rows = []
for name in ah_log.columns:
    phi, hl, se = half_life(ah_log[name])
    # t-stat for H0: phi = 1 (i.e. a unit root / no reversion)
    rows.append({"name": name, "group": "A-H", "phi": phi,
                 "t_phi_eq_1": (phi - 1) / se, "half_life": hl})

# ---- H-ADR group ------------------------------------------
adr = pd.read_csv("data/processed/adr_premiums.csv", index_col=0, parse_dates=True)
summ = pd.read_csv("data/processed/adr_summary.csv", index_col=0)
keep = summ[(summ["ratio_error"] < 0.01) & (summ["drift"] < 0.01)].index
keep = [k for k in keep if k in adr.columns]
adr_log = np.log(1 + adr[keep])

for name in adr_log.columns:
    phi, hl, se = half_life(adr_log[name])
    rows.append({"name": name, "group": "H-ADR", "phi": phi,
                 "t_phi_eq_1": (phi - 1) / se, "half_life": hl})

res = pd.DataFrame(rows).set_index("name")

# ---- Report ------------------------------------------------
for g in ["A-H", "H-ADR"]:
    sub = res[res.group == g]
    finite = sub[np.isfinite(sub["half_life"])]
    print(f"\n{g}  ({len(sub)} pairs)")
    print(f"  mean phi            : {sub['phi'].mean():.4f}")
    print(f"  median phi          : {sub['phi'].median():.4f}")
    print(f"  phi range           : {sub['phi'].min():.4f} - {sub['phi'].max():.4f}")
    if len(finite):
        print(f"  median half-life    : {finite['half_life'].median():.1f} days")
        print(f"  half-life range     : {finite['half_life'].min():.1f}"
              f" - {finite['half_life'].max():.1f} days")
    print(f"  pairs with HL > 1yr : {(sub['half_life'] > 252).sum()} / {len(sub)}")

print("\n" + "=" * 58)
print(f"{'':22s}{'A-H':>16s}{'H-ADR':>16s}")
a, h = res[res.group == "A-H"], res[res.group == "H-ADR"]
print(f"  {'median phi':20s}{a['phi'].median():>16.4f}{h['phi'].median():>16.4f}")
print(f"  {'median half-life':20s}"
      f"{a['half_life'].median():>13.1f} d{h['half_life'].median():>13.1f} d")

print("\nDETAIL (sorted by half-life)")
print(res.sort_values("half_life").to_string(
      float_format=lambda x: f"{x:10.4f}"))

res.to_csv("data/processed/half_lives.csv")
print("\nSaved -> data/processed/half_lives.csv")