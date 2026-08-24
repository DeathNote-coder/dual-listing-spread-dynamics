# ==========================================================
# Stationarity: is the log spread mean-reverting or a random walk?
# ==========================================================
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.multitest import multipletests
from arch.unitroot import PhillipsPerron

warnings.filterwarnings("ignore")   # KPSS warns when p is off its lookup table

panel = pd.read_csv("data/processed/ah_premiums.csv",
                    index_col=0, parse_dates=True)
panel = panel.drop(columns=["EW_INDEX"], errors="ignore")

# Work in log spread: log(P_A * FX / P_H) = log(1 + premium)
logs = np.log(1 + panel)

print(f"Testing {logs.shape[1]} pairs, {logs.shape[0]} observations each\n")

def test_one(series):
    """Run ADF, PP and KPSS on one series. Return a dict of results."""
    s = series.dropna()

    # regression="c": include a constant, no time trend.
    # We expect reversion to a non-zero LEVEL, not to zero and not
    # around a trend. Including a trend when there isn't one costs power.
    # autolag="AIC": let AIC pick the lag order.
    adf_stat, adf_p, adf_lags, *_ = adfuller(s, regression="c", autolag="AIC")

    pp = PhillipsPerron(s, trend="c")

    # KPSS: nlags="auto" uses the data-driven bandwidth.
    # NOTE the flipped null — small p means REJECT stationarity.
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")

    return {
        "adf_stat": adf_stat, "adf_p": adf_p, "adf_lags": adf_lags,
        "pp_stat": pp.stat,   "pp_p": pp.pvalue,
        "kpss_stat": kpss_stat, "kpss_p": kpss_p,
    }

rows = []
for name in logs.columns:
    r = test_one(logs[name])
    r["name"] = name
    rows.append(r)

res = pd.DataFrame(rows).set_index("name")

# ---- Multiple testing correction --------------------------
# 63 tests at 5% would produce ~3 false rejections by chance.
# Benjamini-Hochberg controls the false DISCOVERY rate.
res["adf_p_fdr"] = multipletests(res["adf_p"], alpha=0.05, method="fdr_bh")[1]
res["pp_p_fdr"]  = multipletests(res["pp_p"],  alpha=0.05, method="fdr_bh")[1]

# ---- Verdicts ---------------------------------------------
res["adf_rejects"]  = res["adf_p_fdr"] < 0.05      # evidence FOR stationarity
res["pp_rejects"]   = res["pp_p_fdr"] < 0.05       # evidence FOR stationarity
res["kpss_rejects"] = res["kpss_p"] < 0.05         # evidence AGAINST stationarity

def verdict(row):
    if row["adf_rejects"] and not row["kpss_rejects"]:
        return "STATIONARY"
    if not row["adf_rejects"] and row["kpss_rejects"]:
        return "UNIT ROOT"
    if row["adf_rejects"] and row["kpss_rejects"]:
        return "CONFLICT"          # often a structural break
    return "INCONCLUSIVE"

res["verdict"] = res.apply(verdict, axis=1)

# ---- Report -----------------------------------------------
print("VERDICT COUNTS (after FDR correction)")
print(res["verdict"].value_counts().to_string())

print(f"\n  ADF rejects unit root  : {res['adf_rejects'].sum()} / {len(res)}"
      f"   (uncorrected: {(res['adf_p'] < 0.05).sum()})")
print(f"  PP rejects unit root   : {res['pp_rejects'].sum()} / {len(res)}"
      f"   (uncorrected: {(res['pp_p'] < 0.05).sum()})")
print(f"  KPSS rejects stationary: {res['kpss_rejects'].sum()} / {len(res)}")
print(f"  ADF and PP agree       : "
      f"{(res['adf_rejects'] == res['pp_rejects']).sum()} / {len(res)}")

print("\nDETAIL (sorted by ADF statistic — most negative = strongest evidence)")
show = res[["adf_stat", "adf_p_fdr", "pp_p_fdr", "kpss_stat",
            "kpss_p", "verdict"]].sort_values("adf_stat")
print(show.to_string(float_format=lambda x: f"{x:8.4f}"))

res.to_csv("data/processed/stationarity.csv")
print("\nSaved -> data/processed/stationarity.csv")