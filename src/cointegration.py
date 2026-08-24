# ==========================================================
# Engle-Granger cointegration with a FREE beta
# ==========================================================
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

pa = pd.read_csv("data/processed/prices_a_hkd.csv", index_col=0, parse_dates=True)
ph = pd.read_csv("data/processed/prices_h.csv",     index_col=0, parse_dates=True)

log_a = np.log(pa)
log_h = np.log(ph)

print(f"Testing {log_a.shape[1]} pairs\n")

rows = []
for name in log_a.columns:
    # Align and drop any date where either price is missing.
    df = pd.concat([log_a[name], log_h[name]], axis=1).dropna()
    df.columns = ["la", "lh"]
    if len(df) < 500:
        continue

    # ---- Step 1: the long-run relationship --------------------
    # add_constant gives us the intercept (alpha).
    X = sm.add_constant(df["lh"])
    ols = sm.OLS(df["la"], X).fit()
    alpha, beta = ols.params["const"], ols.params["lh"]
    resid = ols.resid

    # Is beta significantly different from 1?
    # t = (estimate - hypothesised) / standard error
    se_beta = ols.bse["lh"]
    t_beta1 = (beta - 1) / se_beta

    # ---- Step 2: are the residuals stationary? ----------------
    # coint() applies MacKinnon critical values, which correct for
    # the fact that OLS already minimised these residuals' variance.
    # Using plain adfuller here would over-reject badly.
    eg_stat, eg_p, eg_crit = coint(df["la"], df["lh"], trend="c", autolag="AIC")

    # For comparison ONLY: the naive (and wrong) ADF on residuals.
    naive_stat, naive_p, *_ = adfuller(resid, regression="c", autolag="AIC")

    rows.append({
        "name": name, "n": len(df),
        "alpha": alpha, "beta": beta, "se_beta": se_beta,
        "t_beta_eq_1": t_beta1, "r2": ols.rsquared,
        "eg_stat": eg_stat, "eg_p": eg_p,
        "crit_5pct": eg_crit[1],
        "naive_adf_p": naive_p,
    })

res = pd.DataFrame(rows).set_index("name")
res["eg_p_fdr"] = multipletests(res["eg_p"], alpha=0.05, method="fdr_bh")[1]
res["cointegrated"] = res["eg_p_fdr"] < 0.05

# ---- Report ------------------------------------------------
print("ENGLE-GRANGER RESULTS")
print(f"  cointegrated (FDR-corrected) : {res['cointegrated'].sum()} / {len(res)}")
print(f"  cointegrated (uncorrected)   : {(res['eg_p'] < 0.05).sum()} / {len(res)}")
print(f"  NAIVE adf would have said    : {(res['naive_adf_p'] < 0.05).sum()} / {len(res)}"
      "   <- why MacKinnon values matter")

print("\nBETA (the cointegrating coefficient)")
print(f"  mean   {res['beta'].mean():.3f}")
print(f"  median {res['beta'].median():.3f}")
print(f"  range  {res['beta'].min():.3f} to {res['beta'].max():.3f}")
print(f"  |t| > 1.96 for beta = 1 : {(res['t_beta_eq_1'].abs() > 1.96).sum()} / {len(res)}")

print("\nDETAIL (sorted by EG statistic)")
show = res[["beta", "t_beta_eq_1", "r2", "eg_stat", "eg_p_fdr",
            "crit_5pct", "cointegrated"]].sort_values("eg_stat")
print(show.to_string(float_format=lambda x: f"{x:8.4f}"))

res.to_csv("data/processed/cointegration.csv")
print("\nSaved -> data/processed/cointegration.csv")