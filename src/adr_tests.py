# ==========================================================
# Run the A-H test battery on the H-ADR control group
# ==========================================================
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_order
from statsmodels.stats.multitest import multipletests
from arch.unitroot import PhillipsPerron

warnings.filterwarnings("ignore")

prem = pd.read_csv("data/processed/adr_premiums.csv", index_col=0, parse_dates=True)
p_hk = pd.read_csv("data/processed/adr_prices_hk.csv", index_col=0, parse_dates=True)
p_ny = pd.read_csv("data/processed/adr_prices_adr.csv", index_col=0, parse_dates=True)
summ = pd.read_csv("data/processed/adr_summary.csv", index_col=0)

# Same rule-based filter as before: verified integer ratio, stable.
keep = summ[(summ["ratio_error"] < 0.01) & (summ["drift"] < 0.01)].index
keep = [k for k in keep if k in prem.columns]
print(f"Testing {len(keep)} verified H-ADR pairs\n")

rows = []
for name in keep:
    # --- 1. Stationarity of the log spread ------------------
    s = np.log(1 + prem[name]).dropna()
    adf_stat, adf_p, *_ = adfuller(s, regression="c", autolag="AIC")
    pp = PhillipsPerron(s, trend="c")
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")

    # --- 2. Engle-Granger on the log prices -----------------
    df = pd.concat([np.log(p_ny[name]), np.log(p_hk[name])], axis=1).dropna()
    df.columns = ["lny", "lhk"]
    ols = sm.OLS(df["lny"], sm.add_constant(df["lhk"])).fit()
    beta = ols.params["lhk"]
    t_b1 = (beta - 1) / ols.bse["lhk"]
    eg_stat, eg_p, _ = coint(df["lny"], df["lhk"], trend="c", autolag="AIC")

    # --- 3. Johansen ----------------------------------------
    k = max(1, select_order(df, maxlags=12, deterministic="ci").aic)
    j = coint_johansen(df, det_order=0, k_ar_diff=k)

    rows.append({
        "name": name, "n": len(s),
        "adf_stat": adf_stat, "adf_p": adf_p,
        "pp_p": pp.pvalue, "kpss_stat": kpss_stat, "kpss_p": kpss_p,
        "beta": beta, "t_beta_eq_1": t_b1, "r2": ols.rsquared,
        "eg_stat": eg_stat, "eg_p": eg_p,
        "trace_r0": j.lr1[0], "crit_r0": j.cvt[0, 1],
        "joh_rejects": j.lr1[0] > j.cvt[0, 1],
    })

res = pd.DataFrame(rows).set_index("name")

# Same FDR correction applied to the A-H group.
res["adf_p_fdr"] = multipletests(res["adf_p"], method="fdr_bh")[1]
res["eg_p_fdr"]  = multipletests(res["eg_p"],  method="fdr_bh")[1]
res["adf_rej"]  = res["adf_p_fdr"] < 0.05
res["eg_rej"]   = res["eg_p_fdr"] < 0.05
res["kpss_rej"] = res["kpss_p"] < 0.05

n = len(res)
print("H-ADR RESULTS (FDR-corrected)")
print(f"  ADF rejects unit root    : {res['adf_rej'].sum():2d} / {n}")
print(f"  KPSS rejects stationarity: {res['kpss_rej'].sum():2d} / {n}")
print(f"  Engle-Granger cointegrated: {res['eg_rej'].sum():2d} / {n}")
print(f"  Johansen rejects rank=0   : {res['joh_rejects'].sum():2d} / {n}")
print(f"\n  beta: mean {res['beta'].mean():.3f}, "
      f"median {res['beta'].median():.3f}, "
      f"range {res['beta'].min():.3f}-{res['beta'].max():.3f}")
print(f"  |t| > 1.96 for beta = 1  : {(res['t_beta_eq_1'].abs() > 1.96).sum():2d} / {n}")
print(f"  mean R^2                 : {res['r2'].mean():.4f}")

print("\n" + "=" * 62)
print("SIDE BY SIDE")
print("=" * 62)

ah = {"adf": "0 / 63", "eg": "0 / 63", "joh": "31 / 63",
      "prem": "+77.95%", "beta": "0.770", "r2": "~0.70 (spurious)"}

adr = {"adf": f"{res['adf_rej'].sum()} / {n}",
       "eg":  f"{res['eg_rej'].sum()} / {n}",
       "joh": f"{res['joh_rejects'].sum()} / {n}",
       "prem": "-0.15%",
       "beta": f"{res['beta'].mean():.3f}",
       "r2":   f"{res['r2'].mean():.4f}"}

print(f"{'':26s}{'A-H (63)':>18s}{'H-ADR (' + str(n) + ')':>18s}")
for label, key in [("ADF rejects unit root", "adf"),
                   ("EG cointegrated", "eg"),
                   ("Johansen rejects r=0", "joh"),
                   ("mean premium", "prem"),
                   ("mean beta", "beta"),
                   ("mean R^2", "r2")]:
    print(f"  {label:24s}{ah[key]:>18s}{adr[key]:>18s}")

print("\nDETAIL")
print(res[["n", "adf_stat", "adf_p_fdr", "kpss_stat", "beta",
           "t_beta_eq_1", "r2", "eg_stat", "eg_p_fdr", "trace_r0",
           "joh_rejects"]].sort_values("eg_stat").to_string(
      float_format=lambda x: f"{x:8.4f}"))

res.to_csv("data/processed/adr_tests.csv")
print("\nSaved -> data/processed/adr_tests.csv")