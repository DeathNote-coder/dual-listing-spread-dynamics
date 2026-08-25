import warnings, numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_order
warnings.filterwarnings("ignore")

pa = pd.read_csv("data/processed/prices_a_hkd.csv", index_col=0, parse_dates=True)
ph = pd.read_csv("data/processed/prices_h.csv",     index_col=0, parse_dates=True)

rows = []
for name in pa.columns:
    df = pd.concat([np.log(pa[name]), np.log(ph[name])], axis=1).dropna()
    df.columns = ["la", "lh"]
    if len(df) < 500:
        continue

    # (i) PREREQUISITE: are the individual log prices I(1)?
    # Both should FAIL to reject (i.e. both should be unit roots)
    # for Johansen's framework to even apply correctly.
    p_la = adfuller(df["la"], regression="c", autolag="AIC")[1]
    p_lh = adfuller(df["lh"], regression="c", autolag="AIC")[1]

    k = max(1, select_order(df, maxlags=12, deterministic="ci").aic)
    j = coint_johansen(df, det_order=0, k_ar_diff=k)

    # (ii) Does it ALSO reject rank=1? If so, both series look
    # stationary, which contradicts the I(1) assumption entirely.
    rows.append({
        "name": name, "lags": k,
        "adf_p_la": p_la, "adf_p_lh": p_lh,
        "both_I1": (p_la > 0.05) and (p_lh > 0.05),
        "trace_r0": j.lr1[0], "crit_r0": j.cvt[0, 1],
        "trace_r1": j.lr1[1], "crit_r1": j.cvt[1, 1],
        "rej_r0": j.lr1[0] > j.cvt[0, 1],
        "rej_r1": j.lr1[1] > j.cvt[1, 1],
    })

res = pd.DataFrame(rows).set_index("name")

# A VALID cointegration finding needs: both series I(1),
# rank 0 rejected, rank 1 NOT rejected.
res["valid_coint"] = res["both_I1"] & res["rej_r0"] & ~res["rej_r1"]

print(f"both log prices I(1)          : {res['both_I1'].sum()} / {len(res)}")
print(f"rejects rank=0                : {res['rej_r0'].sum()} / {len(res)}")
print(f"ALSO rejects rank=1 (bad sign): {res['rej_r1'].sum()} / {len(res)}")
print(f"VALID cointegration           : {res['valid_coint'].sum()} / {len(res)}")

# (iii) Sensitivity to det_order — rerun with the alternative spec
rows2 = []
for name in pa.columns:
    df = pd.concat([np.log(pa[name]), np.log(ph[name])], axis=1).dropna()
    df.columns = ["la", "lh"]
    if len(df) < 500:
        continue
    k = max(1, select_order(df, maxlags=12, deterministic="ci").aic)
    j2 = coint_johansen(df, det_order=-1, k_ar_diff=k)  # no constant at all
    rows2.append({"name": name, "rej_r0_altspec": j2.lr1[0] > j2.cvt[0, 1]})

res2 = pd.DataFrame(rows2).set_index("name")
merged = res.join(res2)
print(f"\nrej_r0 with det_order=0  : {merged['rej_r0'].sum()} / {len(merged)}")
print(f"rej_r0 with det_order=-1 : {merged['rej_r0_altspec'].sum()} / {len(merged)}")
print(f"agree across both specs  : {(merged['rej_r0'] == merged['rej_r0_altspec']).sum()} / {len(merged)}")

res.to_csv("data/processed/johansen_diagnostic.csv")