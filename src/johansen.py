import warnings, numpy as np, pandas as pd
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

    # Pick VAR lag order by AIC, then Johansen with k_ar_diff = lags-1.
    k = max(1, select_order(df, maxlags=12, deterministic="ci").aic)
    jres = coint_johansen(df, det_order=0, k_ar_diff=k)

    # trace statistic for H0: rank = 0 (no cointegration)
    # vs 5% critical value (column index 1)
    trace0, crit0 = jres.lr1[0], jres.cvt[0, 1]
    rows.append({"name": name, "lags": k,
                 "trace_r0": trace0, "crit_5pct": crit0,
                 "rejects_r0": trace0 > crit0})

res = pd.DataFrame(rows).set_index("name")
print(f"Johansen rejects rank=0 : {res['rejects_r0'].sum()} / {len(res)}")
print(res.sort_values("trace_r0", ascending=False).to_string(
      float_format=lambda x: f"{x:8.3f}"))
res.to_csv("data/processed/johansen.csv")