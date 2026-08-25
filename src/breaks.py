import warnings, numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller, zivot_andrews
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

panel = pd.read_csv("data/processed/ah_premiums.csv", index_col=0, parse_dates=True)
logs  = np.log(1 + panel.drop(columns=["EW_INDEX"], errors="ignore"))

rows = []
for name in logs.columns:
    s = logs[name].dropna()

    # Zivot-Andrews: H0 = unit root. regression="c" allows a break
    # in the INTERCEPT (a level shift), which is what we expect.
    za_stat, za_p, za_crit, za_lag, za_bpidx = zivot_andrews(s, regression="c", autolag="AIC")
    n = len(s)
    trim = int(0.15 * n)
    za_lower = s.index[trim]
    za_upper = s.index[n - trim]

    # Same series, post-2016 only.
    sub = s.loc["2016-01-01":]
    sub_stat, sub_p, *_ = adfuller(sub, regression="c", autolag="AIC")

    rows.append({"name": name,
                     "za_stat": za_stat, "za_p": za_p,
                     "break_date": s.index[za_bpidx].date(),
                     "za_window_start": za_lower.date(),
                     "za_window_end": za_upper.date(),
                     "sub_stat": sub_stat, "sub_p": sub_p})

res = pd.DataFrame(rows).set_index("name")
res["za_p_fdr"]  = multipletests(res["za_p"],  method="fdr_bh")[1]
res["sub_p_fdr"] = multipletests(res["sub_p"], method="fdr_bh")[1]



print(f"Zivot-Andrews rejects unit root : {(res['za_p_fdr'] < 0.05).sum()} / {len(res)}")
print(f"Post-2016 ADF rejects           : {(res['sub_p_fdr'] < 0.05).sum()} / {len(res)}")
print("\nMost common break dates:")
print(pd.to_datetime(res["break_date"]).dt.to_period("Q").value_counts().head(8))
print(res.sort_values("za_stat").to_string(float_format=lambda x: f"{x:8.4f}"))
res.to_csv("data/processed/breaks.csv")
print(f"\nZA search window (same for all pairs): {res['za_window_start'].iloc[0]} to {res['za_window_end'].iloc[0]}")