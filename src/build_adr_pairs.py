"""
build_adr_pairs.py
==================
H-ADR pairs: the same company listed in Hong Kong AND New York.

This is the CONTROL GROUP. Unlike A-H pairs, these have a working
arbitrage mechanism: a depositary bank is contractually obliged to
convert N Hong Kong shares into 1 ADR and back, on demand.

If the law of one price is enforced by institutional machinery rather
than by economics, these spreads should be near zero while A-H spreads
are not.

Ratio verification
------------------
Rather than trusting a hardcoded conversion ratio, the script INFERS it:

    implied ratio = median( ADR price in HKD / HK price )

and flags any pair where that does not round cleanly to an integer.
It also splits the sample in half to detect a ratio CHANGE mid-sample
(the Alibaba 8:1 trap, where the ratio changed in 2019).

Known limitation
----------------
Hong Kong closes at 08:00 GMT; New York closes at 21:00 GMT. Prices on
"the same date" are 13 hours apart, with a whole US session in between.
This mechanically WIDENS measured H-ADR spreads, so results here are a
conservative estimate of how tightly the two listings are tethered.
"""

import numpy as np
import pandas as pd
import yfinance as yf

START, END = "2014-01-01", "2026-08-20"

# ratio_guess is only a starting point — the script verifies it.
ADR_PAIRS = [
    {"name": "Alibaba",          "hk": "9988.HK", "adr": "BABA", "ratio_guess": 8},
    {"name": "JD.com",           "hk": "9618.HK", "adr": "JD",   "ratio_guess": 2},
    {"name": "NetEase",          "hk": "9999.HK", "adr": "NTES", "ratio_guess": 5},
    {"name": "Baidu",            "hk": "9888.HK", "adr": "BIDU", "ratio_guess": 8},
    {"name": "Bilibili",         "hk": "9626.HK", "adr": "BILI", "ratio_guess": 1},
    {"name": "Li Auto",          "hk": "2015.HK", "adr": "LI",   "ratio_guess": 2},
    {"name": "XPeng",            "hk": "9868.HK", "adr": "XPEV", "ratio_guess": 2},
    {"name": "NIO",              "hk": "9866.HK", "adr": "NIO",  "ratio_guess": 1},
    {"name": "Trip.com",         "hk": "9961.HK", "adr": "TCOM", "ratio_guess": 1},
    {"name": "Yum China",        "hk": "9987.HK", "adr": "YUMC", "ratio_guess": 1},
    {"name": "Weibo",            "hk": "9898.HK", "adr": "WB",   "ratio_guess": 1},
    {"name": "BeiGene",          "hk": "6160.HK", "adr": "ONC",  "ratio_guess": 13},
    {"name": "Zai Lab",          "hk": "9688.HK", "adr": "ZLAB", "ratio_guess": 10},
    {"name": "Autohome",         "hk": "2518.HK", "adr": "ATHM", "ratio_guess": 4},
    {"name": "New Oriental Ed.", "hk": "9901.HK", "adr": "EDU",  "ratio_guess": 10},
    {"name": "ZTO Express",      "hk": "2057.HK", "adr": "ZTO",  "ratio_guess": 1},
    {"name": "Baozun",           "hk": "9991.HK", "adr": "BZUN", "ratio_guess": 3},
    {"name": "Lufax",            "hk": "6623.HK", "adr": "LU",   "ratio_guess": 2},
    {"name": "Tuya",             "hk": "2391.HK", "adr": "TUYA", "ratio_guess": 1},
    {"name": "Miniso",           "hk": "9896.HK", "adr": "MNSO", "ratio_guess": 4},
    {"name": "Kanzhun (BOSS)",   "hk": "2076.HK", "adr": "BZ",   "ratio_guess": 2},
    {"name": "Qifu Technology",  "hk": "3660.HK", "adr": "QFIN", "ratio_guess": 2},
    {"name": "H World (Huazhu)", "hk": "1179.HK", "adr": "HTHT", "ratio_guess": 10},
    {"name": "Hutchmed",         "hk": "0013.HK", "adr": "HCM",  "ratio_guess": 5},
    {"name": "Futu",     "hk": "3588.HK", "adr": "FUTU", "ratio_guess": 8},
    {"name": "GigaCloud","hk": "9860.HK", "adr": "GCT",  "ratio_guess": 1},
    
]


def get_close(ticker):
    d = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
    if d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    out = d[["Close"]].copy()
    out.columns = [ticker]
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


print("Downloading FX...")
usdhkd = get_close("HKD=X")          
usdhkd.columns = ["usdhkd"]

premiums, prices_hk, prices_adr, rows = {}, {}, {}, []

for p in ADR_PAIRS:
    name = p["name"]
    print(f"  {name:18s}", end=" ")
    try:
        hk = get_close(p["hk"])
        adr = get_close(p["adr"])
    except Exception as e:
        print(f"FAILED ({type(e).__name__})")
        continue
    if hk is None or adr is None:
        print("FAILED (no data)")
        continue

    df = pd.concat([hk, adr, usdhkd], axis=1, join="inner").dropna()
    df.columns = ["p_hk", "p_adr", "usdhkd"]
    if len(df) < 100:
        print(f"FAILED (only {len(df)} rows)")
        continue

    # ADR price restated in HKD so the two are comparable
    df["p_adr_hkd"] = df["p_adr"] * df["usdhkd"]

    # ---- Infer the conversion ratio from the data ----------
    imp = df["p_adr_hkd"] / df["p_hk"]
    implied = imp.median()
    ratio = round(implied)
    error = abs(implied - ratio) / ratio        # distance from a clean integer

    # ---- Did the ratio CHANGE mid-sample? ------------------
    # A stable-but-wrong ratio means the guess was wrong.
    # A drifting ratio means a corporate action split the sample.
    half = len(imp) // 2
    r_first = imp.iloc[:half].median()
    r_second = imp.iloc[half:].median()
    drift = abs(r_second - r_first) / r_first

    # Premium: ADR vs the equivalent bundle of HK shares.
    # Positive = New York more expensive than Hong Kong.
    df["premium"] = df["p_adr_hkd"] / (ratio * df["p_hk"]) - 1

    premiums[name] = df["premium"]
    prices_hk[name] = ratio * df["p_hk"]        # bundle comparable to 1 ADR
    prices_adr[name] = df["p_adr_hkd"]

    rows.append({
        "name": name, "n": len(df), "start": df.index.min().date(),
        "ratio_guess": p["ratio_guess"], "ratio_implied": implied,
        "ratio_used": ratio, "ratio_error": error,
        "r_first": r_first, "r_second": r_second, "drift": drift,
        "mean": df["premium"].mean(), "median": df["premium"].median(),
        "std": df["premium"].std(),
        "min": df["premium"].min(), "max": df["premium"].max(),
        "pct_neg": (df["premium"] < 0).mean(),
    })

    # Diagnose in one line: clean / ratio changed / guess wrong
    if error < 0.02:
        flag = ""
    elif drift > 0.05:
        flag = "   <-- RATIO CHANGED MID-SAMPLE"
    else:
        flag = "   <-- STABLE BUT NON-INTEGER (guess wrong)"
    print(f"OK  n={len(df):5d}  ratio~{implied:6.2f} (use {ratio:2d})  "
          f"drift={drift:5.1%}  mean={df['premium'].mean():+7.2%}{flag}")


s = pd.DataFrame(rows).set_index("name")

pct = "{:+.2%}".format
p3 = "{:.3%}".format
print("\nH-ADR PREMIUM SUMMARY")
print(s[["n", "start", "ratio_implied", "ratio_used", "ratio_error",
         "r_first", "r_second", "drift",
         "mean", "median", "std", "min", "max", "pct_neg"]].to_string(
      formatters={"mean": pct, "median": pct, "min": pct, "max": pct,
                  "std": "{:.2%}".format, "pct_neg": "{:.1%}".format,
                  "ratio_error": p3, "drift": p3}))

# ---- Clean subsample: ratio verified to <1% of an integer ----
# Rule-based exclusion, applied uniformly to every pair.
# Two conditions catch two different failures:
#   ratio_error -> the implied ratio isn't a clean integer (wrong guess)
#   drift       -> the ratio moved mid-sample (corporate action / artefact)
# GigaCloud passes the first (0.96%) and fails the second (80.4%).
clean = s[(s["ratio_error"] < 0.01) & (s["drift"] < 0.01)]

print("\n" + "=" * 70)
print("THE COMPARISON")
print("=" * 70)
print(f"  H-ADR, all {len(s)} pairs        : mean {s['mean'].mean():+.2%}"
      f"   std {s['std'].mean():.2%}")
print(f"  H-ADR, {len(clean)} verified pairs   : mean {clean['mean'].mean():+.2%}"
      f"   std {clean['std'].mean():.2%}")
print(f"  A-H,   63 pairs             : mean       +94.92%")
print("\n  Verified H-ADR pairs:", ", ".join(clean.index))
# Does the conclusion depend on where the threshold sits?
# If the mean stays near zero across thresholds, the filter isn't
# doing the work -- the data is.
print("\nSENSITIVITY TO THE EXCLUSION THRESHOLD")
for t in [0.005, 0.01, 0.02, 0.05, 1.0]:
    sub = s[(s["ratio_error"] < t) & (s["drift"] < t)]
    print(f"  threshold {t:5.1%}: {len(sub):2d} pairs   "
          f"mean {sub['mean'].mean():+7.2%}   std {sub['std'].mean():6.2%}")

pd.DataFrame(premiums).to_csv("data/processed/adr_premiums.csv")
pd.DataFrame(prices_hk).to_csv("data/processed/adr_prices_hk.csv")
pd.DataFrame(prices_adr).to_csv("data/processed/adr_prices_adr.csv")
s.to_csv("data/processed/adr_summary.csv")
print("\nSaved -> data/processed/adr_*.csv")