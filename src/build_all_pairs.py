# ==========================================================
# Build the A-H premium for every pair
# ==========================================================

"""
Universe
--------
CORE      : large, liquid A-H pairs (megacap SOEs, sector-spread)
EXTENDED  : the remainder of the 56-firm A-H universe catalogued in
            Pan, Li & Jarrett (2012), mostly smaller / less liquid names
 
Both tiers are downloaded together and tagged, so the tier becomes a
VARIABLE you can test on ("is the premium bigger in small caps?")
rather than a decision baked in before looking at the data.

remium definition
------------------
    premium_t = (P_A,t x HKD-per-CNY_t) / P_H,t - 1
"""

import yfinance as yf
import pandas as pd
import numpy as np

START = "2014-01-01"
END   = "2026-08-20"   # TODO(before report): make configurable

MIN_OBS = 100   # pairs with fewer usable rows than this are dropped

# ---- The universe -----------------------------------------
# A list of dictionaries. Each dict is one company: its name,
# its two tickers, and its sector. Sector matters because we
# want to test whether the premium is a banking phenomenon.
CORE = [
    # --- Banking ---
    {"name": "ICBC",                    "h": "1398.HK", "a": "601398.SS", "sector": "Banking"},
    {"name": "China Construction Bank", "h": "0939.HK", "a": "601939.SS", "sector": "Banking"},
    {"name": "Bank of China",           "h": "3988.HK", "a": "601988.SS", "sector": "Banking"},
    {"name": "Bank of Communications",  "h": "3328.HK", "a": "601328.SS", "sector": "Banking"},
    {"name": "China Merchants Bank",    "h": "3968.HK", "a": "600036.SS", "sector": "Banking"},
    {"name": "CITIC Bank",              "h": "0998.HK", "a": "601998.SS", "sector": "Banking"},
    {"name": "Minsheng Bank",           "h": "1988.HK", "a": "600016.SS", "sector": "Banking"},
 
    # --- Insurance ---
    {"name": "Ping An",                 "h": "2318.HK", "a": "601318.SS", "sector": "Insurance"},
    {"name": "China Life",              "h": "2628.HK", "a": "601628.SS", "sector": "Insurance"},
    {"name": "China Pacific Insurance", "h": "2601.HK", "a": "601601.SS", "sector": "Insurance"},
    {"name": "New China Life",          "h": "1336.HK", "a": "601336.SS", "sector": "Insurance"},
 
    # --- Brokerage ---
    {"name": "CITIC Securities",        "h": "6030.HK", "a": "600030.SS", "sector": "Brokerage"},
    {"name": "Huatai Securities",       "h": "6886.HK", "a": "601688.SS", "sector": "Brokerage"},
 
    # --- Energy ---
    {"name": "Sinopec",                 "h": "0386.HK", "a": "600028.SS", "sector": "Energy"},
    {"name": "PetroChina",              "h": "0857.HK", "a": "601857.SS", "sector": "Energy"},
    {"name": "CNOOC",                   "h": "0883.HK", "a": "600938.SS", "sector": "Energy"},
    {"name": "China Shenhua",           "h": "1088.HK", "a": "601088.SS", "sector": "Energy"},
    {"name": "China Coal Energy",       "h": "1898.HK", "a": "601898.SS", "sector": "Energy"},
    {"name": "Yanzhou Coal",            "h": "1171.HK", "a": "600188.SS", "sector": "Energy"},
 
    # --- Utilities ---
    {"name": "Huaneng Power",           "h": "0902.HK", "a": "600011.SS", "sector": "Utilities"},
 
    # --- Materials ---
    {"name": "Zijin Mining",            "h": "2899.HK", "a": "601899.SS", "sector": "Materials"},
    {"name": "Chalco",                  "h": "2600.HK", "a": "601600.SS", "sector": "Materials"},
    {"name": "Jiangxi Copper",          "h": "0358.HK", "a": "600362.SS", "sector": "Materials"},
    {"name": "Anhui Conch Cement",      "h": "0914.HK", "a": "600585.SS", "sector": "Materials"},
    {"name": "Angang Steel",            "h": "0347.HK", "a": "000898.SZ", "sector": "Materials"},
    {"name": "Maanshan Iron",           "h": "0323.HK", "a": "600808.SS", "sector": "Materials"},
    {"name": "Shanghai Petrochemical",  "h": "0338.HK", "a": "600688.SS", "sector": "Materials"},
 
    # --- Industrials ---
    {"name": "CRRC",                    "h": "1766.HK", "a": "601766.SS", "sector": "Industrials"},
    {"name": "China Railway Group",     "h": "0390.HK", "a": "601390.SS", "sector": "Industrials"},
    {"name": "China Comm. Construction","h": "1800.HK", "a": "601800.SS", "sector": "Industrials"},
    {"name": "Weichai Power",           "h": "2338.HK", "a": "000338.SZ", "sector": "Industrials"},
    {"name": "Guangshen Railway",       "h": "0525.HK", "a": "601333.SS", "sector": "Industrials"},
 
    # --- Autos ---
    {"name": "BYD",                     "h": "1211.HK", "a": "002594.SZ", "sector": "Autos"},
    {"name": "Great Wall Motor",        "h": "2333.HK", "a": "601633.SS", "sector": "Autos"},
    {"name": "Guangzhou Automobile",    "h": "2238.HK", "a": "601238.SS", "sector": "Autos"},
 
    # --- Airlines ---
    {"name": "Air China",               "h": "0753.HK", "a": "601111.SS", "sector": "Airlines"},
    {"name": "China Southern Airlines", "h": "1055.HK", "a": "600029.SS", "sector": "Airlines"},
    {"name": "China Eastern Airlines",  "h": "0670.HK", "a": "600115.SS", "sector": "Airlines"},
 
    # --- Technology / Consumer ---
    {"name": "ZTE",                     "h": "0763.HK", "a": "000063.SZ", "sector": "Technology"},
    {"name": "Tsingtao Brewery",        "h": "0168.HK", "a": "600600.SS", "sector": "Consumer"},
]
EXTENDED = [
    # --- Remainder of the Pan/Li/Jarrett (2012) A-H universe ---
    {"name": "Chenming Paper",          "h": "1812.HK", "a": "000488.SZ", "sector": "Materials"},
    {"name": "NE Electric",             "h": "0042.HK", "a": "000585.SZ", "sector": "Industrials"},
    {"name": "Jingwei Textile",         "h": "0350.HK", "a": "000666.SZ", "sector": "Industrials"},
    {"name": "Shandong Xinhua Pharma",  "h": "0719.HK", "a": "000756.SZ", "sector": "Healthcare"},
    {"name": "Guangdong Kelon",         "h": "0921.HK", "a": "000921.SZ", "sector": "Consumer"},
    {"name": "Anhui Expressway",        "h": "0995.HK", "a": "600012.SS", "sector": "Infrastructure"},
    {"name": "China Shipping Dev",      "h": "1138.HK", "a": "600026.SS", "sector": "Shipping"},
    {"name": "Huadian Power",           "h": "1071.HK", "a": "600027.SS", "sector": "Utilities"},
    {"name": "Guangzhou Pharmaceutical","h": "0874.HK", "a": "600332.SS", "sector": "Healthcare"},
    {"name": "Jiangsu Expressway",      "h": "0177.HK", "a": "600377.SS", "sector": "Infrastructure"},
    {"name": "Shenzhen Expressway",     "h": "0548.HK", "a": "600548.SS", "sector": "Infrastructure"},
    {"name": "Guangzhou Shipyard",      "h": "0317.HK", "a": "600685.SS", "sector": "Industrials"},
    {"name": "Nanjing Panda",           "h": "0553.HK", "a": "600775.SS", "sector": "Technology"},
    {"name": "Jiaoda Hightech",         "h": "0300.HK", "a": "600806.SS", "sector": "Technology"},
    {"name": "Beiren Printing",         "h": "0187.HK", "a": "600860.SS", "sector": "Industrials"},
    {"name": "Yizheng Chemical",        "h": "1033.HK", "a": "600871.SS", "sector": "Materials"},
    {"name": "Tianjin Capital Env.",    "h": "1065.HK", "a": "600874.SS", "sector": "Utilities"},
    {"name": "Dongfang Electric",       "h": "1072.HK", "a": "600875.SS", "sector": "Industrials"},
    {"name": "Luoyang Glass",           "h": "1108.HK", "a": "600876.SS", "sector": "Materials"},
    {"name": "Chongqing Iron & Steel",  "h": "1053.HK", "a": "601005.SS", "sector": "Materials"},
    {"name": "China Railway Const.",    "h": "1186.HK", "a": "601186.SS", "sector": "Industrials"},
    {"name": "Beijing North Star",      "h": "0588.HK", "a": "601588.SS", "sector": "Real Estate"},
    {"name": "China Oilfield Services", "h": "2883.HK", "a": "601808.SS", "sector": "Energy"},
    {"name": "COSCO Shipping",          "h": "1919.HK", "a": "601919.SS", "sector": "Shipping"},
    {"name": "China Shipping Container","h": "2866.HK", "a": "601866.SS", "sector": "Shipping"},
    {"name": "Datang Power",            "h": "0991.HK", "a": "601991.SS", "sector": "Utilities"},
]

# Tag each pair with its tier, then merge them into ONE list the roop runs over.

for p in CORE:
    p["tier"] = "core"
for p in EXTENDED:
    p["tier"] = "extended"

PAIRS = CORE + EXTENDED

# ----------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------
def get_close(ticker):
    """Download one ticker. Return a one-column table of closes, or None."""
    data = yf.download(ticker, start=START, end=END,
                       auto_adjust=False, progress=False)
    if data.empty:
        return None
 
    # yfinance returns stacked (Price / Ticker) headers even for one
    # ticker. Flatten so "Close" is just "Close".
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
 
    out = data[["Close"]].copy()
    out.columns = [ticker]
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out
 
# ----------------------------------------------------------
# QUALITY GUARDS  (from Piece 5)
# ----------------------------------------------------------
def flag_stale(series, min_run=3):
    """True on days where the price hasn't moved for >= min_run days."""
    unchanged = series.diff() == 0
    block = (~unchanged).cumsum()
    run_length = unchanged.groupby(block).transform("sum")
    return unchanged & (run_length >= min_run)
 
 
def flag_jumps(premium, threshold=0.10):
    """Flag days the price RATIO moved >10% — scale-free."""
    log_ratio = np.log(1 + premium)
    return log_ratio.diff().abs() > threshold
 
 # --------------------------------------------------------
 # FX (identical for every pair -> download once)
 # --------------------------------------------------------

print("Donwloading FX...")
usdcny = get_close("CNY=X")   # yuan per US dollar
usdhkd = get_close("HKD=X")   # HK dollars per US dollar

fx = pd.concat([usdcny, usdhkd], axis = 1, join="inner").dropna()
fx.columns = ["usdcny", "usdhkd"]
fx["hkd_per_cny"] = fx["usdhkd"]/fx["usdcny"]
print(f" FX rows {len(fx)}\n")

# ----------------------------------------------------------
# BUILD ONE PAIR
# ----------------------------------------------------------
def build_pair(pair):
    """Return a DataFrame for one pair (premium + guards), or None."""
    h = get_close(pair["h"])
    a = get_close(pair["a"])
    if h is None or a is None:
        return None
    
     # INNER join: keep only dates present in HK, Shanghai AND FX.
    # This is what removes Golden Week, Chinese New Year, Easter,
    # typhoon closures etc. No forward-filling: inventing a price for
    # a day a market was shut is how fake results are born.
    df = pd.concat([h, a, fx["hkd_per_cny"]], axis=1, join="inner").dropna()
    df.columns = ["p_h", "p_a", "hkd_per_cny"]
 
    df["p_a_hkd"] = df["p_a"] * df["hkd_per_cny"]
    df["premium"] = df["p_a_hkd"] / df["p_h"] - 1
 
    df["stale_a"] = flag_stale(df["p_a"])
    df["stale_h"] = flag_stale(df["p_h"])
    df["jump"] = flag_jumps(df["premium"])
    return df

# ----------------------------------------------------------
# RUN EVERY PAIR
# ----------------------------------------------------------
premiums = {}   # name -> premium Series
rows = []       # one summary dict per successful pair
failures = []   # (name, reason) for pairs that didn't make it
prices_a = {}
prices_h = {}

print(f"Building {len(PAIRS)} pairs... \n")

for pair in PAIRS:
    name = pair["name"]
    print(f" {name:28s}", end = " ")

    # If ANY pair blows up, catch it and keep going. With 65 pairs,
    # one dead ticker should not kill the whole run.
    try:
        df = build_pair(pair)
    except Exception as e:
        print(f"FAILED ({type(e).__name__})")
        failures.append((name, str(e)[:60]))
        continue
    if df is None or len(df) < MIN_OBS:
        n =0 if df is None else len(df)
        print(f"FAILED (only {n} rows)")
        failures.append((name, f"only {n} rows"))
        continue

    p = df["premium"]
    premiums[name] = p
    prices_a[name] = df["p_a_hkd"]
    prices_h[name] = df["p_h"]
    rows.append({
        "name": name,
        "tier": pair["tier"],
        "sector": pair["sector"],
        "n": len(p),
        "start": p.index.min().date(),
        "mean": p.mean(),
        "median": p.median(),
        "std": p.std(),
        "min": p.min(),
        "max": p.max(),
        "pct_neg": (p < 0).mean(),
        "stale_a": int(df["stale_a"].sum()),
        "stale_h": int(df["stale_h"].sum()),
        "jumps": int(df["jump"].sum()),

    })

    print(f"OK n={len(p): 5d} mean={p.mean():+7.2%}")
     
# ----------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------
print("\n" + "=" * 78)
print(f"SUCCEEDED: {len(rows)} / {len(PAIRS)}      FAILED: {len(failures)}")
print("=" * 78)

if failures:
    print("\nFAILURES (check these tickers):")
    for name, reason in failures:
        print(f" {name:28s} {reason}")

summary = pd.DataFrame(rows).sort_values("mean", ascending=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 100)

pct= "{:+.1%}".format
fmt = {"mean": pct, "median": pct, "min": pct, "max": pct,
       "std": "{:.1%}".format, "pct_neg": "{:.1%}".format}

print("\nA-H PREMIUM BY PAIR")
print(summary.to_string(index=False, formatters=fmt))

print("\nBY SECTOR")
by_sector = summary.groupby("sector")["mean"].agg(["count", "mean", "median"])
by_sector[["mean", "median"]] *= 100
print(by_sector.round(1))

print("\nBY TIER  (does size / liquidity matter?)")
by_tier = summary.groupby("tier")["mean"].agg(["count", "mean", "median"])
by_tier[["mean", "median"]] *= 100
print(by_tier.round(1))

print("\nCROSS-SECTIONAL DISTRIBUTION OF THE MEAN PREMIUM")
print(f"  pairs                  : {len(summary)}")
print(f"  mean of means          : {summary['mean'].mean():+.2%}")
print(f"  median of means        : {summary['mean'].median():+.2%}")
print(f"  pairs with mean > 0    : {(summary['mean'] > 0).sum()} / {len(summary)}")
print(f"  smallest mean premium  : {summary['mean'].min():+.2%}"
      f"  ({summary.loc[summary['mean'].idxmin(), 'name']})")
print(f"  largest mean premium   : {summary['mean'].max():+.2%}"
      f"  ({summary.loc[summary['mean'].idxmax(), 'name']})")

# ----------------------------------------------------------
# SAVE
# ----------------------------------------------------------
# ----------------------------------------------------------
# SAVE
# ----------------------------------------------------------
panel = pd.DataFrame(premiums)
panel.to_csv("data/processed/ah_premiums.csv")
summary.to_csv("data/processed/ah_summary.csv", index=False)
pd.DataFrame(prices_a).to_csv("data/processed/prices_a_hkd.csv")
pd.DataFrame(prices_h).to_csv("data/processed/prices_h.csv")

quality = summary[["name", "stale_a", "stale_h", "jumps", "n"]].copy()
quality["stale_pct"] = ((quality["stale_a"] + quality["stale_h"])
                         / (2 * quality["n"]) * 100).round(2)
quality = quality.sort_values("stale_pct", ascending=False)
quality.to_csv("data/processed/ah_quality_flags.csv", index=False)

print(f"\nSaved panel  : {panel.shape[0]} dates x {panel.shape[1]} pairs")
print( "               -> data/processed/ah_premiums.csv")
print( "               -> data/processed/ah_summary.csv")
print( "               -> data/processed/ah_quality_flags.csv")
