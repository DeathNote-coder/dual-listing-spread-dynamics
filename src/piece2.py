import yfinance as yf
import pandas as pd

def get_close(ticker):
    """"Download one ticker, return a clean single-column table of closes."""
    data = yf.download(
        ticker, 
        start = "2014-01-01",
        end = "2026-08-20",
        auto_adjust= True,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

        out = data[["Close"]]
        out.columns = [ticker]
        out.index = pd.to_datetime(out.index).tz_localize(None).normalize()

        return out

h = get_close("1398.HK") # ICBC, Hong Kong -> HKD
a = get_close("601398.SS") # ICBC, Shanghai -> CNY
usdcny = get_close("CNY=X") # yuan per US dollar
usdhkd = get_close("HKD=X") # HK dollar per US dollars

print("Rows downloaded: ")
print(f" Hong Kong : {len(h)}" )
print(f" Shanghai : {len(a)}" )
print(f" USDCNY : {len(usdcny)}")
print(f" USDHKD : {(len(usdhkd))}")

df = pd.concat([h, a, usdcny, usdhkd], axis = 1, join="inner")

df.columns = ["p_h", "p_a", "usdcny", "usdhkd"]

# Even after an inner join, you can get rows with missing values, since inner only checks intersection of indices
# Drop any row that isn't complete
df = df.dropna()

print(f"\nAfter joining all four: {len(df)} rows")
print(f"Lost {len(h) - len(df)} rows vs the Hong Kong series alone")

print("\nLast 5 rows:")
print(df.tail())
print(df.shape)

# ===========================================
# PIECE 3 - Convert currency and compute the premium
# ===========================================

# p_a is in yua,, p_h is in HK dollars. Different units.
# so comparing them is menaingless.
df["hkd_per_cny"] = df["usdhkd"]/df["usdcny"]

# HKD per CNY = (HKD per USD)/(CNY per CNY). The US dollar cancels out; it's just a common reference point.
# Restate the Shanghai price in Hong Kong dollars. So both listings are finally in the same currency.
df["p_a_hkd"] = df["p_a"]*df["hkd_per_cny"]

# The premium is how much more expensive is Shanghai?
df["premium"] = df["p_a_hkd"]/df["p_h"] -1
print("\nA-H PREMIUM")
print(f" mean {df['premium'].mean():+.2%}")
print(f" median {df['premium'].median():+.2%}")
print(f" std dev {df['premium'].std():.2%}")
print(f" min {df['premium'].min():+.2%} on {df['premium'].idxmin().date()}")
print(f" max {df['premium'].max():+.2%} on {df['premium'].idxmax().date()}")
cheaper = df["premium"] < 0
print(f" days Shanghai was cheaper: {cheaper.sum()} of {len(df)} ({cheaper.mean():.1%})")

# ===========================================
# PIECE 3 - Plot the Premium
# ===========================================

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex = True)

# Top: Both Prices, Same currency
ax1.plot(df.index, df["p_a_hkd"], label="Shanghai (in HKD)", lw=1)
ax1.plot(df.index, df["p_h"], label="Hong Kong", lw=1)
ax1.set_ylabel("Price (HKD)")
ax1.set_title("ICBC: one Company, two listings, two prices")
ax1.legend()
ax1.grid(alpha=0.3)

# Bottom: The gap between them
ax2.plot(df.index, df["premium"]*100, color="firebrick", lw=1)
ax2.axhline(0, color="black", lw=0.8, ls="--")          # law of one price
ax2.set_ylabel("A-H prmeium (%)")
ax2.set_xlabel("Date")
ax2.set_title("Shanghai Premium over Hong Kong (0% = one price)")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/icbc_premium.png", dpi=150)
plt.show

# ==================
# Side-Check
# ==================
print("\nMEAN PREMIUM BY YEAR")
print((df["premium"].groupby(df.index.year).mean()*100).round(1))

# Sanity check after 2018(Priori Hypothesis)
pre = df.loc[:"2017-12-31", "premium"]
post = df.loc["2018-01-01":, "premium"]
print(f"\n2014-2017 mean: {pre.mean():.2%} (n={len(pre)})")
print(f"2018-2026 mean: {post.mean():.2%} (n={len(post)})")

# EXPLORATORY (break date chosen after inspecting yearly means —
# not valid for inference; formal break dating comes later)
pre2 = df.loc[:"2020-12-31", "premium"]
post2 = df.loc["2021-01-01":, "premium"]
print(f"\n2014-2020 mean: {pre2.mean():.2%} (n={len(pre2)})")
print(f"2021-2026 mean: {post2.mean():.2%} (n={len(post2)})")

# ==========================================================
# PIECE 5 — Data quality guards
# ==========================================================

# ---- Guard 1: stale prices --------------------------------
# Chinese A-shares get suspended often, sometimes for weeks.
# Data providers forward-fill the last known price, so you get
# a flat line that LOOKS like data but is frozen. Meanwhile the
# HK twin keeps trading, so the premium goes on a journey that
# never actually happened.
def flag_stale(series, min_run=3):
    """True on days where the price hasn't moved for >= min_run days."""
    unchanged = series.diff() == 0          # did it change vs yesterday?
    block = (~unchanged).cumsum()           # label each run of unchanged days
    run_length = unchanged.groupby(block).transform("sum")
    return unchanged & (run_length >= min_run)

# ---- Guard 2: implausible jumps ---------------------------
# Real economics doesn't move 10 percentage points overnight and
# stay there. A jump this big is almost always a corporate action
# you haven't handled, or an ex-dividend date that fell on
# different days in the two markets.
def flag_jumps(premium, threshold=0.10):
    """True on dates where the premium moved more than the 'threshold'."""
    return premium.diff().abs() > threshold

df["stale_a"] = flag_stale(df["p_a"])
df["stale_h"] = flag_stale(df["p_h"])
df["jump"] = flag_jumps(df["premium"])

print("\nDATA QUALITY")
print(f" stale Shanghai days : {df['stale_a'].sum()}")
print(f" stale Hong Kong days : {df['stale_h'].sum()}")
print(f" premium jumps > 10pp : {df['jump'].sum()}")

if df["jump"].any():
    print("\n INSPECT THESE DAYS:")
    for d in df.index[df["jump"]]:
        print(f"{d.date()} premium {df.loc[d, 'premium']:+.1%}")


# Check flagged episodes
print(df.loc["2015-08-28":"2015-09-04", ["p_a", "p_h", "premium"]])
print(df.loc["2015-07-01":"2015-07-08", ["p_a", "p_h", "premium"]])