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

