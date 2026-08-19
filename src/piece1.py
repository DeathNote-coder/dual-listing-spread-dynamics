import yfinance as yf

# ICBC Hong Kong listing; dates chosen to cover the study period.

df = yf.download(
    "1398.HK",
    start="2014-01-01",
    end="2026-08-20",
    auto_adjust=True,
)

print(df.tail())
print(df.shape)