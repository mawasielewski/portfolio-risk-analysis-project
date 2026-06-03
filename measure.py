import yfinance as yf
import pandas as pd

TICKERS = [
    "AAPL", "MSFT", "NVDA",        #Tech
    "JPM", "GS", "BAC",            #Finance
    "PG", "KO",                    #Consumer Goods
    "SPY", "AGG"                   #ETFs (market + bonds)
]

#5 years of daily closing prices
raw = yf.download(TICKERS, start="2020-01-01", end="2025-01-01", auto_adjust=True)["Close"]

print(raw.shape)
print(raw.head())

#data checks
print(raw.isnull().sum())

print("From:", raw.index.min())
print("To:  ", raw.index.max())
print("Total trading days:", len(raw))

print("\n=== Price Std Dev (sanity check) ===")
print(raw.std().round(2))

#export to csv
quality_log = pd.DataFrame({
    "Ticker": raw.columns,
    "Sector": ["Technology", "ETF", "Finance", "Finance", "Finance", 
                "Consumer", "Technology", "Technology", "Consumer", "ETF"],
    "Missing Values": raw.isnull().sum().values,
    "Std Dev": raw.std().round(2).astype(str).values,
    "Status": "Pass"
})

quality_log.to_csv("data_quality_log.csv", index=False)
print(quality_log)

#KPI CALCULATION
# Calculate daily returns (% change day to day)
returns = raw.pct_change().dropna()

# Assume equal weight across all 10 assets
weights = pd.Series([1/10] * 10, index=raw.columns)

# Daily portfolio return (weighted average of each asset's daily return)
portfolio_daily_return = (returns * weights).sum(axis=1)

# Annualised return (252 trading days in a year)
annualised_return = portfolio_daily_return.mean() * 252

# Annualised volatility
annualised_volatility = portfolio_daily_return.std() * (252 ** 0.5)

# Sharpe ratio (we assume risk-free rate = 0 for simplicity)
sharpe_ratio = annualised_return / annualised_volatility

# Max drawdown
cumulative = (1 + portfolio_daily_return).cumprod()
rolling_max = cumulative.cummax()
drawdown = (cumulative - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print("Baseline Portfolio KPIs")
print(f"Annualised Return:     {annualised_return:.2%}")
print(f"Annualised Volatility: {annualised_volatility:.2%}")
print(f"Sharpe Ratio:          {sharpe_ratio:.2f}")
print(f"Max Drawdown:          {max_drawdown:.2%}")