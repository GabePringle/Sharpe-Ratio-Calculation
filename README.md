import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Set risk-free rate (e.g., 4%)
risk_free_rate = 0.04  # 4% per year

# Define tickers and time range
tickers = ['SPY', 'TLT']
start_date = "2015-01-01"
end_date = "2025-03-24"

# Download data
df = yf.download(tickers, start=start_date, end=end_date, interval="1d")['Close']

# Store results
results = {}

# Loop through each asset
for ticker in tickers:
    closing_prices = df[ticker].dropna().values.flatten()
    
    # Daily returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    
    # Daily stats
    mean_daily_return = np.mean(daily_returns)
    std_daily_return = np.std(daily_returns)
    
    # Annualize
    annualized_return = (1 + mean_daily_return)**252 - 1
    annualized_volatility = std_daily_return * np.sqrt(252)
    
    # Sharpe ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    # Store results
    results[ticker] = {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio
    }

# Display results
print("\n--- Risk Metrics ---")
for ticker in tickers:
    print(f"\n{ticker}:")
    print(f"Annualized Return: {results[ticker]['Annualized Return']:.2%}")
    print(f"Annualized Volatility: {results[ticker]['Annualized Volatility']:.2%}")
    print(f"Sharpe Ratio (RF={risk_free_rate:.2%}): {results[ticker]['Sharpe Ratio']:.4f}")

# Optional: Plot comparison
labels = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio']
x = np.arange(len(labels))
width = 0.35

spy_vals = [results['SPY'][label] for label in labels]
tlt_vals = [results['TLT'][label] for label in labels]

plt.figure(figsize=(10,6))
plt.bar(x - width/2, spy_vals, width, label='SPY')
plt.bar(x + width/2, tlt_vals, width, label='TLT')
plt.xticks(x, labels)
plt.ylabel("Value")
plt.title("SPY vs TLT Risk Metrics")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Sharpe-Ratio-Calculation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Set risk-free rate and weights
risk_free_rate = 0.04  # 4% annual
weights = np.array([0.6, 0.4])  # 60% AAPL, 40% MSFT

# Download data from 2015 to 2025
tickers = ['AAPL', 'MSFT']
df = yf.download(tickers, start="2015-01-01", end="2025-03-24", interval="1d")['Close'].dropna()

# Calculate daily returns
daily_returns = df.pct_change().dropna()

# Portfolio daily returns
portfolio_daily_returns = daily_returns.dot(weights)

# Daily stats
mean_daily_return = portfolio_daily_returns.mean()
std_daily_return = portfolio_daily_returns.std()

# Annualize
annualized_return = (1 + mean_daily_return)**252 - 1
annualized_volatility = std_daily_return * np.sqrt(252)
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

# Print results
print("\n--- 60/40 AAPL-MSFT Portfolio Metrics (2015–2025) ---")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_volatility:.2%}")
print(f"Sharpe Ratio (RF={risk_free_rate:.2%}): {sharpe_ratio:.4f}")
