import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

ACCENT  = "#2D6A9F"
WARN    = "#E07B39"
GREEN   = "#27AE60"
PURPLE  = "#8E44AD"
DANGER  = "#C0392B"
BG      = "#F8F9FA"
GRID    = "#E0E0E0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
})

TICKERS = ["AAPL","MSFT","NVDA","JPM","GS","BAC","PG","KO","SPY","AGG"]
SECTORS = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology",
    "JPM":"Finance","GS":"Finance","BAC":"Finance",
    "PG":"Consumer","KO":"Consumer",
    "SPY":"ETF","AGG":"ETF"
}

# Regenerate synthetic data (same seed as analyse.py)
np.random.seed(42)
dates = pd.bdate_range("2020-01-01", "2025-01-01")
n = len(dates)

asset_params = {
    "AAPL": (0.32, 0.32), "MSFT": (0.28, 0.28), "NVDA": (0.55, 0.55),
    "JPM":  (0.18, 0.25), "GS":   (0.16, 0.26), "BAC":  (0.14, 0.28),
    "PG":   (0.10, 0.15), "KO":   (0.08, 0.14),
    "SPY":  (0.14, 0.18), "AGG":  (0.01, 0.05),
}
corr_matrix = np.array([
    [1.00,0.82,0.78,0.55,0.50,0.48,0.32,0.28,0.76,-0.05],
    [0.82,1.00,0.75,0.57,0.52,0.50,0.33,0.30,0.78,-0.04],
    [0.78,0.75,1.00,0.48,0.44,0.42,0.25,0.22,0.68,-0.08],
    [0.55,0.57,0.48,1.00,0.80,0.78,0.40,0.35,0.72,-0.10],
    [0.50,0.52,0.44,0.80,1.00,0.76,0.38,0.33,0.68,-0.09],
    [0.48,0.50,0.42,0.78,0.76,1.00,0.36,0.31,0.66,-0.11],
    [0.32,0.33,0.25,0.40,0.38,0.36,1.00,0.62,0.50, 0.05],
    [0.28,0.30,0.22,0.35,0.33,0.31,0.62,1.00,0.46, 0.06],
    [0.76,0.78,0.68,0.72,0.68,0.66,0.50,0.46,1.00,-0.02],
    [-0.05,-0.04,-0.08,-0.10,-0.09,-0.11,0.05,0.06,-0.02,1.00],
])
daily_means = np.array([p[0]/252 for p in asset_params.values()])
daily_vols  = np.array([p[1]/np.sqrt(252) for p in asset_params.values()])
L = np.linalg.cholesky(corr_matrix)
z = np.random.randn(n, 10)
correlated_z = z @ L.T
daily_returns_arr = daily_means + correlated_z * daily_vols
returns = pd.DataFrame(daily_returns_arr, index=dates, columns=TICKERS)

cov_matrix = returns.cov()
mean_returns = returns.mean()

# Helper: portfolio metrics
def portfolio_metrics(w, mean_ret, cov):
    ret  = (mean_ret @ w) * 252
    vol  = np.sqrt(w @ cov.values @ w) * np.sqrt(252)
    sr   = ret / vol
    port = (returns.values @ w)
    cum  = np.cumprod(1 + port)
    dd   = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
    mdd  = dd.min()
    return ret, vol, sr, mdd

# Baseline (equal weight)
w_baseline = np.array([0.10]*10)
b_ret, b_vol, b_sr, b_mdd = portfolio_metrics(w_baseline, mean_returns, cov_matrix)

# Portfolio A: Low volatility (risk-parity style) 
# Weight inversely proportional to individual asset volatility
inv_vol = 1 / returns.std()
w_A = (inv_vol / inv_vol.sum()).values

# Portfolio B: Max Sharpe (Markowitz)
def neg_sharpe(w):
    ret = (mean_returns @ w) * 252
    vol = np.sqrt(w @ cov_matrix.values @ w) * np.sqrt(252)
    return -(ret / vol)

constraints = [{"type":"eq","fun": lambda w: w.sum()-1}]
bounds = [(0.02, 0.30)] * 10   # 2%-30% per asset
res = minimize(neg_sharpe, w_baseline, method="SLSQP",
               bounds=bounds, constraints=constraints,
               options={"maxiter":1000})
w_B = res.x

# Portfolio C: Conservative (tilt bonds/consumer, cap tech)
w_C = np.array([0.06, 0.06, 0.04,   # tech - reduced
                0.10, 0.08, 0.08,   # finance - moderate
                0.12, 0.12,         # consumer - increased
                0.14, 0.20])        # SPY reduced, AGG increased (bonds)
w_C = w_C / w_C.sum()

portfolios = {
    "Baseline\n(Equal weight)": (w_baseline, WARN,   "--"),
    "Portfolio A\n(Low-vol)":   (w_A,        ACCENT, "-"),
    "Portfolio B\n(Max Sharpe)":(w_B,        GREEN,  "-"),
    "Portfolio C\n(Conservative)":(w_C,      PURPLE, "-"),
}

#Print scenario table
print(f"\n{'Portfolio':<25} {'Ann. Return':>12} {'Ann. Vol':>10} {'Sharpe':>8} {'Max DD':>10}")
print("-"*68)
metrics_store = {}
for name, (w, color, ls) in portfolios.items():
    r, v, s, d = portfolio_metrics(w, mean_returns, cov_matrix)
    metrics_store[name] = (r, v, s, d)
    label = name.replace('\n', ' ')
    print(f"{label:<25} {r:>11.2%} {v:>10.2%} {s:>8.2f} {d:>10.2%}")

# CHART 4: Cumulative returns comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: cumulative growth
ax = axes[0]
for name, (w, color, ls) in portfolios.items():
    port_daily = returns.values @ w
    cum = pd.Series(np.cumprod(1 + port_daily), index=dates)
    ax.plot(cum.index, cum * 100, color=color, linewidth=2,
            linestyle=ls, label=name.replace('\n',' '), zorder=4)

ax.axhline(100, color="#AAAAAA", linewidth=0.8, linestyle=":")
ax.set_title("Cumulative Growth of $100 - Portfolio Comparison",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Portfolio Value ($)", fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)

# Right: KPI bar chart
ax2 = axes[1]
names_short = ["Baseline", "Port A\nLow-vol", "Port B\nMax Sharpe", "Port C\nConservative"]
colors_bar   = [WARN, ACCENT, GREEN, PURPLE]
vols  = [metrics_store[k][1]*100 for k in portfolios]
rets  = [metrics_store[k][0]*100 for k in portfolios]

x = np.arange(len(names_short))
w_bar = 0.35
bars1 = ax2.bar(x - w_bar/2, vols, w_bar, label="Volatility (%)", color=colors_bar, alpha=0.85)
bars2 = ax2.bar(x + w_bar/2, rets, w_bar, label="Return (%)",
                color=colors_bar, alpha=0.45, edgecolor=colors_bar, linewidth=1.2)

# Target vol line
target = b_vol * 0.85 * 100
ax2.axhline(target, color=DANGER, linewidth=1.5, linestyle="--",
            label=f"Vol target: {target:.1f}%")

ax2.set_xticks(x); ax2.set_xticklabels(names_short, fontsize=8)
ax2.set_title("Volatility vs Return by Portfolio", fontsize=12, fontweight="bold")
ax2.set_ylabel("Annualised (%)", fontsize=10)
ax2.legend(fontsize=8)

# Value labels on bars
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
plt.savefig(r"C:\Users\Mateusz\Desktop\portfolio-risk-analysis\charts\chart4_portfoliocomparison.png",
            dpi=150, bbox_inches="tight")
plt.close()

# CHART 5: Monte Carlo simulation (1-year, best portfolio)
# Use Portfolio B (Max Sharpe) as the recommended portfolio
w_mc = w_B
port_daily = returns.values @ w_mc
mu    = port_daily.mean()
sigma = port_daily.std()

N_SIMS  = 500
N_DAYS  = 252
np.random.seed(0)
simulations = np.zeros((N_DAYS, N_SIMS))
for s in range(N_SIMS):
    daily = np.random.normal(mu, sigma, N_DAYS)
    simulations[:, s] = np.cumprod(1 + daily)

fig, ax = plt.subplots(figsize=(12, 5))

# Fan of simulations
for s in range(N_SIMS):
    ax.plot(simulations[:, s], color=GREEN, alpha=0.03, linewidth=0.8)

# Percentile bands
p5   = np.percentile(simulations, 5,  axis=1)
p25  = np.percentile(simulations, 25, axis=1)
p50  = np.percentile(simulations, 50, axis=1)
p75  = np.percentile(simulations, 75, axis=1)
p95  = np.percentile(simulations, 95, axis=1)

ax.fill_between(range(N_DAYS), p5,  p95, alpha=0.12, color=GREEN, label="5–95th pct")
ax.fill_between(range(N_DAYS), p25, p75, alpha=0.22, color=GREEN, label="25–75th pct")
ax.plot(p50, color=GREEN, linewidth=2.5, label=f"Median: ${p50[-1]*100:.0f}")
ax.plot(p5,  color=DANGER, linewidth=1.2, linestyle="--",
        label=f"5th pct (worst case): ${p5[-1]*100:.0f}")
ax.axhline(1.0, color="#AAAAAA", linewidth=0.8, linestyle=":")

ax.set_title(f"Monte Carlo Simulation - Portfolio B (Max Sharpe), {N_SIMS} paths, 1 year",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Trading days", fontsize=10)
ax.set_ylabel("Portfolio value (normalised to 1.0)", fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.set_xlim(0, N_DAYS)

plt.tight_layout()
plt.savefig(r"C:\Users\Mateusz\Desktop\portfolio-risk-analysis\charts\chart5_montecarlo.png",
            dpi=150, bbox_inches="tight")
plt.close()

# Export scenario table to CSV (for Google Sheets import)
rows = []
for name, (w, _, _) in portfolios.items():
    r, v, s, d = metrics_store[name]
    rows.append({
        "Portfolio": name.replace('\n',' '),
        "Annualised Return (%)": round(r*100, 2),
        "Annualised Volatility (%)": round(v*100, 2),
        "Sharpe Ratio": round(s, 3),
        "Max Drawdown (%)": round(d*100, 2),
        "Vol Reduction vs Baseline (%)": round((b_vol - v)/b_vol*100, 1),
        "Return Change vs Baseline (%)": round((r - b_ret)/b_ret*100, 1),
        "Meets Vol Target": "Yes" if v <= b_vol*0.85 else "No",
        "Meets Return Target (>-5%)": "Yes" if (r - b_ret)/b_ret*100 >= -5 else "No",
    })

scenario_df = pd.DataFrame(rows)
scenario_df.to_csv(r"C:\Users\Mateusz\Desktop\portfolio-risk-analysis\scenario_analysis.csv", index=False)
print("scenario_analysis.csv")

print("\nScenario summary:")
print(scenario_df.to_string(index=False))
