import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

ACCENT = "#2D6A9F"; WARN = "#E07B39"; GREEN = "#27AE60"
PURPLE = "#8E44AD"; DANGER = "#C0392B"; BG = "#F8F9FA"; GRID = "#E0E0E0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": BG, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})

TICKERS = ["AAPL","MSFT","NVDA","JPM","GS","BAC","PG","KO","SPY","AGG"]

# Regenerate data (same seed)
np.random.seed(42)
dates = pd.bdate_range("2020-01-01", "2025-01-01")
n = len(dates)
asset_params = {
    "AAPL":(0.32,0.32),"MSFT":(0.28,0.28),"NVDA":(0.55,0.55),
    "JPM":(0.18,0.25),"GS":(0.16,0.26),"BAC":(0.14,0.28),
    "PG":(0.10,0.15),"KO":(0.08,0.14),"SPY":(0.14,0.18),"AGG":(0.01,0.05),
}
corr_matrix = np.array([
    [1.00,0.82,0.78,0.55,0.50,0.48,0.32,0.28,0.76,-0.05],
    [0.82,1.00,0.75,0.57,0.52,0.50,0.33,0.30,0.78,-0.04],
    [0.78,0.75,1.00,0.48,0.44,0.42,0.25,0.22,0.68,-0.08],
    [0.55,0.57,0.48,1.00,0.80,0.78,0.40,0.35,0.72,-0.10],
    [0.50,0.52,0.44,0.80,1.00,0.76,0.38,0.33,0.68,-0.09],
    [0.48,0.50,0.42,0.78,0.76,1.00,0.36,0.31,0.66,-0.11],
    [0.32,0.33,0.25,0.40,0.38,0.36,1.00,0.62,0.50,0.05],
    [0.28,0.30,0.22,0.35,0.33,0.31,0.62,1.00,0.46,0.06],
    [0.76,0.78,0.68,0.72,0.68,0.66,0.50,0.46,1.00,-0.02],
    [-0.05,-0.04,-0.08,-0.10,-0.09,-0.11,0.05,0.06,-0.02,1.00],
])
daily_means = np.array([p[0]/252 for p in asset_params.values()])
daily_vols  = np.array([p[1]/np.sqrt(252) for p in asset_params.values()])
L = np.linalg.cholesky(corr_matrix)
z = np.random.randn(n, 10)
daily_returns_arr = daily_means + (z @ L.T) * daily_vols
returns = pd.DataFrame(daily_returns_arr, index=dates, columns=TICKERS)
cov_matrix = returns.cov(); mean_returns = returns.mean()

from scipy.optimize import minimize
w_baseline = np.array([0.10]*10)
def neg_sharpe(w):
    r = (mean_returns @ w)*252; v = np.sqrt(w @ cov_matrix.values @ w)*np.sqrt(252)
    return -(r/v)
res = minimize(neg_sharpe, w_baseline, method="SLSQP",
               bounds=[(0.02,0.30)]*10,
               constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
               options={"maxiter":1000})
w_opt = res.x

# Simulate "live" monitoring - last 6 months as the "control" period
control_returns = returns.iloc[-126:]  # ~6 months
port_daily_ctrl = control_returns.values @ w_opt
roll_vol_ctrl   = pd.Series(port_daily_ctrl, index=control_returns.index).rolling(21).std() * np.sqrt(252) * 100
roll_ret_ctrl   = pd.Series(port_daily_ctrl, index=control_returns.index).rolling(21).mean() * 252 * 100
cum_ctrl        = np.cumprod(1 + port_daily_ctrl)

b_vol = (returns.values @ w_baseline).std() * np.sqrt(252) * 100
b_ret = (returns.values @ w_baseline).mean() * 252 * 100
target_vol = b_vol * 0.85
target_ret = b_ret * 0.95  # must not lose more than 5%

opt_vol = (returns.values @ w_opt).std() * np.sqrt(252) * 100
opt_ret = (returns.values @ w_opt).mean() * 252 * 100
opt_sr  = opt_ret / opt_vol
opt_mdd = ((np.cumprod(1 + returns.values @ w_opt) /
            np.maximum.accumulate(np.cumprod(1 + returns.values @ w_opt))) - 1).min() * 100

#RAG status helper
def rag(value, green_thresh, amber_thresh, higher_is_better=True):
    if higher_is_better:
        if value >= green_thresh: return "GREEN",  "#27AE60"
        if value >= amber_thresh: return "AMBER",  "#E07B39"
        return "RED", "#C0392B"
    else:
        if value <= green_thresh: return "GREEN",  "#27AE60"
        if value <= amber_thresh: return "AMBER",  "#E07B39"
        return "RED", "#C0392B"

kpis = [
    ("Annualised\nVolatility",  f"{opt_vol:.1f}%",  f"Target: <{target_vol:.1f}%",
     *rag(opt_vol, target_vol, b_vol, higher_is_better=False)),
    ("Annualised\nReturn",      f"{opt_ret:.1f}%",  f"Target: >{target_ret:.1f}%",
     *rag(opt_ret, target_ret*1.05, target_ret, higher_is_better=True)),
    ("Sharpe\nRatio",           f"{opt_sr:.2f}",    "Target: >1.20",
     *rag(opt_sr, 1.20, 1.00, higher_is_better=True)),
    ("Max\nDrawdown",           f"{opt_mdd:.1f}%",  "Target: >-25%",
     *rag(opt_mdd, -25, -35, higher_is_better=True)),
]

#CHART 6: Monitoring dashboard prototype
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("white")

# Title bar
fig.text(0.5, 0.97, "Portfolio Risk Monitoring Dashboard - DMAIC Control Phase",
         ha="center", va="top", fontsize=16, fontweight="bold", color="#1A1A2E")
fig.text(0.5, 0.935, "Portfolio B (Max Sharpe Optimisation)  |  Reporting period: Jul 2024 - Jan 2025",
         ha="center", va="top", fontsize=10, color="#555555")

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4,
                       top=0.90, bottom=0.06, left=0.06, right=0.97)

#Row 1: RAG KPI cards
for i, (label, value, target, status, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # Card background
    card = FancyBboxPatch((0.05, 0.05), 0.90, 0.90,
                          boxstyle="round,pad=0.02",
                          facecolor=color+"22", edgecolor=color, linewidth=2)
    ax.add_patch(card)

    # RAG badge
    badge = FancyBboxPatch((0.62, 0.72), 0.32, 0.22,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor="none")
    ax.add_patch(badge)
    ax.text(0.78, 0.83, status, ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

    ax.text(0.50, 0.62, label, ha="center", va="center",
            fontsize=9, color="#333333", fontweight="500")
    ax.text(0.50, 0.38, value, ha="center", va="center",
            fontsize=18, fontweight="bold", color=color)
    ax.text(0.50, 0.18, target, ha="center", va="center",
            fontsize=7.5, color="#777777")

#Row 2: Rolling vol + cumulative return
ax_vol = fig.add_subplot(gs[1, :2])
ax_vol.plot(roll_vol_ctrl.index, roll_vol_ctrl, color=ACCENT, linewidth=2, label="30d rolling vol")
ax_vol.axhline(target_vol, color=GREEN,  linewidth=1.4, linestyle="--", label=f"Target: {target_vol:.1f}%")
ax_vol.axhline(b_vol,      color=DANGER, linewidth=1.2, linestyle=":",  label=f"Baseline: {b_vol:.1f}%")
ax_vol.fill_between(roll_vol_ctrl.index, roll_vol_ctrl, target_vol,
                    where=(roll_vol_ctrl > target_vol), alpha=0.15, color=WARN)
ax_vol.set_title("Rolling 21-day Volatility (Annualised)", fontsize=10, fontweight="bold")
ax_vol.set_ylabel("Vol (%)", fontsize=9)
ax_vol.legend(fontsize=8); ax_vol.tick_params(labelsize=8)

ax_ret = fig.add_subplot(gs[1, 2:])
ax_ret.plot(control_returns.index, cum_ctrl * 100, color=GREEN, linewidth=2)
ax_ret.fill_between(control_returns.index, 100, cum_ctrl * 100,
                    where=(cum_ctrl * 100 >= 100), alpha=0.15, color=GREEN)
ax_ret.fill_between(control_returns.index, 100, cum_ctrl * 100,
                    where=(cum_ctrl * 100 < 100), alpha=0.15, color=DANGER)
ax_ret.axhline(100, color="#AAAAAA", linewidth=0.8, linestyle=":")
ax_ret.set_title("Cumulative Return - Control Period", fontsize=10, fontweight="bold")
ax_ret.set_ylabel("Value ($, start = 100)", fontsize=9)
ax_ret.tick_params(labelsize=8)

#Row 3: Weight allocation bar + weight table
ax_wt = fig.add_subplot(gs[2, :2])
SECTOR_COLORS = {"Technology":ACCENT,"Finance":WARN,"Consumer":GREEN,"ETF":PURPLE}
SECTORS = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology",
    "JPM":"Finance","GS":"Finance","BAC":"Finance",
    "PG":"Consumer","KO":"Consumer","SPY":"ETF","AGG":"ETF"
}
colors_bar = [SECTOR_COLORS[SECTORS[t]] for t in TICKERS]
bars = ax_wt.bar(TICKERS, w_opt * 100, color=colors_bar, edgecolor="white", linewidth=0.5)
ax_wt.axhline(10, color=WARN, linewidth=1, linestyle="--", alpha=0.6, label="Baseline (10% each)")
ax_wt.set_title("Optimised Portfolio Weights", fontsize=10, fontweight="bold")
ax_wt.set_ylabel("Weight (%)", fontsize=9)
ax_wt.tick_params(labelsize=8)
for bar, w in zip(bars, w_opt):
    ax_wt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               f"{w*100:.1f}%", ha="center", va="bottom", fontsize=7)
legend_handles = [mpatches.Patch(color=c, label=s) for s,c in SECTOR_COLORS.items()]
ax_wt.legend(handles=legend_handles, fontsize=7.5, loc="upper right")

# Summary table
ax_tbl = fig.add_subplot(gs[2, 2:])
ax_tbl.axis("off")
table_data = [
    ["KPI", "Baseline", "Portfolio B", "Target", "Status"],
    ["Ann. Volatility", f"{b_vol:.1f}%", f"{opt_vol:.1f}%", f"<{target_vol:.1f}%", "✓ GREEN"],
    ["Ann. Return",     f"{b_ret:.1f}%", f"{opt_ret:.1f}%", f">{target_ret:.1f}%", "✓ GREEN"],
    ["Sharpe Ratio",    f"{(b_ret/b_vol):.2f}",  f"{opt_sr:.2f}",  ">1.20",  "✓ GREEN"],
    ["Max Drawdown",    f"{((np.cumprod(1+returns.values@w_baseline)/np.maximum.accumulate(np.cumprod(1+returns.values@w_baseline)))-1).min()*100:.1f}%",
                        f"{opt_mdd:.1f}%", ">-25%", "✓ GREEN"],
]
tbl = ax_tbl.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
tbl.scale(1, 1.6)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#DDDDDD")
    if row == 0:
        cell.set_facecolor("#2D6A9F"); cell.set_text_props(color="white", fontweight="bold")
    elif col == 4:
        cell.set_facecolor("#D5F0DC"); cell.set_text_props(color="#1A6B3A", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#F0F4FA")
    else:
        cell.set_facecolor("white")
ax_tbl.set_title("KPI Summary Table", fontsize=10, fontweight="bold", pad=12)

plt.savefig(r"C:\Users\Mateusz\Desktop\portfolio-risk-analysis\charts\chart6_controldashboard.png",
            dpi=150, bbox_inches="tight")
plt.close()
