import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

#Shared style
ACCENT   = "#2D6A9F"
WARN     = "#E07B39"
DANGER   = "#C0392B"
BG       = "#F8F9FA"
GRID     = "#E0E0E0"
FONT     = "DejaVu Sans"

plt.rcParams.update({
    "font.family": FONT,
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
SECTOR_COLORS = {
    "Technology":"#2D6A9F","Finance":"#E07B39",
    "Consumer":"#27AE60","ETF":"#8E44AD"
}

print("Generating synthetic data calibrated to baseline KPIs…")
np.random.seed(42)
dates = pd.bdate_range("2020-01-01", "2025-01-01")
n = len(dates)

# Calibrated annualised means and vols per asset (matching real-world profile)
asset_params = {
    #         ann_ret  ann_vol
    "AAPL":  (0.32,    0.32),
    "MSFT":  (0.28,    0.28),
    "NVDA":  (0.55,    0.55),
    "JPM":   (0.18,    0.25),
    "GS":    (0.16,    0.26),
    "BAC":   (0.14,    0.28),
    "PG":    (0.10,    0.15),
    "KO":    (0.08,    0.14),
    "SPY":   (0.14,    0.18),
    "AGG":   (0.01,    0.05),
}

# Correlation structure (block: high within tech/finance, low cross)
corr_matrix = np.array([
    #AAPL  MSFT  NVDA  JPM   GS    BAC   PG    KO    SPY   AGG
    [1.00, 0.82, 0.78, 0.55, 0.50, 0.48, 0.32, 0.28, 0.76, -0.05],  # AAPL
    [0.82, 1.00, 0.75, 0.57, 0.52, 0.50, 0.33, 0.30, 0.78, -0.04],  # MSFT
    [0.78, 0.75, 1.00, 0.48, 0.44, 0.42, 0.25, 0.22, 0.68, -0.08],  # NVDA
    [0.55, 0.57, 0.48, 1.00, 0.80, 0.78, 0.40, 0.35, 0.72, -0.10],  # JPM
    [0.50, 0.52, 0.44, 0.80, 1.00, 0.76, 0.38, 0.33, 0.68, -0.09],  # GS
    [0.48, 0.50, 0.42, 0.78, 0.76, 1.00, 0.36, 0.31, 0.66, -0.11],  # BAC
    [0.32, 0.33, 0.25, 0.40, 0.38, 0.36, 1.00, 0.62, 0.50,  0.05],  # PG
    [0.28, 0.30, 0.22, 0.35, 0.33, 0.31, 0.62, 1.00, 0.46,  0.06],  # KO
    [0.76, 0.78, 0.68, 0.72, 0.68, 0.66, 0.50, 0.46, 1.00, -0.02],  # SPY
    [-0.05,-0.04,-0.08,-0.10,-0.09,-0.11, 0.05, 0.06,-0.02,  1.00], # AGG
])

# Cholesky decomposition to generate correlated returns
daily_means = np.array([p[0]/252 for p in asset_params.values()])
daily_vols  = np.array([p[1]/np.sqrt(252) for p in asset_params.values()])

L = np.linalg.cholesky(corr_matrix)
z = np.random.randn(n, 10)
correlated_z = z @ L.T
daily_returns_arr = daily_means + correlated_z * daily_vols

returns = pd.DataFrame(daily_returns_arr, index=dates, columns=TICKERS)

# Build price series from returns
prices_arr = 100 * np.cumprod(1 + daily_returns_arr, axis=0)
raw = pd.DataFrame(prices_arr, index=dates, columns=TICKERS)

weights = np.array([1/10]*10)
port_ret = returns.values @ weights

#CHART 1: Correlation heatmap
corr = returns.corr()

# Sort tickers by sector for a cleaner heatmap
sector_order = sorted(TICKERS, key=lambda t: (SECTORS[t], t))
corr = corr.loc[sector_order, sector_order]

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("white")

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True  # show lower triangle + diag

sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlGn", vmin=-1, vmax=1,
    linewidths=0.5, linecolor="#DDDDDD",
    annot_kws={"size": 9, "weight": "500"},
    ax=ax, square=True,
    cbar_kws={"shrink": 0.7, "label": "Pearson correlation"}
)

# Colour tick labels by sector
for tick, label in zip(ax.get_xticklabels(), corr.columns):
    tick.set_color(SECTOR_COLORS[SECTORS[label]])
    tick.set_fontweight("bold")
for tick, label in zip(ax.get_yticklabels(), corr.index):
    tick.set_color(SECTOR_COLORS[SECTORS[label]])
    tick.set_fontweight("bold")

# Sector legend
legend_handles = [
    mpatches.Patch(color=c, label=s)
    for s, c in SECTOR_COLORS.items()
]
ax.legend(handles=legend_handles, loc="upper right",
          framealpha=0.9, fontsize=9, title="Sector", title_fontsize=9)

ax.set_title("Asset Correlation Matrix (Baseline Portfolio)",
             fontsize=14, fontweight="bold", pad=16)
ax.set_xlabel(""); ax.set_ylabel("")

# Highlight high-correlation cells (≥0.75) with a red border
for i in range(len(corr)):
    for j in range(i):          # lower triangle only
        if corr.iloc[i, j] >= 0.75:
            ax.add_patch(plt.Rectangle(
                (j, i), 1, 1, fill=False,
                edgecolor=DANGER, linewidth=2, zorder=5
            ))

plt.tight_layout()
plt.savefig(r"C:\Users\Mateusz\Desktop\portfolio-risk-analysis\charts\chart1_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

#CHART 2: Rolling volatility
port_series = pd.Series(port_ret, index=returns.index)
roll_vol = port_series.rolling(30).std() * np.sqrt(252) * 100   # annualised %

# Individual asset rolling vol for context
asset_roll = returns.rolling(30).std() * np.sqrt(252) * 100

# Target: reduce baseline vol by 15%
baseline_vol = port_series.std() * np.sqrt(252) * 100
target_vol   = baseline_vol * 0.85

fig, ax = plt.subplots(figsize=(12, 5))

# Faint individual asset lines
for col in returns.columns:
    ax.plot(asset_roll.index, asset_roll[col],
            color=SECTOR_COLORS[SECTORS[col]], alpha=0.12, linewidth=0.8)

# Portfolio rolling vol
ax.plot(roll_vol.index, roll_vol,
        color=ACCENT, linewidth=2.2, label="Portfolio (equal weight)", zorder=5)

# Baseline + target lines
ax.axhline(baseline_vol, color=WARN,   linewidth=1.4, linestyle="--",
           label=f"Baseline avg: {baseline_vol:.1f}%")
ax.axhline(target_vol,   color="green", linewidth=1.4, linestyle="--",
           label=f"Target (−15%): {target_vol:.1f}%")

# Shade COVID spike
ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-01"),
           alpha=0.08, color=DANGER, label="COVID-19 shock")
ax.text(pd.Timestamp("2020-03-15"), roll_vol.max() * 0.92,
        "COVID-19\nshock", fontsize=8, color=DANGER, ha="center")

# 2022 rate hike annotation
ax.annotate("Fed rate hikes\nbegin", xy=(pd.Timestamp("2022-03-01"), 22),
            xytext=(pd.Timestamp("2021-06-01"), 33),
            arrowprops=dict(arrowstyle="->", color=WARN, lw=1.2),
            fontsize=8, color=WARN)

ax.set_title("30-Day Rolling Annualised Volatility - Portfolio vs Individual Assets",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Annualised Volatility (%)", fontsize=10)
ax.set_xlabel("")
ax.legend(fontsize=9, framealpha=0.9)
ax.set_ylim(0, roll_vol.max() * 1.15)

plt.tight_layout()
plt.savefig(r"C:\Users\Mateusz\Desktop\portfolio-risk-analysis\charts\chart2_rolling_volatility.png", dpi=150, bbox_inches="tight")
plt.close()

#CHART 3: Ishikawa (fishbone) diagram

causes = {
    "Asset\nConcentration": [
        "3 tech stocks (30%)",
        "High NVDA weight",
        "Low bond allocation",
    ],
    "Macro\nExposure": [
        "Fed rate sensitivity",
        "USD-correlated assets",
        "No inflation hedge",
    ],
    "Sector\nCorrelation": [
        "Tech–Finance r > 0.75",
        "SPY amplifies swings",
        "No negative corr. assets",
    ],
    "Rebalancing\nPolicy": [
        "Static equal weights",
        "No vol-targeting rule",
        "No drawdown circuit",
    ],
}

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 14); ax.set_ylim(0, 7)
ax.axis("off")

# Spine (horizontal backbone)
ax.annotate("", xy=(13.2, 3.5), xytext=(0.5, 3.5),
            arrowprops=dict(arrowstyle="-|>", color="#333333",
                            lw=2.5, mutation_scale=20))

# Effect box (right)
effect_box = dict(boxstyle="round,pad=0.6", facecolor=DANGER,
                  edgecolor="#8B0000", linewidth=1.5)
ax.text(13.5, 3.5, "High\nPortfolio\nVolatility",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="white", bbox=effect_box)

# Title
ax.text(7, 6.6, "Ishikawa Diagram - Root Cause Analysis: Portfolio Volatility",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#222222")

# Four branches (top-left, top-right, bottom-left, bottom-right)
branch_configs = [
    ("Asset\nConcentration", causes["Asset\nConcentration"],  2.5, "top",    ACCENT),
    ("Macro\nExposure",      causes["Macro\nExposure"],        5.5, "top",    WARN),
    ("Sector\nCorrelation",  causes["Sector\nCorrelation"],    2.5, "bottom", "#27AE60"),
    ("Rebalancing\nPolicy",  causes["Rebalancing\nPolicy"],    5.5, "bottom", "#8E44AD"),
]

for label, items, spine_x, position, color in branch_configs:
    if position == "top":
        branch_y_end = 3.5
        branch_y_start = 5.8
        sub_dir = 1          # ribs point downward from branch
        label_y = 6.1
    else:
        branch_y_end = 3.5
        branch_y_start = 1.2
        sub_dir = -1         # ribs point upward from branch
        label_y = 0.85

    # Main branch line
    ax.annotate("", xy=(spine_x, branch_y_end),
                xytext=(spine_x - 0.3, branch_y_start),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2, mutation_scale=14))

    # Branch label
    box_style = dict(boxstyle="round,pad=0.4", facecolor=color,
                     edgecolor=color, alpha=0.9)
    ax.text(spine_x - 0.3, label_y, label,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white", bbox=box_style)

    # Sub-cause ribs
    for i, item in enumerate(items):
        rib_x = spine_x - 0.6 - i * 0.55
        rib_y_start = branch_y_start + sub_dir * (-0.6 - i * 0.3)
        # interpolate point on the branch line
        t = (rib_x - (spine_x - 0.3)) / (spine_x - (spine_x - 0.3) + 0.01)
        t = max(0, min(1, abs(t)))
        rib_y_end = branch_y_start + (branch_y_end - branch_y_start) * t

        ax.plot([rib_x, rib_x + 0.55], [rib_y_start, rib_y_end],
                color=color, linewidth=1.2, alpha=0.7)

        text_y = rib_y_start + sub_dir * (-0.25)
        ax.text(rib_x, text_y, item,
                ha="center", va="center", fontsize=7.5,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, linewidth=0.8, alpha=0.9))

plt.tight_layout()
plt.savefig(r"C:\Users\Mateusz\Desktop\portfolio-risk-analysis\charts\chart3_ishikawa.png", dpi=150, bbox_inches="tight")
plt.close()

