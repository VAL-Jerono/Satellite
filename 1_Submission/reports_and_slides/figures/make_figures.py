"""Generate report figures from values reported in CAPSTONE_NOTEBOOK.ipynb.

All values are transcribed from notebook outputs; nothing is re-estimated.
Palette matches the submission's document and slide design system.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
INK = "#1B262C"
MUTED = "#5A6966"
SURFACE = "#E9EDEC"
ACCENT = "#B85B3E"
SAGE = "#5C746A"
SAND = "#D4C5B0"
PAPER = "#F5F7F6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- LOZO by zone
zones = ["Rift Valley", "Highland", "Lake Basin", "Coastal", "ASAL"]
lozo = [0.5443, 0.5050, 0.4728, 0.3665, 0.2732]
mean_lozo = 0.4324

fig, ax = plt.subplots(figsize=(8.2, 4.3))
colors = [INK, INK, INK, SAGE, ACCENT]
bars = ax.bar(zones, lozo, color=colors, width=0.62)
ax.axhline(mean_lozo, color=MUTED, lw=1.1, ls=(0, (4, 3)))
ax.text(4.42, mean_lozo + 0.012, f"Mean {mean_lozo:.4f}", ha="right", va="bottom",
        fontsize=9, color=MUTED)
for bar, value in zip(bars, lozo):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.4f}",
            ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
ax.set_ylim(0, 0.62)
ax.set_ylabel("Macro-F1 (held-out zone)")
ax.set_title("Leave-one-zone-out performance falls sharply in the ASAL holdout",
             loc="left", fontsize=12, fontweight="bold", pad=12)
ax.grid(axis="y", color="#D8DEDC", lw=0.7)
ax.set_axisbelow(True)
fig.text(0.01, -0.06,
         "Values as reported in CAPSTONE_NOTEBOOK.ipynb §8.1. Each bar is a separate out-of-domain fold; "
         "the mean is not a national accuracy estimate.",
         fontsize=8, color=MUTED, ha="left")
save(fig, "fig_lozo.png")

# ------------------------------------------------------------- model benchmark
models = ["Decision tree\n(depth 4)", "Monolithic\nLightGBM", "Zone LightGBM\n(5 specialists)",
          "Spectral CNN", "Stacking\nensemble"]
scores = [0.4309, 0.4200, 0.6271, 0.6184, 0.6504]
colors = [SAND, SAGE, INK, INK, ACCENT]

fig, ax = plt.subplots(figsize=(8.4, 4.2))
y = np.arange(len(models))[::-1]
ax.hlines(y, 0, scores, color="#C9D2CF", lw=2)
ax.scatter(scores, y, s=110, color=colors, zorder=3, edgecolor=PAPER, linewidth=1.5)
for yi, s in zip(y, scores):
    ax.text(s + 0.008, yi, f"{s:.4f}", va="center", fontsize=10, fontweight="bold", color=INK)
ax.set_yticks(y, models, fontsize=9)
ax.set_xlim(0, 0.72)
ax.set_xlabel("Macro-F1 as reported in the notebook")
ax.set_title("Reported benchmark table mixes evaluation protocols",
             loc="left", fontsize=12, fontweight="bold", pad=12)
ax.grid(axis="x", color="#D8DEDC", lw=0.7)
ax.set_axisbelow(True)
fig.text(0.01, -0.08,
         "Notebook §6.2. The 0.6271 zone-LightGBM figure and the 0.4324 LOZO mean come from different "
         "protocols and are not directly comparable; treat this chart as a reporting inventory, not a leaderboard.",
         fontsize=8, color=MUTED, ha="left")
save(fig, "fig_benchmark.png")

# ---------------------------------------------------------------- calibration
fig, ax = plt.subplots(figsize=(6.4, 3.8))
labels = ["Before\ncalibration", "After\ncalibration"]
vals = [0.2525, 0.0921]
bars = ax.bar(labels, vals, color=[ACCENT, SAGE], width=0.46)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.annotate("", xy=(1, 0.105), xytext=(0, 0.245),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.2,
                            connectionstyle="arc3,rad=-0.25"))
ax.text(0.5, 0.205, "63.5% reduction", ha="center", fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec="none"))
ax.set_ylim(0, 0.30)
ax.set_ylabel("Expected calibration error (ECE)")
ax.set_title("Reported calibration improvement (T = 1.42)", loc="left",
             fontsize=12, fontweight="bold", pad=10)
ax.grid(axis="y", color="#D8DEDC", lw=0.7)
ax.set_axisbelow(True)
fig.text(0.01, -0.09,
         "Notebook §8.3. ECE is reported, not recomputed from a saved calibration set; "
         "the reliability bins are not included in the submission.",
         fontsize=8, color=MUTED, ha="left")
save(fig, "fig_calibration.png")

# --------------------------------------------------------------- class balance
classes = ["Grass / shrub", "Bare", "Cropland", "Trees", "Built-up", "Water"]
counts = [5125, 4845, 4786, 4689, 4623, 4053]
colors = [SAGE, SAND, "#8C7A5B", "#3F5F4C", ACCENT, "#4C6E86"]

fig, ax = plt.subplots(figsize=(7.4, 3.6))
bars = ax.barh(classes[::-1], counts[::-1], color=colors[::-1], height=0.62)
for bar, v in zip(bars, counts[::-1]):
    ax.text(v + 45, bar.get_y() + bar.get_height() / 2, f"{v:,}", va="center",
            fontsize=9.5, color=INK)
ax.set_xlim(0, 5800)
ax.set_xlabel("Observations")
ax.set_title("Reference-label classes are close to balanced", loc="left",
             fontsize=12, fontweight="bold", pad=10)
ax.grid(axis="x", color="#D8DEDC", lw=0.7)
ax.set_axisbelow(True)
fig.text(0.01, -0.10,
         "Notebook §3.1, n = 28,121. Labels derive from ESA WorldCover 2021, not from independent field survey.",
         fontsize=8, color=MUTED, ha="left")
save(fig, "fig_class_balance.png")

# ------------------------------------------------------------- county allocations
counties = ["Nakuru", "Kajiado", "Nyeri", "Nyandarua", "Kisumu", "Siaya",
            "Kilifi", "Lamu", "Turkana", "Marsabit"]
settle = np.array([31.2, 18.5, 24.8, 16.9, 29.4, 21.0, 26.3, 14.2, 12.1, 11.8])
farm = np.array([44.1, 51.3, 38.9, 55.3, 36.2, 48.5, 32.1, 28.6, 22.3, 21.4])
protect = np.array([24.7, 30.2, 36.3, 27.8, 34.4, 30.5, 41.6, 57.2, 65.6, 66.8])

fig, ax = plt.subplots(figsize=(9.4, 4.4))
x = np.arange(len(counties))
ax.bar(x, settle, color=ACCENT, label="SETTLE", width=0.68)
ax.bar(x, farm, bottom=settle, color=SAND, label="FARM", width=0.68)
ax.bar(x, protect, bottom=settle + farm, color=SAGE, label="PROTECT", width=0.68)
ax.set_xticks(x, counties, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, 100)
ax.set_ylabel("Share of sampled observations (%)")
ax.set_title("Prototype rule outputs by county, as tabulated in the notebook",
             loc="left", fontsize=12, fontweight="bold", pad=10)
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=9)
ax.grid(axis="y", color="#D8DEDC", lw=0.7)
ax.set_axisbelow(True)
fig.text(0.01, -0.28,
         "Notebook §10.2. Shares are descriptive outputs of a rule applied to sampled points; they are not "
         "county land-area statistics and were not produced by an audited cadastral overlay.",
         fontsize=8, color=MUTED, ha="left")
save(fig, "fig_allocations.png")

# ------------------------------------------------------- evidence provenance map
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.1)
ax.axis("off")
levels = [
    ("A. Reproducible from this package", "Code, configs and model artifacts are present;\nlimited by missing raw data and outputs.", SAGE),
    ("B. Reported in the notebook", "Metric tables, figures and narrative outputs;\nnot independently rerun for this submission.", INK),
    ("C. Simulated or illustrative", "Drift audit, load tests and smoke-test tables\nbuilt from synthetic or hard-coded values.", SAND),
    ("D. Unverified external claim", "Live deployment, uptime, CI status and field\naccuracy; no primary evidence in the package.", ACCENT),
]
y0 = 4.55
for i, (title, body, color) in enumerate(levels):
    y = y0 - i * 1.32
    ax.add_patch(Rectangle((0.1, y - 0.52), 9.8, 1.12, facecolor=SURFACE, edgecolor="none"))
    ax.add_patch(Rectangle((0.1, y - 0.52), 0.12, 1.12, facecolor=color, edgecolor="none"))
    ax.text(0.42, y + 0.34, title, fontsize=10.5, fontweight="bold", color=INK, va="center")
    ax.text(0.42, y - 0.16, body, fontsize=8.8, color=MUTED, va="center", linespacing=1.4)
ax.text(0.1, 5.72, "Evidence register used throughout this submission", fontsize=13,
        fontweight="bold", color=INK)
save(fig, "fig_evidence.png")
print("figures written")
