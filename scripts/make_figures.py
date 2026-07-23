"""Generate the paper's figures from the campaign results (paper/figures/*.png).

Design: Okabe-Ito colorblind-safe qualitative palette, identity in fixed order, with
distinct markers + line styles as secondary encoding (print/CVD robustness); one axis
per panel; recessive grid; legend present. Numbers are read from result CSVs.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Okabe-Ito (2008) colorblind-safe qualitative palette
OI = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#7f7f7f",
}  # fmt: skip

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 300,
})  # fmt: skip


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(REPO / "results" / "raw" / f"{name}_full" / "summaries.csv").set_index(
        "task_name"
    )


FP32 = load("chronos2_fp32")


def retention(name: str, col: str = "WQL") -> np.ndarray:
    df = load(name)
    c = FP32.index.intersection(df.index)
    return (df.loc[c, col] / FP32.loc[c, col]).clip(1e-2, 1e2).to_numpy()


def gm(name: str, col: str = "WQL") -> float:
    return float(scipy.stats.gmean(retention(name, col)))


def coverage_curve(name: str) -> tuple[list[float], list[float]]:
    df = load(name)
    nominal = [0.2, 0.4, 0.6, 0.8]
    emp = [float(df[f"coverage[{q}]"].mean()) for q in nominal]
    return nominal, emp


def make_composite() -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # --- Panel A: WQL retention vs weight bits, by method ---
    methods = {
        "GPTQ": (OI["blue"], "o", "-", {8: "chronos2_w8_gptq", 4: "chronos2_w4_gptq",
                                        3: "chronos2_w3_gptq"}),
        "HQQ": (OI["green"], "s", "--", {8: "chronos2_w8_hqq", 4: "chronos2_w4_hqq",
                                         3: "chronos2_w3_hqq"}),
        "RTN": (OI["vermillion"], "^", ":", {8: "chronos2_w8a16_rtn", 6: "chronos2_w6_rtn_sim",
                                             4: "chronos2_w4_rtn_sim", 3: "chronos2_w3_rtn_sim",
                                             2: "chronos2_w2_rtn_sim"}),
    }  # fmt: skip
    for label, (color, marker, ls, runs) in methods.items():
        bits = sorted(runs)
        vals = [gm(runs[b]) for b in bits]
        ax1.plot(bits, vals, ls, color=color, marker=marker, markersize=5,
                 linewidth=1.6, label=label, clip_on=False)
    # standalone W4 points: NF4 and AWQ-fold
    ax1.scatter([4], [gm("chronos2_w4_bnb_nf4")], color=OI["purple"], marker="D",
                s=34, label="NF4 (W4)", zorder=5)
    ax1.scatter([4], [gm("chronos2_w4_awq_rtn")], color=OI["orange"], marker="v",
                s=40, label="AWQ-fold (W4)", zorder=5)
    ax1.axhline(1.0, color=OI["gray"], linewidth=0.8, linestyle="-", alpha=0.7)
    ax1.text(7.6, 1.005, "fp32", color=OI["gray"], fontsize=7, va="bottom", ha="right")
    ax1.set_yscale("log")
    ax1.set_yticks([1.0, 1.2, 1.5, 2.0, 3.0, 4.0])
    ax1.set_yticklabels(["1.0", "1.2", "1.5", "2.0", "3.0", "4.0"])
    ax1.set_xticks([2, 3, 4, 6, 8])
    ax1.set_xlabel("weight bits")
    ax1.set_ylabel("WQL retention vs fp32  (lower better)")
    ax1.set_title("(a) accuracy vs bit-width", fontsize=9)
    ax1.legend(frameon=False, fontsize=6.8, loc="upper right", ncol=1,
               handlelength=1.8, labelspacing=0.3)
    ax1.set_xlim(1.7, 8.3)

    # --- Panel B: calibration curves ---
    cal = [
        ("fp32", "chronos2_fp32", OI["gray"], "o", "-"),
        ("GPTQ W4", "chronos2_w4_gptq", OI["blue"], "o", "-"),
        ("NF4 W4", "chronos2_w4_bnb_nf4", OI["purple"], "D", "--"),
        ("HQQ W4", "chronos2_w4_hqq", OI["green"], "s", "--"),
        ("HQQ W3", "chronos2_w3_hqq", OI["vermillion"], "^", ":"),
    ]
    ax2.plot([0.15, 0.85], [0.15, 0.85], color="#bbbbbb", linewidth=0.9, zorder=0)
    ax2.text(0.8, 0.83, "ideal", fontsize=6.5, color="#999999", rotation=38,
             va="top", ha="right")
    for label, name, color, marker, ls in cal:
        nominal, emp = coverage_curve(name)
        ax2.plot(nominal, emp, ls, color=color, marker=marker, markersize=5,
                 linewidth=1.5, label=label, clip_on=False)
    ax2.set_xlabel("nominal interval coverage")
    ax2.set_ylabel("empirical coverage")
    ax2.set_title("(b) interval calibration", fontsize=9)
    ax2.set_xlim(0.12, 0.88)
    ax2.set_ylim(0.12, 0.88)
    ax2.legend(frameon=False, fontsize=6.8, loc="upper left", labelspacing=0.3,
               handlelength=1.8)

    fig.tight_layout(pad=0.6)
    out = FIG / "fig1_accuracy_calibration.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = make_composite()
    print(f"wrote {p}")
