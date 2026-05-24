import scipy as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from pathlib import Path
import csv
import os

# ──────────────────────────────────────────────────────────────
# IEEE-quality global rcParams
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":        False,            # set True if you have a LaTeX install
    "font.family":        "serif",
    "font.serif":         ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size":          8,
    "axes.labelsize":     8,
    "axes.titlesize":     9,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    6.5,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "lines.linewidth":    0.8,
    "patch.linewidth":    0.6,
    "figure.dpi":         300,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
})

# ──────────────────────────────────────────────────────────────
# Figure width for 3 side-by-side subfigures on A4 (textwidth
# ≈ 7.16 in for IEEE two-column, ≈ 6.9 in single-column).
# Each panel ≈ one-third of textwidth with a small gap.
# ──────────────────────────────────────────────────────────────
SINGLE_FIG_W = 2.5          # inches  (≈ 7.0 / 3)
SINGLE_FIG_H = 2.2          # inches  – compact but legible


def Bar_Chart_Plotting(folder_path):
    """
    Plots the results for 4B (5-5 & 10-10) and 27B (5-5).
    The 4B model produces THREE separate per-metric PDFs suitable
    for IEEE \subfigure placement (a, b, c).
    """
    allowed_ablations = ["NN", "MN", "NM", "MM"]
    Data_4B_5_5   = np.zeros((4, 3))
    Data_4B_10_10 = np.zeros((4, 3))
    Data_27B_5_5  = np.zeros((4, 3))

    Lower_CI_4B_5_5 = np.zeros((4, 3))
    Upper_CI_4B_5_5 = np.zeros((4, 3))

    allowed_sizes = ["4B", "27B"]

    for i, item in enumerate(allowed_ablations):
        Stats_Path = folder_path / item
        if Stats_Path.exists():
            for file in Stats_Path.iterdir():
                for size in allowed_sizes:
                    if size == "4B":
                        if file.suffix == ".csv" and "5" in file.name and "4B" in file.name and "10" not in file.name and "_CI" not in file.name:
                            df = pd.read_csv(file)
                            Data_4B_5_5[i] = df[["Success_Rate", "Debug_Attempts", "LLM-as-a-Judge_Score"]].to_numpy()

                        elif file.suffix == ".csv" and "5" in file.name and "4B" in file.name and "10" not in file.name and "_CI" in file.name:
                            df = pd.read_csv(file)
                            Lower_CI_4B_5_5[i] = df[["Success_Rate", "Debug_Attempts", "LLM-as-a-Judge_Score"]].iloc[0].to_numpy()
                            Upper_CI_4B_5_5[i] = df[["Success_Rate", "Debug_Attempts", "LLM-as-a-Judge_Score"]].iloc[1].to_numpy()

                        elif file.suffix == ".csv" and "10" in file.name and "4B" in file.name and "_CI" not in file.name:
                            df = pd.read_csv(file)
                            Data_4B_10_10[i] = df[["Success_Rate", "Debug_Attempts", "LLM-as-a-Judge_Score"]].to_numpy()
                    elif size == "27B":
                        if file.suffix == ".csv" and "27B" in file.name and "_CI" not in file.name:
                            df = pd.read_csv(file)
                            Data_27B_5_5[i] = df[["Success_Rate", "Debug_Attempts", "LLM-as-a-Judge_Score"]].to_numpy()

        # ──────────────────────────────────────────────────────────
    # Helper: plot ONE metric for the 4B model
    # ──────────────────────────────────────────────────────────
    def _plot_single_metric_4B(
        metric_idx,          # 0=Success Rate, 1=Debug Attempts, 2=LLM Judge
        ylabel,
        colour,
        filename,
        title="",
        scale=1.0,           # multiplier applied to raw values
        is_zero_one=False,   # True → fixed 0-1 ticks (Success Rate)
        is_data_driven=False,# True → tight y-axis around data (LLM Judge)
        baselines_27B=None,  # array of 4 values: one per ablation (already scaled)
        lower_ci_55=None,
        upper_ci_55=None,
    ):
        categories = ["NN", "MN", "NM", "MM"]
        x = np.arange(len(categories))
        width = 0.30

        vals_55   = Data_4B_5_5[:, metric_idx]   * scale
        vals_1010 = Data_4B_10_10[:, metric_idx]  * scale

        yerr_55 = None
        if lower_ci_55 is not None and upper_ci_55 is not None:
            lower_err = vals_55 - (lower_ci_55[:, metric_idx] * scale)
            upper_err = (upper_ci_55[:, metric_idx] * scale) - vals_55
            yerr_55 = np.vstack([lower_err, upper_err])

        fig, ax = plt.subplots(figsize=(SINGLE_FIG_W, SINGLE_FIG_H))

        ax.bar(
            x - width / 2, vals_55, width,
            color=colour, edgecolor="black", linewidth=0.5,
            label="4B\n5 Iters", zorder=3,
            yerr=yerr_55,
            capsize=2,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "black"},
        )
        ax.bar(
            x + width / 2, vals_1010, width,
            color=colour, edgecolor="black", linewidth=0.5,
            hatch="////", label="4B\n10 Iters", zorder=3,
        )

        # ── Per-ablation 27B baseline segments (solid red) ──
        if baselines_27B is not None:
            margin = 0.12
            for j, val in enumerate(baselines_27B):
                seg_left  = x[j] - width / 2 - margin
                seg_right = x[j] + width / 2 + margin
                label = "27B\n5 Iters" if j == 0 else None
                ax.plot(
                    [seg_left, seg_right], [val, val],
                    color="red", linestyle="-", linewidth=1.2,
                    zorder=5, label=label,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episodic Memory Placement")
        if title:
            ax.set_title(title, pad=4)

        # ── Collect all plotted values for limit calculations ──
        all_vals = np.concatenate([vals_55, vals_1010])
        if baselines_27B is not None:
            all_vals = np.concatenate([all_vals, baselines_27B])
        v_min = np.min(all_vals)
        v_max = np.max(all_vals)

        # ── Y-axis scaling ──
        if is_zero_one:
            # Success Rate: fixed 0–1, uniform 0.2 steps
            data_top = 1.0
            ticks = np.arange(0, 1.1, 0.2)
            ax.yaxis.set_major_locator(plt.FixedLocator(ticks))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
            display_top = 1.35
            ax.set_ylim(0, display_top)

        elif is_data_driven:
            # LLM Judge: tight axis so differences are visible
            floor  = 0
            ceil   = 0.7
            data_top = ceil
            ticks = np.arange(floor, ceil + 0.05, 0.1)
            ax.yaxis.set_major_locator(plt.FixedLocator(ticks))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
            display_top = ceil + (ceil - floor) * 0.35
            ax.set_ylim(floor, display_top)

        else:
            # Debug Attempts: auto range from 0
            data_top = np.ceil(v_max * 10) / 10
            ticks = np.linspace(0, data_top, 6)
            ax.yaxis.set_major_locator(plt.FixedLocator(ticks))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2g}"))
            display_top = data_top * 1.35
            ax.set_ylim(0, display_top)

        ax.grid(axis="y", linewidth=0.3, color="#D0D0D0", alpha=1.0, zorder=0)

        # ── Legend: single row across the top, INSIDE the headroom ──
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            frameon=True,
            edgecolor="black",
            fancybox=False,
            handletextpad=0.3,
            borderpad=0.25,
            columnspacing=0.4,
            labelspacing=0.3,
            ncol=3,
            fontsize=6.5,
        )

        fig.tight_layout(pad=0.3)

        #os.makedirs(folder_path, exist_ok=True)
        fig.savefig(os.path.join(folder_path, filename), format="png")
        plt.close(fig)
        print(f"Saved  {filename}")

    # ──────────────────────────────────────────────────────────
    # 27B per-ablation baselines (one value per config)
    # ──────────────────────────────────────────────────────────
    baselines_27B_sr  = Data_27B_5_5[:, 0]
    baselines_27B_dbg = Data_27B_5_5[:, 1]
    baselines_27B_llm = Data_27B_5_5[:, 2] / 10.0

    # ──────────────────────────────────────────────────────────
    # Generate the three 4B per-metric figures
    # ──────────────────────────────────────────────────────────
    if not (np.all(Data_4B_5_5 == 0) and np.all(Data_4B_10_10 == 0)):

        # (a) LLM-as-a-Judge Score — tight scale to see differences
        _plot_single_metric_4B(
            metric_idx=2,
            ylabel="LLM-as-a-Judge Score",
            colour="#4C72B0",
            filename="4B_llm_judge.png",
            title=None,
            scale=1.0 / 10.0,
            is_data_driven=True,
            baselines_27B=baselines_27B_llm,
            #lower_ci_55=Lower_CI_4B_5_5,
            #upper_ci_55=Upper_CI_4B_5_5,
        )

        # (b) Success Rate — fixed 0-1 scale
        _plot_single_metric_4B(
            metric_idx=0,
            ylabel="Success Rate",
            colour="#55A868",
            filename="4B_success_rate.png",
            title=None,
            scale=1.0,
            is_zero_one=True,
            baselines_27B=baselines_27B_sr,
            lower_ci_55=Lower_CI_4B_5_5,
            upper_ci_55=Upper_CI_4B_5_5,
        )

        # (c) Debug Attempts — auto range from 0
        _plot_single_metric_4B(
            metric_idx=1,
            ylabel="Avg. Debug Attempts",
            colour="#DD8452",
            filename="4B_debug_attempts.png",
            title=None,
            scale=1.0,
            baselines_27B=baselines_27B_dbg,
            lower_ci_55=Lower_CI_4B_5_5,
            upper_ci_55=Upper_CI_4B_5_5,
        )


if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent
    RESULT_DIR  = CURRENT_DIR / "Report_Results"
    Bar_Chart_Plotting(RESULT_DIR)
