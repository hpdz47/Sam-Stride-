import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import os

def Violin_Plots(root_dir, model_name, plot_ablations=None):

    if not root_dir.exists():
        raise FileNotFoundError(f"The folder {root_dir} does not exist.")

    allowed_ablations = ["NN", "MN", "NM", "MM"]
    if plot_ablations is not None:
        allowed_ablations = [a for a in allowed_ablations if a in plot_ablations]

    allowed_iters = ["5", "10"]
    results = {}

    for model_dir in sorted(root_dir.iterdir()):
        if not model_dir.is_dir() or model_name not in model_dir.name.lower():
            continue

        for iter_dir in sorted(model_dir.iterdir()):
            if not iter_dir.is_dir():
                continue

            iter_match = None
            for allowed in allowed_iters:
                if allowed in iter_dir.name:
                    iter_match = allowed
                    break
            if iter_match is None:
                continue

            ablation_data = {}
            for ablation in allowed_ablations:
                abl_path = iter_dir / ablation
                if abl_path.exists():
                    for file in abl_path.iterdir():
                        if file.suffix == '.csv' and file.name != "LLM_Judge_Scoring.csv":
                            data = pd.read_csv(file)
                            data = data[data["Success"] == 1]

                            debug = data["Debug_Attempts"].to_numpy()
                            complexity = (data["Cyclomatic_Complexity"]).to_numpy()
                            ablation_data[ablation] = np.column_stack([debug, complexity])

            if ablation_data:
                results[(model_dir.name, iter_dir.name)] = ablation_data

    # ===================== Violin Plot =====================
    # Scale factor: bar charts are 2.5 in → 3.5 in column = 1.4×
    # Violin is 7.0 in → 7.16 in page = ~1.0×
    # So violin text must be 1.4× larger to match on the printed page.
    SF = 1.4  # scale factor

    plt.rcParams.update({
        "text.usetex":        False,
        "font.family":        "serif",
        "font.serif":         ["Times", "Times New Roman", "DejaVu Serif"],
        "font.size":          8 * SF,
        "axes.labelsize":     8 * SF,
        "axes.titlesize":     9 * SF,
        "xtick.labelsize":    7 * SF,
        "ytick.labelsize":    7 * SF,
        "legend.fontsize":    6.5 * SF,
        "axes.linewidth":     0.6 * SF,
        "xtick.major.width":  0.6 * SF,
        "ytick.major.width":  0.6 * SF,
        "lines.linewidth":    0.8 * SF,
        "patch.linewidth":    0.6 * SF,
        "figure.dpi":         300,
        "savefig.dpi":        600,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    })

    ablations = ["NN", "MN", "NM", "MM"]
    c_55, c_1010 = "#4C72B0", "#DD8452"

    # ── Gather per-ablation debug-attempt arrays for 4B only ──
    data_55   = {}
    data_1010 = {}

    for (model_dir_name, iter_name), ablation_data in results.items():
        if "4b" not in model_dir_name.lower():
            continue
        for abl in ablations:
            if abl not in ablation_data:
                continue
            debug_vals = ablation_data[abl][:, 1]
            if "10" in iter_name:
                data_1010[abl] = debug_vals
            elif "5" in iter_name:
                data_55[abl] = debug_vals

    # ── Build figure: full page width, compact height ──
    fig, ax = plt.subplots(figsize=(7.0, 2.8))

    x = np.arange(len(ablations))
    vwidth = 0.35
    offset = 0.22

    for i, abl in enumerate(ablations):
        for side, (store, colour, lbl) in enumerate([
            (data_55,   c_55,   "Iterations\n(5-5)"),
            (data_1010, c_1010, "Iterations\n(10-10)"),
        ]):
            if abl not in store or len(store[abl]) < 2:
                continue

            vals = store[abl]
            pos = x[i] + (-offset if side == 0 else offset)

            parts = ax.violinplot(
                vals,
                positions=[pos],
                widths=vwidth,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            # Style the violin body
            for pc in parts["bodies"]:
                pc.set_facecolor(colour)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.6 * SF)
                pc.set_alpha(0.85)

            # ── Mean bar (thick) ──
            mean_val = np.mean(vals)
            ax.plot(
                [pos - vwidth / 3, pos + vwidth / 3], [mean_val, mean_val],
                color="black", linewidth=2.0 * SF, zorder=6,
            )

            # ── Min / Max bars (thinner) ──
            v_min, v_max = np.min(vals), np.max(vals)
            for edge_val in [v_min, v_max]:
                ax.plot(
                    [pos - vwidth / 5, pos + vwidth / 5], [edge_val, edge_val],
                    color="black", linewidth=1.0 * SF, zorder=6,
                )

            # ── Central vertical line connecting min to max ──
            ax.plot(
                [pos, pos], [v_min, v_max],
                color="black", linewidth=0.7 * SF, zorder=5,
            )

    # ── Axes ──
    ax.set_xticks(x)
    ax.set_xticklabels(ablations)
    ax.set_xlabel("Episodic Memory Placement")
    ax.set_ylabel("Cyclomatic Complexity")
    
    #ax.set_title("Qwen3.5-4B Cyclomatic Complexity Distribution", pad=6)

    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    ax.grid(axis="y", linewidth=0.3 * SF, color="#D0D0D0", alpha=1.0, zorder=0)

    # ── Legend (manual patches so labels match bar chart style) ──
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c_55,   edgecolor="black", label="4B \n5 Iterations"),
        Patch(facecolor=c_1010, edgecolor="black", label="4B \n10 Iterations"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        frameon=True,
        edgecolor="black",
        fancybox=False,
        handletextpad=0.3,
        handleheight=1.5,
        handlelength=1.5,
        borderpad=0.25,
        columnspacing=0.4,
        labelspacing=0.3,
        ncol=1,
    )

    fig.tight_layout(pad=0.3)

    output_dir = root_dir / "Report_Results"
    #os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_dir / "4B_debug_violin.png", format="png")
    plt.close(fig)
    print("Saved  4B_debug_violin.png")

     # ── 27B Cyclomatic Complexity (5-5 only, single colour) ──
    data_27B_55 = {}
    for (model_dir_name, iter_name), ablation_data in results.items():
        if "27b" not in model_dir_name.lower():
            continue
        if "5" not in iter_name or "10" in iter_name:
            continue
        for abl in ablations:
            if abl in ablation_data:
                data_27B_55[abl] = ablation_data[abl][:, 1]  # column 1 = Cyclomatic Complexity

    if data_27B_55:
        fig27, ax27 = plt.subplots(figsize=(7.0, 2.8))
        x = np.arange(len(ablations))

        for i, abl in enumerate(ablations):
            if abl not in data_27B_55 or len(data_27B_55[abl]) < 2:
                continue
            vals = data_27B_55[abl]
            pos = x[i]

            parts = ax27.violinplot(
                vals, positions=[pos], widths=vwidth,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(c_55)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.6 * SF)
                pc.set_alpha(0.85)

            mean_val = np.mean(vals)
            ax27.plot([pos - vwidth/3, pos + vwidth/3], [mean_val, mean_val],
                      color="black", linewidth=2.0 * SF, zorder=6)
            v_min, v_max = np.min(vals), np.max(vals)
            for edge_val in [v_min, v_max]:
                ax27.plot([pos - vwidth/5, pos + vwidth/5], [edge_val, edge_val],
                          color="black", linewidth=1.0 * SF, zorder=6)
            ax27.plot([pos, pos], [v_min, v_max],
                      color="black", linewidth=0.7 * SF, zorder=5)

        ax27.set_xticks(x)
        ax27.set_xticklabels(ablations)
        ax27.set_xlabel("Episodic Memory Placement")
        ax27.set_ylabel("Cyclomatic Complexity")
        #ax27.set_title("Qwen3.5-27B Cyclomatic Complexity Distribution", pad=6)
        ax27.set_xlim(x[0] - 0.6, x[-1] + 0.6)
        ax27.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
        ax27.grid(axis="y", linewidth=0.3 * SF, color="#D0D0D0", alpha=1.0, zorder=0)

        legend_elements_27 = [
            Patch(facecolor=c_55, edgecolor="black", label="27B \n5 Iterations"),
        ]
        ax27.legend(
            handles=legend_elements_27, loc="upper left", frameon=True,
            edgecolor="black", fancybox=False, handletextpad=0.3,
            handleheight=1.5, handlelength=1.5, borderpad=0.25,
            columnspacing=0.4, labelspacing=0.3, ncol=1,
        )

        fig27.tight_layout(pad=0.3)
        fig27.savefig(output_dir / "27B_complexity_violin.png", format="png")
        plt.close(fig27)
        print("Saved  27B_complexity_violin.png")


def main():
    if len(sys.argv) < 2:
        print("Usage: python Scatter_Plot_Analysis.py <model_name>")
        sys.exit(1)

    CURRENT_DIR = Path(__file__).parent
    model_name = sys.argv[1].lower()

    results = Violin_Plots(CURRENT_DIR, model_name)

    if not results:
        raise ValueError(f"No folders found for model name: {model_name}")

    for (model_dir_name, iter_name), ablation_data in sorted(results.items()):
        print(f"\n===== {model_dir_name} | Iteration: {iter_name} =====")
        for ablation, data in ablation_data.items():
            print(f"  {ablation} | Rows: {len(data)}")
            print(f"    Debug_Attempts:     {data[:, 0]}")
            print(f"    CC + Nesting_Depth: {data[:, 1]}")


if __name__ == "__main__":
    main()