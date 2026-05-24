import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import os

def Scatter_Plot_Analysis(root_dir, model_name, plot_ablations=None):

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
                            complexity = (data["Cyclomatic_Complexity"] + data["Nesting_Depth"]).to_numpy()
                            ablation_data[ablation] = np.column_stack([debug, complexity])

            if ablation_data:
                results[(model_dir.name, iter_dir.name)] = ablation_data

    # ===================== Plotting =====================
    model_groups = {}
    for (model_dir_name, iter_name), ablation_data in results.items():
        if model_dir_name not in model_groups:
            model_groups[model_dir_name] = []
        model_groups[model_dir_name].append((iter_name, ablation_data))

    # Unique marker per ablation
    ablation_markers = {"NN": "o", "MN": "s", "NM": "^", "MM": "D"}

    # Unique colour per (ablation, iteration) combination
    combo_colours = {
        ("NN", "5"): "tab:blue",    ("NN", "10"): "lightskyblue",
        ("MN", "5"): "tab:orange",  ("MN", "10"): "moccasin",
        ("NM", "5"): "tab:green",   ("NM", "10"): "lightgreen",
        ("MM", "5"): "tab:red",     ("MM", "10"): "lightsalmon",
    }

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6
    })

    for model_name_key, entries in sorted(model_groups.items()):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))

        for iter_name, ablation_data in entries:
            iter_num = None
            for allowed in ["5", "10"]:
                if allowed in iter_name:
                    iter_num = allowed
                    break

            for ablation, data in ablation_data.items():
                marker = ablation_markers[ablation]
                colour = combo_colours.get((ablation, iter_num), "tab:grey")
                debug_attempts = data[:, 0]
                complexity = data[:, 1]

                ax.scatter(
                    complexity, debug_attempts,
                    c=colour, marker=marker, s=20, alpha=0.7,
                    label=f"{ablation} (Iter {iter_num})"
                )

        ax.set_xlabel("CC + Nesting Depth")
        ax.set_ylabel("Debug Attempts")
        ax.set_title(f"{model_name_key}", fontsize=9, pad=4)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(),
                  loc="best", fontsize=6,
                  markerscale=0.8, handletextpad=0.3,
                  borderpad=0.3, labelspacing=0.3)

        plt.tight_layout(pad=0.4)

        output_dir = root_dir / "Report_Results"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_dir / f"{model_name_key}_scatter.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


def main():
    # Testing the indexing before doing the scatter plot analysis. This function does not plot anything.
    if len(sys.argv) < 2:
        print("Usage: python Scatter_Plot_Analysis.py <model_name>")
        sys.exit(1)

    CURRENT_DIR = Path(__file__).parent
    model_name = sys.argv[1].lower()

    results = Scatter_Plot_Analysis(CURRENT_DIR, model_name)

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