import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV files
csv_files = [
    "Usability_Plan_Scoring.csv",
    "HPLC_Plan_Scoring.csv",
    "MS_Plan_Scoring.csv"
]

labels = ["Usability", "HPLC", "MS"]
output_image = "Results.png"

# Read files and compute row-wise means
row_means_list = []
for file in csv_files:
    df = pd.read_csv(file, header=None)
    row_means_list.append(df.mean(axis=1))

# Assume all files have the same number of rows
num_rows = len(row_means_list[0])
row_numbers = np.arange(1, num_rows + 1)

# Bar settings
bar_width = 0.25

plt.figure(figsize=(10, 5))

# Plot each dataset
for i, row_means in enumerate(row_means_list):
    plt.bar(
        row_numbers + i * bar_width,
        row_means,
        width=bar_width,
        label=labels[i]
    )

# Axis labels and formatting
plt.xlabel("Planning Iteration")
plt.ylabel("Scores")
plt.title("Planning Iteration Scores")
plt.xticks(row_numbers + bar_width, row_numbers)
plt.legend()

plt.tight_layout()
plt.savefig(output_image, dpi=300)
plt.close()