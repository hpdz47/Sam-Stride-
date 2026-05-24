from Bootstrap_Analysis import Bootstrap_Data
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import shutil
from Judge import LLM_Judge, Evaluate, Average_Score
from Bar_Chart_Analysis import Bar_Chart_Plotting
from Scatter_Plotting_Analysis import Scatter_Plot_Analysis
from Violin_Plotting import Violin_Plots

if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("\n=============\nRunning Analysis \n============= \n")

    if len(sys.argv) < 2:
        print("Usage: python your_script.py <folder_path>")
        sys.exit(1)
    # Finding the correct folder path to be used with the evaluation code.
    CURRENT_DIR = Path(__file__).parent
    model_name = sys.argv[1].lower()
    model_size = int(sys.argv[2]) # NOTE: Must be a number
    iteration_count = int(sys.argv[3]) # NOTE: Must be a number
    Calc = sys.argv[4].lower()
    plot_ablations=["NN", "MN", "NM", "MM"]

    if not Calc == ("R").lower():
        if len(sys.argv) > 5:
            raise ValueError("Too many arguments provided. Extra arguments only allowed when re-plotting")
        # ================= Starting LLM as a Judge evaluation, bootstrap as normal ========================
        if not type(model_size) == int and not type(iteration_count) == int:
            raise ValueError("Model size and iteration count must be integers.")
        Possible_Files = []
        for files in CURRENT_DIR.iterdir():
            if files.is_dir():
                if model_name in files.name.lower() and str(model_size) in files.name:
                    Possible_Files.append(files)
        
        if len(Possible_Files) == 0:
            raise ValueError(f"No folders found for model name: {model_name}. Please check the folder names and try again.")
        
        for item in Possible_Files:
            for f in item.iterdir():
                if str(iteration_count) in f.name:
                    folder_path = f
                    break
            else:
                raise ValueError(f"Only the following folders could be found for your model and size: {Possible_Files}")
        # NOTE: Avoidng putting the evaluation code in a for loop to mitigate risk of over-calling LLM.
        # Starting the LLM-as-a-Judge evaluation
        x = Evaluate(folder_path, iteration_count) # Run the evaluation and store the final scores in a .csv file.
        # Bootstrap Analysis:
        Bootstrap_Data(folder_path, Num_Bootstrap=1000000, LLM_Model=model_name, LLM_Size=model_size, Iteration_Count=iteration_count)
        # Plotting for the final report:
        if Calc == "y":
            CURRENT_DIR = Path(__file__).parent
            RESULT_DIR = CURRENT_DIR / "Report_Results"
            if RESULT_DIR.exists():
                # Bar Charts:
                Bar_Chart_Plotting(RESULT_DIR)
                # Scatter Plot Analysis:
                Scatter_Plot_Analysis(CURRENT_DIR, model_name, plot_ablations)
                # Violin Plot Analysis:
                Violin_Plots(CURRENT_DIR, model_name, plot_ablations)
        else:
            print("Plotting skipped.")
        #==================================================================================================================
    elif Calc == ("R").lower():
        CURRENT_DIR = Path(__file__).parent
        RESULT_DIR = CURRENT_DIR / "Report_Results"
        if not RESULT_DIR.exists():
            raise ValueError("Report_Results folder not found. Please run the full analysis first to generate the necessary .csv files for re-plotting.")
        
        # Re-Calculating the power mean and re-plotting all figures.
        print("Re-calculating power mean and re-plotting all figures.")

        allowed_sizes = ["4","27"]
        allowed_iters = ["5","10"]
        Possible_Files = []
        for files in CURRENT_DIR.iterdir():
            if files.is_dir():
                if model_name in files.name.lower() and any(size in files.name for size in allowed_sizes):
                    Possible_Files.append(files)
        if len(Possible_Files) == 0:
            raise ValueError(f"No folders found for model name: {model_name}. Please check the folder names and try again.")
        
        # Currently Obtained folder paths: [Model_4B, Model_27B]
        folder_paths = []
        for item in Possible_Files:
            for f in item.iterdir():
                if any(iter_count in f.name for iter_count in allowed_iters):
                    folder_paths.append(f)
            
        # NOTE: Folder_Paths = [Model_4B_5_Iters, Modl_4B_10_Iters, Model_27B_5_Iters]

        # Setting the Power mean value
        p = int(sys.argv[5]) # New power mean parameter.
        # Re-calculating using the new power mean parameter. NOTE: Need to loop through all folders to re-calculate for ALL data.
        print(f"\nFolder_PATHS_TEST\n{folder_paths}\n")
        for scoring_paths in folder_paths:
            iteration_count_match = next((count for count in allowed_iters if count in str(scoring_paths.name)), None)
            Average_Score(scoring_paths, p, int(iteration_count_match))
        # All LLM as a Judge power means calculated and stored in the LLM_Judge_Scoring.csv fle for each model siz and ablation.
        # ----------------------------------------------------------------
        # As overall averaging of the LLM as a Judge scorng is done in the Bootstrapping function, need to bypass this to avoid re-running bootstrap.
        for MemAbl in folder_paths: # IE: Qwen_YB/Memory_Abaltion_X_X/
            match_size = next((sz for sz in allowed_sizes if sz in MemAbl.parent.name), None)
            match_iter = next((its for its in allowed_iters if its in str(MemAbl.name)), None)

            # Finding the LLM as a Judge file that contains a vector of all power means.
            for ablation in ["NN", "MN", "NM", "MM"]:
                judge_file = MemAbl / ablation / "LLM_Judge_Scoring.csv"
                summary_dir = RESULT_DIR / ablation
                if judge_file.exists() and summary_dir.exists():
                    new_mean = pd.read_csv(judge_file)["Average_Score"].mean()
                    # Finding the correct csv file to overwrite the LLM Judge Scoring value.
                    for csv_file in summary_dir.glob("*.csv"):
                        if match_size in csv_file.name and match_iter in csv_file.name:
                            df = pd.read_csv(csv_file)
                        else:
                            continue
                        if "LLM-as-a-Judge_Score" in df.columns:
                            df["LLM-as-a-Judge_Score"] = new_mean
                            df.to_csv(csv_file, index=False)

        # Re-Plotting Bar Charts:
        Bar_Chart_Plotting(RESULT_DIR) # RESULTS_DIR Must exist to have got this far.

        # Scatter Plot Analysis:
        Scatter_Plot_Analysis(CURRENT_DIR, model_name, plot_ablations)

        # Violin Plot Analysis:
        Violin_Plots(CURRENT_DIR, model_name, plot_ablations)
