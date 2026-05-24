import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import csv

def Bootstrap_Data(Input_Folder: Path, Num_Bootstrap: int, LLM_Model: str, LLM_Size: int, Iteration_Count: int):

    if not Input_Folder.exists():
        raise FileNotFoundError(f"The folder {Input_Folder} does not exist.")

    def get_Iter_Dir(root: Path, Iters:int):
        subdirs = [d for d in root.iterdir() if str(Iters) in d.name] # Finds all folders that have the correct iteration number in the name.
        if not subdirs:
            return root
        return get_Iter_Dir(subdirs[0], Iters) # Returns the single subdir with the results for that iteration number.
    
    ablation_dir = get_Iter_Dir(Input_Folder, Iteration_Count)
    print(f"DEBUG: ablation_dir resolved to: {ablation_dir}")
    Study_Folders = [ablation_dir/"NN", ablation_dir/"MN", ablation_dir/"NM", ablation_dir/"MM"]

    for k, item in enumerate(Study_Folders):
        if item.exists():
            if k==0:
                ablation = "NN"
            elif k==1:
                ablation = "MN"
            elif k==2:
                ablation = "NM"
            elif k==3:
                ablation = "MM"

            mean_entries = np.zeros(6) # Initialise an array of zeros.
            upper_CI = np.zeros(6)
            lower_CI = np.zeros(6)

            Bootstrap_Done = False # Reset flags for each ablation.
            Judge_Average = False # Reset flags for each ablation.
            for file in item.iterdir():
                if file.suffix == '.csv' and file.name != "LLM_Judge_Scoring.csv": # Ensures the file is a .csv file and not the othr results file.
                    print(f"DEBUG: Processing file: {file}")

                    # Load the data from the .csv file into pandas dataframe.
                    data = pd.read_csv(file)
                    success_data = data["Success"].to_numpy(dtype=float)

                    # Calculate success rate FIRST (before filtering)

                    success_rate = data["Success"].mean()
                    # Remove unnsuccessful runs (Due to NaN entries)
                    data = data[data["Success"] == 1] # Ensures no NaN entries. Only gathers successful runs for this.

                    # NOTE: Group By Run ID and take run-level mean. Enable/Disable here.
                    #data = data.groupby('Run_ID').mean().reset_index()

                    # Convert to numpy array for bootstrap analysis.
                    data_array = data.to_numpy(dtype=float)

                    # Perform the BCa bootstrap.
                    result = sp.stats.bootstrap((data_array,), np.mean, 
                            n_resamples=Num_Bootstrap, 
                            batch=None, 
                            vectorized=None, 
                            paired=False, 
                            axis=0, 
                            confidence_level=0.95, 
                            alternative='two-sided', 
                            method='BCa', 
                            bootstrap_result=None, 
                            rng=None)
                    # Plot the bootstrap distribution and confidence intervals.
                    bootstrap_distribution = result.bootstrap_distribution
                    ci_lower = result.confidence_interval.low
                    ci_upper = result.confidence_interval.high  

                    columns = data.columns 

                    print("bootstrap_distribution shape:", bootstrap_distribution.shape) 

                    for i, col in enumerate(columns):
                        plt.figure(figsize=(8, 5)) 
                        boot_mean = np.mean(bootstrap_distribution[i])

                        plt.hist(
                            bootstrap_distribution[i], 
                            bins=30,
                            density=True,
                            alpha=0.7
                        )

                        plt.axvline(
                            ci_lower[i],
                            color='red',
                            linestyle='--',
                            label=f'Lower 95% BCa CI = {ci_lower[i]:.3f}'
                        )  

                        plt.axvline(
                            ci_upper[i],
                            color='red',
                            linestyle='--',
                            label=f'Upper 95% BCa CI = {ci_upper[i]:.3f}'
                        ) 

                        plt.axvline(
                            np.mean(data_array[:, i]),
                            color='black',
                            linestyle='-',
                            label=f'Mean = {np.mean(data_array[:, i]):.3f}'
                        )  

                        plt.axvline(
                            boot_mean,
                            color='blue',
                            linestyle=':',
                            label=f'Bootstrap Mean = {boot_mean:.3f}'
                        )

                        plt.title(f'Bootstrap Distribution for {col} ({LLM_Model.upper()} {LLM_Size}B, {Iteration_Count} Iterations)') 
                        plt.xlabel(f'Bootstrap Mean of {col}')  
                        plt.ylabel('Density') 
                        plt.legend()

                        CURRENT_DIR = Path(__file__).parent
                        Output_DIR = CURRENT_DIR/"Report_Results"/ablation
                        if not Output_DIR.exists():
                            Output_DIR.mkdir(parents=True, exist_ok=True)
                        output_path = Output_DIR / f"{LLM_Model}_{LLM_Size}B_{ablation}_{Iteration_Count}-{Iteration_Count}_Bootstrap_{col}.png" 
                        plt.savefig(output_path, bbox_inches='tight') 
                        #print(f"Bootstrap analysis complete. Plot saved to {output_path}.")  
                        plt.close()

                        if col == "Success":
                            idx = 0
                            mean_entries[idx] = success_rate # Store the success rate in the mean entries array.
                        elif col == "Debug_Attempts":
                            idx = 1
                        elif col == "Code_Size":
                            idx = 2
                        elif col == "Cyclomatic_Complexity":
                            idx = 3 
                        elif col == "Nesting_Depth":
                            idx = 4
                        else:
                            idx = None
                        if idx is not None and idx!= 0:
                            mean_entries[idx] = boot_mean
                            lower_CI[idx] = ci_lower[i]
                            upper_CI[idx] = ci_upper[i]
                        if i == len(columns) - 1:
                            Bootstrap_Done = True

                        
                # Calculate the LLM as a Judge Averages for each ablation as we can only write a full row to a .csv file.
                if file.name == "LLM_Judge_Scoring.csv":

                    CURRENT_DIR = Path(__file__).parent
                    Output_DIR = CURRENT_DIR/"Report_Results"/ablation
                    if not Output_DIR.exists():
                        Output_DIR.mkdir(parents=True, exist_ok=True)

                    data = pd.read_csv(file)
                    data_array = data.to_numpy(dtype=float)
                    mean_entries[5] = np.mean(data_array[:, 0]) # Mean down first column
                    Judge_Average = True
                
            if Bootstrap_Done and Judge_Average:
                # Do Success Rate Bootstrap Separately:
                success_result = sp.stats.bootstrap(
                    (success_data,),
                    np.mean,
                    n_resamples=Num_Bootstrap,
                    batch=None,
                    vectorized=None,
                    paired=False,
                    axis=0,
                    confidence_level=0.95,
                    alternative='two-sided',
                    method='BCa',
                    bootstrap_result=None,
                    rng=None)
                
                mean_entries[0] = np.mean(success_result.bootstrap_distribution)
                lower_CI[0] = success_result.confidence_interval.low
                upper_CI[0] = success_result.confidence_interval.high
                headers = ["Success_Rate", "Debug_Attempts", "Code_Size", "Cyclomatic_Complexity", "Nesting_Depth", "LLM-as-a-Judge_Score"]

                with open(Output_DIR/f"{LLM_Model.upper()}-{LLM_Size}B_{Iteration_Count}-{Iteration_Count}.csv", mode='w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                        writer.writerow(mean_entries)

                if Iteration_Count == 5 and LLM_Size == 4:
                    with open(Output_DIR/f"{LLM_Model.upper()}-{LLM_Size}B_{Iteration_Count}-{Iteration_Count}_CI.csv", mode='w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                        writer.writerow(lower_CI)
                        writer.writerow(upper_CI)



if __name__ == "__main__":
    Bootstrap_Data(Path("Qwen_Qwen3.5-27B"), Num_Bootstrap=1000000, LLM_Model="Qwen", LLM_Size=27, Iteration_Count=5)
