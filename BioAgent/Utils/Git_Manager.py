from __future__ import annotations
import atexit
import logging
import uuid
from hashlib import md5
from pathlib import Path
from time import sleep
from types import TracebackType
from typing import Any, ClassVar, Dict, Optional, Type, Union
import json
import os
import subprocess
from Utils.Git_Memory_Handling import GitMemory
import shutil
from autogen.agentchat.group import ContextVariables, ReplyResult

class GitManager:
    """ DocString:
    The GitManager is responsible only for step-level operations. It does the following:
    - Creates Step Folder and Initialises Git in that folder.
    - Uses the GitMemory (Tier 1 Only) to record commit history and fetch history.
    - Manages retrieval of Diffs which are only useful for current step and never needed in future.
    - Adds the .md file with the plan step for readability.
    - Adds the .SUCCESS file or .FAIL file to indicate whether step was successful within the allowed iteration limits.
    """
    def __init__(self, plan_step: Union[int, str], Git_Memory: GitMemory, context_variables: ContextVariables, Hard_Reset:bool=False)-> None:
        # The GitMemory lives above the GitManager, and so only 1 version of GitMemory is needed and 
        # then it is passed to the GitManager to use the Tier 1 features.
        self.context_variables = context_variables # This ensures that the GitManager can access what it needs from shared context variables.
        self.Root_Dir = Path("/workspace")
        self.plan_step = plan_step
        self.repo_path = self.Root_Dir/ "Repos" / f"Step_{self.plan_step}"

        if Hard_Reset:
            self.hard_reset() # This will reset the entire Repos folder to ensure a clean slate for the new run. This is important to ensure that there are no artifacts from previous runs that could affect the new run. The GitMemory should be able to handle this as it will just create new .JSON files for each step and the old ones will be deleted with the Repos folder reset.
        else:
            pass

        #1-- Create Folder if it doesn't exist yet.
        if not self.repo_path.exists():
            self.repo_path.mkdir(parents=True, exist_ok=True)
        else:
            shutil.rmtree(self.repo_path, ignore_errors=True)
            self.repo_path.mkdir(parents=True, exist_ok=True)
            # If folder already exists (artifact from previous run), delete and remake the folder to ensure a clean slate for the new run.
        #2-- Initialise the **Existing** GitMemory with this step.
        self.Git_Memory = Git_Memory
        self.Git_Memory.set_step(self.plan_step, self.repo_path) # This will set the step for the GitMemory and load any existing memory for that step if it exists.
        #3-- Initialise Git Repo and Add the .md file.
        self._initialise_repo()
    
    def _git_commands(self, commands: list[str], capture: Optional[bool]=False) -> str:
        """ DocString:
        This method is used to run git commands in the terminal by injecting teh required command
        into a subprocess call. This is a private helper method that is used to help with other methods
        that need to run git commands frequently.
        The commands are in the form ["git", "add", " ."], etc...
        """
        result = subprocess.run(
        commands,
        cwd=self.repo_path,
        capture_output=capture,
        text=True,
        check=True)
        return result.stdout if capture else ""

    def _initialise_repo(self)-> None:
        """ DocString:
        This method allows a repo to be initialised within a certain step folder. This is crucial for
        keeping track of changes for code within a given step implementation.
        """
        # Initialise Repo
        self._git_commands(["git","init"])
        self._git_commands(["git", "config", "user.name", "BioAgent"]) # Required to enable commits (Fake Username).
        self._git_commands(["git", "config", "user.email", "bioagent@local"]) # Required to enable commits (Fake Email).

        if isinstance(self.plan_step, int):
            # Add the .md file with the plan step for readability.
            plan_dict = self.context_variables.get("Plan", {})
            plan_section = plan_dict.get("Plan_Section", [])
            
            # Find this step's details
            step_data = None
            for step in plan_section:
                if step["Step_Number"] == self.plan_step:
                    step_data = step
                    break
            
            if step_data is None:
                raise ValueError(f"Step {self.plan_step} not found in plan")
            
            # Format as markdown
            markdown_lines = [
                f"# Step {step_data['Step_Number']}: Analysis Plan",
                "",
                f"## Analysis Type",
                step_data['Analysis_Type'],
                "",
                f"## Data File",
                f"`{step_data['Data_File']}`",
                "",
                f"## Variables",
                ", ".join(f"`{var}`" for var in step_data['Variables']),
                "",
                f"## Context",
                step_data['Context'],
                "",
                f"## Output Format",
                step_data['Output_Format'],
                "",
                f"## Output Details",
                step_data['Output_Details'],
            ]
            
            # Write to file
            plan_file = self.repo_path / f"Step_{self.plan_step}_Plan.md"
            plan_file.write_text("\n".join(markdown_lines))
        else:
            plan_file = self.repo_path / f"{self.plan_step}.md"
            plan_file.write_text("Placeholder to ensure Repo not empty.")
            
        
        # Commit the plan file
        self.commit(approach="Initial Commit with Plan", outcome="Success", error_message=None)

    def commit(self, approach: Optional[str], outcome: Optional[Union[str, dict]], error_message: Optional[str])-> None:
        """ DocString:
        This method allows commits to be made to the git repo for the current plan step.
        The commit message is recorded in the GitMemory successfully.
        """
        commit_message= f"Approach: {approach if approach else 'None'}\nOutcome: {outcome if outcome else 'None'} \nError Message: {error_message if error_message else 'None'}"
        self._git_commands(["git", "add", "."])
        self._git_commands(["git", "commit", "--allow-empty", "-m", commit_message])

        # Store Commit as History in GitMemory
        self.Git_Memory.record(Cycle_Number=len(self.Git_Memory.Step_Data["Cycles"])+1, Approach=approach, Outcome=outcome, Error_Message=error_message)

    def retrieve_history(self)-> str:
        """ DocString:
        This method allows for retreival of the LLM-readable history so that the LLM can learn 
        from the past changes and mistakes so it can converge to a solution more quickly.
        (Debug Agent and Control Agent)
        """
        history= self.Git_Memory.history()
        return history

    def retrieve_diff(self)-> str: 
        
        """ DocString: This method allows the (Control Agent) to make sure 
        that major code refactors are not done. Debugging should only produce 
        small changes between commits to ensure that the original plan is still 
        followed. If the control agent decides that the diff is too large and 
        changes the plan, then it can rollback to the prevous commit and tell the 
        debug agent to try again with a smaller change.
        """
        # File patterns to exclude
        exclude_types = ["*.log", "*.json"]  # Add more if needed
        # Base git diff command (HEAD vs previous commit)
        diff_command = ["git", "diff", "HEAD~1", "HEAD"]
        # Add exclusions
        diff_command += [f":(exclude){pattern}" for pattern in exclude_types]
        # Run command
        diff = self._git_commands(diff_command, capture=True)
        return f"Diff from last commit:\n{diff}"

    
    def rollback(self):
        """ DocString:
        This method allows the control agent to roll the code back to the previous commit if necessary.
        The primary reason is if the diff is too large or it goes in the wrong direction. Other reasons
        and more complex rollback strategies can be implemented in future iterations.
        """
        # ------------- Checking How Many Commits Exist Before Rolling Back -------------
        count = self._git_commands(["git", "rev-list", "--count", "HEAD"], capture=True)
        if int(count.strip()) < 4:
            print("Not enough commits to rollback. Rollback aborted.")
            return # This stops the Controller Agent from trying to rollback to before any code exists or any commits were made.

        # Rollback the git repo to the previous commit.
        self._git_commands(["git", "reset", "--hard", "HEAD~1"])
        print("Rollback successful. Reverted to previous commit.") # Helps debugging LLM based MAS.
        # Note: The rollback action will affect the .JSON file that the GitMemory writes to. The 
        # GitMemory must have a rollback recorded to ensure it stays up to date with the actual state
        # of the repo. The .JSON file is overwritten anyway after saving.
        self.Git_Memory.record(Rollback=True) # This MUST come after rolling git repo back to ensure the .JSON file
        # is written to LAST.
    def record_result(self, result:str):
        """ DocString:
        This is to record the final result of the step in the Repo to indicate if the step was successful or
        aborted.
        -- .SUCCESS file ==> Step was successful and obtained error free run.
        -- .FAIL file ==> Step was unsuccessful and the entire plan step was aborted after reaching iteration
                          limit.
        """
        if result == "Success":
            with open(self.repo_path / ".SUCCESS", "w") as f:
                pass
            print("Step marked as SUCCESS. Moving to next step in the plan.")
            self.commit(approach="Step Completed Successfully and .SUCCESS file added", outcome="Success", error_message=None)
        elif result == "Fail":
            with open(self.repo_path / ".FAIL", "w") as f:
                pass
            print("Step marked as FAIL. Moving to next step in the plan.")
            self.commit(approach="Step Failed and .FAIL file added", outcome="Success", error_message=None)
        else:
            raise ValueError("Result must be either 'Success' or 'Fail'.")
    def write_code(self, code:str):
        """ DocString:
        The GitManager must write code to the code file prior to calling commits. This ensures that the 
        latest code is always in the repo. This is to synchronise the repo with the AG2 context variables
        that the LLMs have access to.
        """
        code_file = self.repo_path / "Code.py"
        with open(code_file, "w") as f:
            f.write(code)
        print("Code written to file successfully. Ready for commit.")
    def read_code(self)-> str:
        """ DocString:
        This is to allow the AG2 to read the latest code from the repo to ensure synchronisation between
        the repo and the AG2 context variables. This is useful for the Control Agent to be able to read
        the latest code and make decisions based on that.
        """
        code_file = self.repo_path / "Code.py"
        if code_file.exists():
            with open(code_file, "r") as f:
                code = f.read()
            return code
        else:
            print("Code file does not exist yet.")
            return ""
    def hard_reset(self):
        """ DocString:
        Between Iterations, the GitManager should be able to rese the entire Repos folder.
        This makes sure that there is a clean slate for each iteration with no artifacts from the last.
        Any lessons that need to be learned should come from the GitMemory Medium-Term or Long-Term memory
        in Tiers 2 and 3.
        """
        #-- Check if Repos Folder Exists. If not create it. f it does, delete contents and remake. (Hard Reset)
        if not (self.Root_Dir / "Repos").exists():
            (self.Root_Dir / "Repos").mkdir(parents=True, exist_ok=True)
        else:
            shutil.rmtree(self.Root_Dir / "Repos", ignore_errors=True)
            (self.Root_Dir / "Repos").mkdir(parents=True, exist_ok=True)
        print("Reset of Repos folder Successful. Clean slate for new iteration.")
    
    # =================== Planning / Coding Loop Specific ======================
    # There needs to be a reminder function that means that the GitMemory can be used for Tier 1.
    # During the planning and review cycle, the GitMemory will need to point to different folders constantly.
    # This is because the code will cycle through Reviewer1 -> Reviewer2 -> .... -> Plan Fixer -> [Repeat].

    def jog_memory(self):
        """ DocString:
        This is to jog the GitMemory to point to the correct repo for the current reviewer whilst still
        taking advantage of the full Tier 1 memory features. This is done by callng set_step() so that
        the GitMemory loads the relevant .JSON file and then it will work with the previous GitManager methods 
        that are available.
        """
        self.Git_Memory.set_step(self.plan_step, self.repo_path)
        print(f"GitMemory jogged to Step {self.plan_step} repo successfully.")
    
    def write_plan(self, plan: Union[str, dict]):
        """ DocString:
        The GitManager must write plan to the plan file prior to calling commits. This ensures that the 
        latest plan is always in the repo. This is to synchronise the repo with the AG2 context variables
        that the LLMs have access to.
        """
        plan_file = self.repo_path / "Plan.md"
        with open(plan_file, "w", encoding="utf-8") as f:
            if isinstance(plan, dict):
                json.dump(plan, f, indent=2, ensure_ascii=False)
            else:
                f.write(plan)

        print("Plan written to file successfully. Ready for commit.")
    def read_plan(self)-> str: # Only needed if rollback is implemented. For now, rollback will not be used.
        """ DocString:
        This is to allow the AG2 to read the latest plan from the repo to ensure synchronisation between
        the repo and the AG2 context variables. This is useful for the Control Agent to be able to read
        the latest plan and make decisions based on that.
        """
        plan_file = self.repo_path / "Plan.md"
        if plan_file.exists():
            with open(plan_file, "r") as f:
                plan = f.read()
            return plan
        else:
            print("Plan file does not exist yet.")
            return ""


def main():
    context_variables = ContextVariables({
        "Plan": {'Plan_Section': [{'Step_Number': 1, 'Analysis_Type': 'Peak Detection and Baseline Correction', 'Data_File': 'chromatography_combined.csv', 'Variables': ['UV_1_280_mAU', 'run_no', 'Fraction_unique_ID'], 'Context': 'Use the `hplc-py` or `MOCCA` package for peak detection and baseline correction. Ensure that the peaks are accurately identified and the baseline is corrected to remove any drift. The data may contain noise, so apply the Savitzky-Golay filter for noise reduction. Specify the parameters for the Savitzky-Golay filter (e.g., window size and polynomial order) to ensure consistency and reproducibility. Use `run_no` to ensure consistency across all steps.', 'Output_Format': 'Visualisations', 'Output_Details': 'Save the chromatogram with detected peaks and corrected baseline as an image file. Include labels for the x-axis (time in minutes) and y-axis (mAU). Title the graph with the run number, fraction ID, and date of the run. Add a legend to the graph to distinguish between the original chromatogram, the detected peaks, and the corrected baseline. Clearly mark the peaks with vertical lines and labels.'}, {'Step_Number': 2, 'Analysis_Type': 'Peak Integration and Quantification', 'Data_File': 'chromatography_combined.csv', 'Variables': ['UV_1_280_mAU', 'run_no', 'Fraction_unique_ID'], 'Context': 'Use the peak area for quantification. Apply the internal standard method to reduce variability. Ensure that the peak areas are accurately calculated and compared against known standards. Specify how the internal standard will be used (e.g., the concentration of the internal standard, how it will be added to the samples, and how it will be used in the quantification process). Use `run_no` to ensure consistency across all steps.', 'Output_Format': 'Text based outputs', 'Output_Details': 'Create a new section in the `Usability.md` file for each step. Use headers to clearly label each section, such as `## Peak Integration and Quantification for Run <run_no>`. Include a table format for the peak areas and retention times, which will make the data easier to read. Ensure that the table is well-formatted with appropriate column headers and spacing. Consider increasing the limit of 20 lines per step or providing a summary of the results if the number of peaks is large.'}, {'Step_Number': 3, 'Analysis_Type': 'UV Spectrum Analysis and Retention Time Comparison', 'Data_File': 'MS_combined.csv', 'Variables': ['Observed_TIC_RT_(mins)', 'Observed_UV_RT_(mins)', 'Observed_neutral_mass_(Da)', 'Sample_Code', 'run', 'chromatography_stage'], 'Context': 'Compare the observed retention times with known standards. Use UV spectrum analysis to enhance compound identification. Ensure that the data is aligned with the chromatographic data from the previous steps. Verify that the data file used (`MS_combined.csv`) contains the necessary UV spectrum data. If not, obtain it from the appropriate source. Use `run` to ensure consistency across all steps.', 'Output_Format': 'Visualisations', 'Output_Details': 'Save a graph showing the UV spectrum for each peak. Include labels for the x-axis (wavelength in nm) and y-axis (intensity). Title the graph with the sample code, run number, and chromatography stage. Use subplots to combine the UV spectrum and the retention time comparison in a single image file. Ensure that the UV spectrum and retention time comparison are clearly labeled and distinguishable. Use a consistent color scheme for the UV spectrum and retention time comparison. Additionally, include a legend to explain the different elements in the plot. Save the table of retention times and their corresponding known standards as a text file or CSV file, not just an image file, to facilitate further analysis and data sharing.'}], 'Number_of_Steps': 3},
    })
    Root_Dir = Path("/workspace")
    Repo_Dir = Root_Dir / "Repos"
    Git_Memory = GitMemory(Root_Dir)
    Git_Manager = GitManager(plan_step=1, Git_Memory=Git_Memory, context_variables=context_variables, Hard_Reset=True)

    #Git_Manager.commit(approach="Initial Commit with Plan", outcome="Success", error_message=None)
    #print(Git_Memory.history())

    with open(Repo_Dir/"Step_1"/"Code.py", "w") as f:
        f.write("print('Hello World')")
    #2nd Commit (The FIRST COMMIT IS the .md file in the initialise_repo method).
    Git_Manager.commit(approach="Added Python Code File", outcome="Success", error_message=None)

    with open(Repo_Dir/"Step_1"/"Code.py", "w") as f:
        f.write("print('Hello World') \nprint('This is a test.')")
    #3rd Commit
    Git_Manager.commit(approach="Added test string", outcome="Success", error_message=None)

    with open(Repo_Dir/"Step_1"/"Code.py", "w") as f:
        f.write("print('Hello World') \nprint('This is a test.') \na=5 \nb=10 \nprint(a/b)")
    #4th Commit
    Git_Manager.commit(approach="Added variables for divide function test", outcome="Success", error_message=None)

    print(Git_Manager.retrieve_history())
    print(Git_Manager.retrieve_diff())

    # Example: Error Identified. Need to rollback the entire repo.
    Git_Manager.rollback()
    print("\n--------------------------- ROLLBACK ----------------------------------------\n")
    print(Git_Manager.retrieve_history())
    print(Git_Manager.retrieve_diff())

    print("\n =================== Next Commit After Rollback ===================\n")

    with open(Repo_Dir/"Step_1"/"Code.py", "w") as f:
        f.write("print('Hello World') \nprint('This is a test.') \ndef divide(a,b): \n return a/b")
    #5th Commit
    Git_Manager.commit(approach="Created Divide Function", outcome="Success", error_message=None)
    print(Git_Manager.retrieve_history())
    print(Git_Manager.retrieve_diff())

    Git_Manager.record_result("Success") 




if __name__ == "__main__":
    main()

# Unit Tests Completed:
#1 - GitManager successfully creates a new folder for the plan step, initialises a git repo and commits the
#    .md file with the plan details.

#2 -- GitManager Commit method is successful. The GitMemory history is shown to be successsfully updated with
#     the commit details and it is easily retrieved as an LLM-readable format.

#3 -- GitManager retrieve history and retrieve diff methods successfully obtain the commit hisotry and the
#     diffs between each commit n a readable format. The diffs show the changes between each commit to ensure
#     changes are small so that the plan is being followed without major refactors. This also helps the control
#     agent steer the debugging effectively.

#4 -- GitManager rollback method successfully rolls back to the previous commit and the GitMemory is updated to 
#     reflect the rollback action. The history and diffs show that the repo has been rolled back to the previous commit.

#5 -- GitManager record_result method successfully creates the .SUCCESS file and commits this to the repo to indicate that 
#   the step was successful. The commit message reflects this action and the history shows the final commit with the .SUCCESS 
#   file added. The same process works for recording a "Fail" result with the .FAIL file.

#6 -- GitManager hard reset method successfully resets the entire Repos folder to ensure a clean slate for a new iteration. 
#     The GitMemory should be able to handle this as it will just create new .JSON files for each step and the old ones will 
#     be deleted with the Repos folder reset.

#7 -- In Terminal, the cd command has been used to navigate into the Repos folder and the folders within this drectory to check that
#     the git repo is successfully setup and constrained to the folder for the current plan step.






