# Imports ---------------------------------------
from Chatrooms.Data_Discovery import EDA_Analysis
from Chatrooms.Focus_Area_and_Research import Focus_Area_and_Research_Chatroom
from Chatrooms.Planning import Planning_Chatroom
from autogen.agentchat.group import ContextVariables
from typing import Optional
import shutil
from pathlib import Path
import json
from Utils.Git_Memory_Handling import GitMemory
from Utils.Git_Manager import GitManager
from Chatrooms.Implementation import Implementation_Loop
from Chatrooms.Reporting import Final_Report
from Config.vLLM_Manager import LLM_Manager
#-------------------------------------------------
class BioAgent:
    def __init__(self, Deep_Research_Mode: bool, Max_Plan_Steps: int, Implementation_Granularity: int, User_Requirements: str, Max_Review_Loops:int, Max_Retries:int, Max_Plan_Reviews:int, Plan_Short_Term_Memory: bool, Code_Short_Term_Memory: bool, Max_Word_Count:int, Max_Images: int, Max_Markdown_Files: int, Max_Markdown_Lines: int, Run_ID: int):
        """
        Docstring:
        - Deep Research Mode: (true/false). If true, then the Deep Research Agent will be used to gather some 
                             ChatGPT-style Deep research using the AG2 deep research tool. This feature is optional
                             as  can be very time consuming. If possible, alternative documentation is recommended or
                             run the deep research once and store the results for future runs.

        - Max Plan Steps: The maximum number of steps that the Planning Agent can use in the plan. If this is set too high,
                          then accuracy may be reduced and time taken to implement will increase.

        - Implementation Granularity: This controls how many steps are given to the Data Analysis Coding system at once.
                                      If set to 1, then one step will be given at a time. It s recommended to keep this at 1
                                      for best results, but increasing this may reduce time taken at the cost of accuracy.
        
        - User Requirements: The requirements for the data analysis project provided by the user. This should be a detailed 
                             description of the requirements and can include specific instructions, goals, datasets, or any 
                             other relevant information that will guide the planning and implementation of the data analysis project.
        """
        self.context_variables = ContextVariables({
            #---- Discovery Phase Variables:
            "EDA_Results": [],
            "EDA_Interpretation": "",
            "Interpretation_Available": False,
            "metadata": {},
            #---- Focus Area and Research Variables:
            "User_Requirements": "", # Import from text file to represent a user uploading requirements. This would be used by a User in a ChatGPT-style interface.
            "Focus_Area_Statement": "",
            "FA_Available": False,
            "FA_RAG_QA": "",
            "FA_RAG_Available": False,
            "FA_RAG_Interpretation": "",
            "FA_RAG_Interpretation_Available": False,
            #-------- Planning Variables:
            "Plan": {},
            "Plan_Updated": False,
            "Plan_Fixed": False,
            "Plan_Feedback": "",
            "Feedback_Available": False,
            "Max_Plan_Steps": 0,
            "All_Plans":[],
            # Max_Plan_Reviews tracking:
            "Plan_Review_Count": 0,
            "Max_Plan_Reviews": Max_Plan_Reviews,
            # Plan Diff Tracking:
            "Plan_Approach": {},
            "Plan_Diffs": "",
            "Plan_Diff_Reviewed": False,
            #-------- Review Panel Variables:
            "Var_Review": {},
            "Var_Review_Available": False,
            "FA_Reviews": {},
            "FA_Reviews_Available": False,
            "Review_RAG_QA": "",
            "Review_RAG_Available": False,
            "Review_RAG_Interpret": "",
            "Review_RAG_Interpret_Available": False,
            "Review_OP_Instruction": {},
            "OP_Instruction_Available": False,
            "Plan_Feedback": {},
            "Review_Compilation_Complete": False,
            "Context_Review": {},
            "Context_Review_Available": False,
            "Reviews": {},
            "Scoring_Complete": False, 
            "Idx": 0, 
            "Plan_History": {}, 
            "Specialist_Reviews": {},
            "Specialism": "",
            "Reviews_Adapted": False,
            "Plan_Fix_Approach": "First plan commit. Plan created. No previous plan versions to apply fixes to.",
            "Var_Learnings": {},
            "FA_Learnings": {},
            "OP_Learnings": {},
            "Context_Learnings": {},
            "Var_Improvements": {},
            "FA_Improvements": {},
            "OP_Improvements": {},
            "Ctx_Improvements": {},
            "Specialist_Improvements": {},
            "Specialist_Learnings": {},

            #------- Implementation Phase Context Variables:
            "Max_Review_Loops": Max_Review_Loops,
            "Plan_Section": {},
            "Code_Review_Count": 0, # This must start at 0 to make scoring numbers work correctly.
            "Code": "",
            "Code_Errors": "",
            "Plan_Enforcement":"",
            "Optimisation_Goals":"",
            "Error_Summarised": False,
            "Success": False,
            "Summary":"",
            "Approach":"Initial code commit.",
            "Outcome":"",
            "Rollback_Count": 0, # Gives Total Rollbacks in code run. This is to assess the usage of Git and determine how agents are learning.
            "Max_Retries": Max_Retries,
            "History": "",
            "Diff": "",
            "Run_ID": Run_ID,
            "Suggestions": "",
            "Controller_Finished": False,
            "Rollback": False,
            "Debug_Finished": False,
            #-------- Code Git Based Loops:
            "Code_Updated": False,
            "Code_Diffs_Analysed": False,
            "Code_Fixed": False,
            "Code_Diffs": "",
            "Code_Approach": "",
            "Code_Specialist_Learnings": "",
            "Code_History": "",
            "Code_Specialist_Reviews": "",
            "Code_Specialist_Improvements": "",
            "PEL_Learnings": "",
            "PEL_Improvements": "",
            "ECL_Learnings": "",
            "ECL_Improvements": "",
            "OL_Learnings": "",
            "OL_Improvements": "",
            "Code_Fix_Approach": "First code commit. Code created. No previous code versions to apply fixes to.",
            "Code_Reviews_Adapted": False,
            "Code_Reviews": "",
            #-------------- Code Review Panel Context Variables:
            "Code_Errors": "",
            "Errors_Checked": False,
            "Plan_Enforcement": "",
            "Enforcement_Complete": False,
            "Optimisation_Goals": "",
            "Optimisation_Assessed": False,
            # ---- Pip Manager Context Variables:
            "Packages_Managed": False,
            "Pip_Code": "",
            "Pip": False,
            #------- Error Analyst Context Variables:
            "Failure_Analysed": False,
            "Instructions": "",
            #-------- Planning GitManager Keys:
            "Var_Manager": None,
            "FA_Manager": None,
            "Output_Manager": None,
            "Context_Manager": None,
            "Plan_Manager": None,
            # ------------ Coding GitManager Keys:
            "ECL_Manager": None,
            "PEL_Manager": None,
            "OL_Manager": None,
            "Code_Manager": None,
            # Memory Variables:
            "Short_Term_Memory": Plan_Short_Term_Memory,
            "Code_Short_Term_Memory": Code_Short_Term_Memory,
            # Reliability Improvement
            "Max_Words": Max_Word_Count, # This is passed to agents in an attempt to prevent large verbose outputs that often cause system failure, especially when using smaller models.
            # -------------------- Review Stage
            # VL Model
            "Image_Analysis": [],
            "Current_Image_Name": "",
            "Max_Images": Max_Images,
            # Markdown Model
            "Markdown_Analysis": [],
            "Markdown_File": "",
            "Max_Markdown_Files": Max_Markdown_Files,
            "Max_Markdown_Lines": Max_Markdown_Lines,
            # Report_Writer
            "Final_Report":"",
            "Image_List":[],
        })
        self.context_variables["User_Requirements"] = User_Requirements
        self.Deep_Research_Mode = Deep_Research_Mode
        self.Max_Plan_Steps = Max_Plan_Steps
        self.Implementation_Granularity = Implementation_Granularity
        self.Max_Review_Loops = Max_Review_Loops
        self.Max_Retries = Max_Retries
        self.Max_Plan_Reviews = Max_Plan_Reviews
        self.Plan_Short_Term_Memory = Plan_Short_Term_Memory
        self.Code_Short_Term_Memory = Code_Short_Term_Memory
        self.Max_Images = Max_Images
        self.Max_Markdown_Files = Max_Markdown_Files
        self.Max_Markdown_Lines = Max_Markdown_Lines

        # Add Context Variables for Scoring:
        # 1) ----- Coding Stage Scoring:
        self.Code_Review_Panel_Agent_Count = 3 # NOTE: This must be adjusted if there are more review panel agents added.
        # Current Indexing:
        # 0 => Error_Checker
        # 1 => Plan_Enforcer
        # 2 => Optimiser
        for idx in range(self.Max_Review_Loops):
            self.context_variables[f"Code_Score_{idx}"] = [0]*self.Code_Review_Panel_Agent_Count # Creates a list of 0s with the number of places = number of agents.
        
        # 2) ----- Planning Stage Scoring:
        self.Plan_Review_Panel_Agent_Count = 4 # This must be updated if more review panel agents added.
        # Current Indexing:
        # 0 => Variable_Checker
        # 1 => Focus_Area_Assessor
        # 2 => Review_OP
        # 3 => Compiler
        for idx in range(self.Max_Plan_Reviews):
            self.context_variables[f"Plan_Score_{idx}"] = [0]*self.Plan_Review_Panel_Agent_Count # This is a single score given by the planner to the plan after each review.

        # ========== Git Memory Initalisation ===========
        # NOTE: GitMemory is a Singleton pattern class as it must exist globally to manage memory across runs.
        Root_Dir = Path("/workspace")
        self.Git_Memory = GitMemory(Root_Dir)
        # This Instance MUST be passed to all relevant classes that require access.
        #================================================

    def run(self):
        #1. Data Discovery Phase:
        EDA=EDA_Analysis(context_variables=self.context_variables, Max_Rounds=10)
        EDA.run_Conversation()
        #2. Focus Area and Research Phase:
        FA=Focus_Area_and_Research_Chatroom(context_variables=self.context_variables, Deep_Research_Mode=self.Deep_Research_Mode, Max_Rounds=20)
        FA.run_Conversation()
        #3. Planning Phase:
        Plan=Planning_Chatroom(context_variables=self.context_variables, Max_Plan_Steps=self.Max_Plan_Steps, Max_Rounds=80, Max_Plan_Reviews = self.Max_Plan_Reviews, Review_Agent_Number = self.Plan_Review_Panel_Agent_Count, Git_Memory = self.Git_Memory)
        Plan.run_Conversation()
        #4. Implementation Phase:
        Implementation = Implementation_Loop(context_variables = self.context_variables, Max_Retries = self.Max_Retries, Max_Review_Loops = self.Max_Review_Loops, Git_Memory = self.Git_Memory, Review_Agent_Number = self.Code_Review_Panel_Agent_Count)
        Implementation.solve()
        #5. Report Writing Phase:
        Report = Final_Report(context_variables = self.context_variables, Max_Rounds = 10, Max_Images = self.Max_Images, Max_Markdown = self.Max_Markdown_Files, Max_MD_Lines = self.Max_Markdown_Lines)
        Report.generate_report()

        # NOTE: Commented out the first 3 stages to do testing on the Implemnetation phase.
        # NOTE: Added Necessary Context variables to allow for code to be written.
        # NOTE: Context Variables added to are: Plan,
    
    def cleanup(self):
        # Cleanup files that are not part of the results or deep research.
        allowed_dir = ["Research_and_Documents", "Results", "logs"]
        self.CLEANUP_DIR = Path("/workspace")
        for item in self.CLEANUP_DIR.iterdir():
            if item.name not in allowed_dir:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    print(f"Failed to reset workspace. \n{e}")
        # Cleanup the indexed vector database as it is indexed in every run.
        DEL_DIR = self.CLEANUP_DIR/"Vector_Database"
        if DEL_DIR.exists():
            try:
                shutil.rmtree(DEL_DIR)
            except Exception as e:
                print(f"Failed to reset Vector Database. \n{e}")

        
def main():
    
    hplc_analysis_tasks = [
    "Detect and list all chromatographic peaks present in the HPLC data along with their retention times.",
    "Calculate the integrated peak areas and estimate relative compound abundances from the HPLC chromatogram.",
    "Evaluate the overall signal-to-noise ratio across the chromatogram and summarize baseline stability.",
    "Identify any unusual spikes, missing peaks, or outliers that could indicate experimental or detector issues.",
    "Assess whether the chromatographic data quality is sufficient for downstream statistical or machine learning analysis.",
    "Determine peak widths, heights, and symmetry to evaluate chromatographic peak shape quality.",
    "Estimate the baseline noise level and determine whether it significantly affects peak detection.",
    "Identify overlapping or poorly resolved peaks that may require improved chromatographic separation.",
    "Calculate the relative proportion of each detected compound based on peak area normalization.",
    "Evaluate chromatogram drift over time to detect potential detector instability or column issues.",
    "Identify minor peaks that may represent trace impurities or degradation products.",
    "Assess peak resolution between neighboring peaks to determine separation efficiency.",
    "Estimate retention time reproducibility if multiple runs or injections are present in the data.",
    "Flag any peaks with abnormal shape characteristics such as tailing or fronting.",
    "Compare peak intensities across the chromatogram to identify dominant compounds.",
    "Determine whether baseline correction may be required before quantitative analysis.",
    "Detect sudden signal discontinuities that could indicate injection or instrument artifacts.",
    "Summarize the chromatogram characteristics including number of peaks, retention time distribution, and signal range.",
    "Evaluate whether the chromatogram contains sufficient signal structure to support feature extraction for modeling.",
    "Identify sections of the chromatogram that contain little information (flat baseline) versus regions with high analytical activity."
    ]

    run_ID = 1

    for task in hplc_analysis_tasks:
        User_Requirements = task # Multi-Run
        try:
            Agent=BioAgent(Deep_Research_Mode=False, Max_Plan_Steps=3, Implementation_Granularity=1, User_Requirements=User_Requirements, Max_Review_Loops=10, Max_Retries=20, Max_Plan_Reviews=10, Plan_Short_Term_Memory=True, Code_Short_Term_Memory=False, Max_Word_Count = 800, Max_Images = 10, Max_Markdown_Files = 10, Max_Markdown_Lines = 250, Run_ID = run_ID)
            Agent.cleanup() # Pre-Run Artifact Removal.
            Agent.run()
        except Exception as e:
            print("\n\n==================")
            print("RUN FAILED")
            print(f"Task: {task}")
            print(f"Error: {type(e).__name__}: {e}")
            print("==================\n")
        finally:
            LLM_Manager(LLM_Type="Reasoning").stop_server() # Stops server to clear KV cache.
            Agent.cleanup() # Post-Run Artifact Removal.
        
        run_ID += 1
            
if __name__ == "__main__":
    main()