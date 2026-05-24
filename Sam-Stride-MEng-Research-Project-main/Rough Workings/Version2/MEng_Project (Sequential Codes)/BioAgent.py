from Planning_Conversation import Planning_Chat
from Data_Analysis_Conversation import Data_Analysis_Chat
from autogen.agentchat.group import ContextVariables
from typing import Optional
from Supervisor_Reviewer import Supervisor_Reviewer_Chat
import shutil
from pathlib import Path
import json
import numpy as np

class BioAgent:
    def __init__(self, Deep_Research_Enabled: bool, Single_GPU_Mode: bool, Multi_GPU_Mode: Optional[bool]=None):
        self.Deep_Research_Enabled = Deep_Research_Enabled
        self.Single_GPU_Mode = Single_GPU_Mode
        self.Multi_GPU_Mode = Multi_GPU_Mode if Multi_GPU_Mode is not None else False

        self.context=ContextVariables({
        "metadata": {},
        "Plan": [], # Still needed because of Data Analysis Chatroom.
        "Reviewer_Suggestions": "", # Replaced
        "First_Plan": True, # Replaced
        "Revision_Count": 0, # Replaced
        "Plan_Updated": False, # Replaced
        "Feedback": False, # Replaced
        "Planner_Function_Called": False, # Replaced
        "Reviewer_Function_Called": False, # Replaced
        "Deep_Research_Report_Count": 0,
        "Data_Discovery_Code": "",
        "Data_Discovery_Code_Updated": False,
        "Data_Discovery_Code_Reviews": "",
        "Data_Discovery_Code_Suggestions_Available": False,
        "Data_Discovery_Code_Approval": False,
        "Data_Discovery_Code_Revision_Count": 0,
        "Data_Discovery_Summary": "",
        "Iteration_Count": 0,
        "Profiling_Complete": False,
        "Usability_RAG_Queries": 0,
        "Usability_RAG_Results": [],
        "HPLC_RAG_Queries": 0,
        "HPLC_RAG_Results": [],
        "MS_RAG_Queries": 0,
        "MS_RAG_Results": [],
        "Usability_Plan_Approval": False,
        "HPLC_Analysis_Plan_Approval": False,
        "Mass_Spectrometry_Plan_Approval": False,
        "Usability_Plan":"",
        "HPLC_Analysis_Plan":"",
        "Mass_Spectrometry_Plan":"",
        "Usability_Plan_Updated": False,
        "HPLC_Plan_Updated": False,
        "MS_Plan_Updated": False,
        "Usability_Reviewer_Suggestions": "",
        "Usability_Feedback": False,
        "HPLC_Reviewer_Suggestions": "",
        "HPLC_Feedback": False,
        "MS_Reviewer_Suggestions": "",
        "MS_Feedback": False,
        "Usability_RAG_Used": False,
        "HPLC_RAG_Used": False,
        "MS_RAG_Used": False,
        "Usability_Plan_Review": "",
        "Usability_Steps_To_Add": "",
        "HPLC_Analysis_Plan_Review": "",
        "HPLC_Analysis_Steps_To_Add": "",
        "Mass_Spectrometry_Plan_Review": "",
        "Mass_Spectrometry_Plan_Steps_To_Add": "",
        "RAG_Queries": 0,
        "RAG_Results": [],
        "RAG_Used": False,
        "RAG_Number": 0,
        "RAG_Approval": False,
        "Score_Approval": False,
        "Reviewer_RAG_Used": False,
        "Score_Available": False,
        "Final_Reviews": "",
        "Plan_Score": 0,
        "Final_Reviews_Submitted": False,
        "Combined_Summary": "",
        "HPLC_Analysis_Review": [],
        "Usability_Plan_Review": [],
        "Mass_Spectrometry_Plan_Review": [],
        "HPLC_Analysis_Steps_To_Add": [],
        "Usability_Plan_Steps_To_Add": [],
        "Mass_Spectrometry_Steps_To_Add": [],
        "Issues": False,
        "Usability_Review_Request": False,
        "HPLC_Review_Request": False,
        "MS_Review_Request": False,
        "RAG_Skipped": 0,
        "last_speaker": "",
        "Reviewer_Skipped": 0,
        "Usability_Plan_Score": 0,
        "HPLC_Plan_Score": 0,
        "MS_Plan_Score": 0,
        "Code_Reviews_Available": False,
        "Code_Updated": False,
        "Code_Reviews": "",
        "Issues": False,
        "Usability_Shell_Codes": "",
        "HPLC_Shell_Codes": "",
        "MS_Shell_Codes": "",
        "Usability_Python_Codes": "",
        "HPLC_Python_Codes": "",
        "MS_Python_Codes": "",
        "Usability_Shell_Code":"",
        "Usability_Python_Code":"",
        "HPLC_Shell_Code":"",
        "HPLC_Python_Code":"",
        "MS_Shell_Code":"",
        "MS_Python_Code":"",
        "Approval": False,
        "Univariate_Analysis": "",
        "Multivariate_Analysis": "",
        "Univariate_EDA_Report": "",
        "Multivariate_EDA_Report": "",
        "Domain_Knowledge":[],
        "Univariate_RAG_Used": False,
        "Multivariate_RAG_Used": False,
        "Univariate_Report_Available": False,
        "Multivariate_Report_Available": False,
        "Univariate_Suggestions": "",
        "Multivariate_Suggestions": "",
        "Focus_Area_Usability": "",
        "Focus_Area_HPLC": "",
        "Focus_Area_MS": "",
        "Suggestions_Available": False,
        "Data_Structure_Advice":"",
        "Structure_Advice_Provided": False,
        "RAG_QA":"",
        "QA_Available": False,
        "RAG_Interpretation":"",
        "RAG_Interpretation_Available": False,
        "Compiled_Review":"",
        "Compilation_Complete": False,
        #---- Initialising More Context Variables for scoring than needed for now, to avoid errors during runtime -----
        "Usability_Plan_Scoring_1":[0,0,0,0],
        "Usability_Plan_Scoring_2":[0,0,0,0],
        "Usability_Plan_Scoring_3":[0,0,0,0],
        "Usability_Plan_Scoring_4":[0,0,0,0],
        "Usability_Plan_Scoring_5":[0,0,0,0],
        "Usability_Plan_Scoring_6":[0,0,0,0],
        "Usability_Plan_Scoring_7":[0,0,0,0],
        "Usability_Plan_Scoring_8":[0,0,0,0],
        "Usability_Plan_Scoring_9":[0,0,0,0],
        "Usability_Plan_Scoring_10":[0,0,0,0],
        "Usability_Plan_Scoring_11":[0,0,0,0],
        "Usability_Plan_Scoring_12":[0,0,0,0],
        "Usability_Plan_Scoring_13":[0,0,0,0],
        #--
        "HPLC_Plan_Scoring_1":[0,0,0,0],
        "HPLC_Plan_Scoring_2":[0,0,0,0],
        "HPLC_Plan_Scoring_3":[0,0,0,0],
        "HPLC_Plan_Scoring_4":[0,0,0,0],
        "HPLC_Plan_Scoring_5":[0,0,0,0],
        "HPLC_Plan_Scoring_6":[0,0,0,0],
        "HPLC_Plan_Scoring_7":[0,0,0,0],
        "HPLC_Plan_Scoring_8":[0,0,0,0],
        "HPLC_Plan_Scoring_9":[0,0,0,0],
        "HPLC_Plan_Scoring_10":[0,0,0,0],
        "HPLC_Plan_Scoring_11":[0,0,0,0],
        "HPLC_Plan_Scoring_12":[0,0,0,0],
        "HPLC_Plan_Scoring_13":[0,0,0,0],
        #--
        "MS_Plan_Scoring_1":[0,0,0,0],
        "MS_Plan_Scoring_2":[0,0,0,0],
        "MS_Plan_Scoring_3":[0,0,0,0],
        "MS_Plan_Scoring_4":[0,0,0,0],
        "MS_Plan_Scoring_5":[0,0,0,0],
        "MS_Plan_Scoring_6":[0,0,0,0],
        "MS_Plan_Scoring_7":[0,0,0,0],
        "MS_Plan_Scoring_8":[0,0,0,0],
        "MS_Plan_Scoring_9":[0,0,0,0],
        "MS_Plan_Scoring_10":[0,0,0,0],
        "MS_Plan_Scoring_11":[0,0,0,0],
        "MS_Plan_Scoring_12":[0,0,0,0],
        "MS_Plan_Scoring_13":[0,0,0,0],
    })
   
    def run_Conversation(self):
        # Need to instantiate chatrooms and run them immediately, due to how the vLLM servers are started by the Manager.
        Data_Analysis_Chatrooms=[]
        Planning=Planning_Chat(context_variables=self.context, Deep_Research_Enabled=self.Deep_Research_Enabled, Single_GPU_Mode=self.Single_GPU_Mode, Multi_GPU_Mode=self.Multi_GPU_Mode)
        Planning.run_Conversation()
        #----------- Obtain The Scores ------------------
        #=== Usability ===
        Total_Usability_Iter=0
        for key,_ in self.context:
            if key.startswith("Usability_Plan_Scoring_"):
                Total_Usability_Iter+=1
        Usability_Plan_Scoring_Matrix=np.zeros((Total_Usability_Iter,4))
        for score in range(1, Total_Usability_Iter+1):
            Usability_Plan_Scoring_Matrix[score-1,:]=self.context[f"Usability_Plan_Scoring_{score}"]
        np.savetxt("Usability_Plan_Scoring.csv", Usability_Plan_Scoring_Matrix, delimiter=",", fmt="%.6f")

        #=== HPLC ===
        Total_HPLC_Iter=0
        for key,_ in self.context:
            if key.startswith("HPLC_Plan_Scoring_"):
                Total_HPLC_Iter+=1
        HPLC_Plan_Scoring_Matrix=np.zeros((Total_HPLC_Iter,4))
        for score in range(1, Total_HPLC_Iter+1):
            HPLC_Plan_Scoring_Matrix[score-1,:]=self.context[f"HPLC_Plan_Scoring_{score}"]
        np.savetxt("HPLC_Plan_Scoring.csv", HPLC_Plan_Scoring_Matrix, delimiter=",", fmt="%.6f")

        #=== MS ===
        Total_MS_Iter=0
        for key,_ in self.context:
            if key.startswith("MS_Plan_Scoring_"):
                Total_MS_Iter+=1
        MS_Plan_Scoring_Matrix=np.zeros((Total_MS_Iter,4))
        for score in range(1, Total_MS_Iter+1):
            MS_Plan_Scoring_Matrix[score-1,:]=self.context[f"MS_Plan_Scoring_{score}"]
        np.savetxt("MS_Plan_Scoring.csv", MS_Plan_Scoring_Matrix, delimiter=",", fmt="%.6f")


        #------------------------------------------------
        Data_Analysis_Chatroom=Data_Analysis_Chat(context_variables=self.context,Step_Size=1)
        Data_Analysis_Chatroom.run_Conversation()
        Supervisor_Reviews1=Supervisor_Reviewer_Chat(context_variables=self.context, Analysis_Type="Usability", Max_Rounds=10)
        Supervisor_Reviews1.run_Conversation()
        Supervisor_Reviews2=Supervisor_Reviewer_Chat(context_variables=self.context, Analysis_Type="HPLC", Max_Rounds=10)
        Supervisor_Reviews2.run_Conversation()
        Supervisor_Reviews3=Supervisor_Reviewer_Chat(context_variables=self.context, Analysis_Type="MS", Max_Rounds=10)
        Supervisor_Reviews3.run_Conversation()
        # Clear Data_Results before next iteration:
        # Save First Iteration Results First:
        backup_dir = Path("./Results_History/Attempt6")
        results_dir = Path("./Data_Results")
        shutil.copytree(results_dir, backup_dir, dirs_exist_ok=True)
        shutil.rmtree(results_dir)
        results_dir.mkdir()

        # Save results to file before clearing:
        with open("Iteration1_Results.json", "w") as f:
            json.dump(self.context.model_dump(), f, indent=2)

        # Iteration 2
        Data_Analysis_Chat.ID=0 # Reset so that ID doesn't increase to 4, as this is undefined.
        Data_Analysis_Chatrooms=[]
        # Reset Context Variables ---------------
        self.context["Usability_Plan_Approval"] = False
        self.context["HPLC_Analysis_Plan_Approval"] = False
        self.context["Mass_Spectrometry_Plan_Approval"] = False
        self.context["Usability_Plan_Updated"] = False
        self.context["HPLC_Plan_Updated"] = False
        self.context["MS_Plan_Updated"] = False
        self.context["Usability_Reviewer_Suggestions"] = ""
        self.context["HPLC_Reviewer_Suggestions"] = ""
        self.context["MS_Reviewer_Suggestions"] = ""
        self.context["Usability_Feedback"] = False
        self.context["HPLC_Feedback"] = False
        self.context["MS_Feedback"] = False
        self.context["Usability_RAG_Used"] = False
        self.context["HPLC_RAG_Used"] = False
        self.context["MS_RAG_Used"] = False
        self.context["Usability_RAG_Queries"] = 0
        self.context["HPLC_RAG_Queries"] = 0
        self.context["MS_RAG_Queries"] = 0
        self.context["Usability_RAG_Results"] = []
        self.context["HPLC_RAG_Results"] = []
        self.context["MS_RAG_Results"] = []
        #--------------------------------------------
        Planning=Planning_Chat(context_variables=self.context, Deep_Research_Enabled=self.Deep_Research_Enabled, Single_GPU_Mode=self.Single_GPU_Mode, Multi_GPU_Mode=self.Multi_GPU_Mode)
        Planning.run_Conversation()
        Data_Analysis_Chatroom=Data_Analysis_Chat(context_variables=self.context,Step_Size=1)
        Data_Analysis_Chatroom.run_Conversation()
        # ----- Before New Reviews can Happen, the context variables for previous images, etc MUST be cleared.
        context_var_key_prefix=["Usability_Image_Summary", "HPLC_Image_Summary", "MS_Image_Summary",
                                "Usability_Score", "HPLC_Score", "MS_Score",
                                "Reviewer_RAG_Results",
                                "Usability_image", "HPLC_image", "MS_image"]
        for key in list(self.context.model_dump().keys()):
            if any(key.startswith(prefix) for prefix in context_var_key_prefix):
                del self.context[key]

        #--------------------------------------------------------------------------------------------
        Supervisor_Reviews1=Supervisor_Reviewer_Chat(context_variables=self.context, Analysis_Type="Usability", Max_Rounds=10)
        Supervisor_Reviews1.run_Conversation()
        Supervisor_Reviews2=Supervisor_Reviewer_Chat(context_variables=self.context, Analysis_Type="HPLC", Max_Rounds=10)
        Supervisor_Reviews2.run_Conversation()
        Supervisor_Reviews3=Supervisor_Reviewer_Chat(context_variables=self.context, Analysis_Type="MS", Max_Rounds=10)
        Supervisor_Reviews3.run_Conversation()

        with open("Iteration2_Results.json", "w") as f:
            json.dump(self.context.model_dump(), f, indent=2)

def main():
    Bio_Agent_Analysis=BioAgent(Deep_Research_Enabled=False, Single_GPU_Mode=True, Multi_GPU_Mode=False)
    Bio_Agent_Analysis.run_Conversation()

if __name__ == "__main__":
    main()
        