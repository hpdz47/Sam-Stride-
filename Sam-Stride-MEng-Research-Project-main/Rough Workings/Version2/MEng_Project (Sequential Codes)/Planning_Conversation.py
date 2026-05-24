# ---------------------Imports -----------------------------
from autogen.agentchat.group import AgentTarget, RevertToUserTarget, OnCondition, StringLLMCondition
from autogen.agentchat.group import OnContextCondition, ExpressionContextCondition, ContextExpression
from autogen.agentchat.group.guardrails import Guardrail, RegexGuardrail, GuardrailResult
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function, AssistantAgent
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat.group import ContextVariables, ReplyResult
from autogen.agentchat.group import StringContextCondition
from autogen.agentchat import initiate_group_chat
from autogen import GroupChat, GroupChatManager
from autogen.agentchat.group.targets.transition_target import StayTarget, TerminateTarget
from typing import Any, Dict, List, Optional, Annotated
from autogen import UpdateSystemMessage
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
import numpy as np
import pandas as pd
import json
import csv
import os
from autogen.tools.experimental import DeepResearchTool
from concurrent.futures import ThreadPoolExecutor

from vLLM_Configuration import VLLM_Config
from vLLM_Manager import LLM_Manager
from dotenv import load_dotenv
import subprocess
from pathlib import Path
import logging
import socket
import shutil
import time
from Data_Discovery_Conversation import Data_Discovery_Chat
from RAG_Tools import RAG_Tool
from Planner_Reviewer import Planner_Reviewer_Chat
from autogen import Agent
import copy
import pprint
import re
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from Focus_Area import Focus_Chat
#===========================================================
load_dotenv()

class Agent_Base():
    def __init__(self,name: str,llm_config: LLMConfig, system_message: str, Update_System_Message: Optional[str] = None, context_variables: Optional[ContextVariables] = None):
        self.name=name
        self.llm_config=llm_config # Composition Not Used here in case strucutred responses are needed for specific agents.
        self.system_message=system_message
        self.human_input_mode="NEVER" # Hard Coded as this is an AUTONMOUS system.
        if Update_System_Message:
            Updated_Message = [UpdateSystemMessage(Update_System_Message)]
        self._agent=ConversableAgent(
            name=self.name,
            llm_config=self.llm_config,
            system_message=self.system_message,
            human_input_mode=self.human_input_mode,
            update_agent_state_before_reply=Updated_Message if Update_System_Message else None,
            context_variables=context_variables

        )

    @property
    def agent(self) -> ConversableAgent:
        return self._agent # Getter for retrieving the agent instance.

# Setup Specific Agents by Inheritance of main Conversable Agent Setup.
class Profiler_Agent(Agent_Base):
    pass

class Deep_Research_Agent(Agent_Base):
    pass

class Profiling_Chat: # Defining the Data Profiling and Research Chatroom with set agents and system messages.
    Deep_Research_LLM=None
    GPU_Mode_Var=None
    def __init__(self,context_variables: ContextVariables, Deep_Research_Enabled: bool, Single_GPU_Mode: bool, Multi_GPU_Mode: Optional[bool]=None):
        # Start the vLLM servers FIRST!!!!
        if Single_GPU_Mode and Multi_GPU_Mode:
            self.GPU_Mode="Single" # Single GPU mode is the default mode.
        elif Single_GPU_Mode:
            self.GPU_Mode="Single"
        elif Multi_GPU_Mode:
            self.GPU_Mode="Multi"
        else:
            raise ValueError(f"Invalid GPU Mode: {Single_GPU_Mode} or {Multi_GPU_Mode}")
        Profiling_Chat.GPU_Mode_Var=self.GPU_Mode

        if self.GPU_Mode == "Single":
            LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        elif self.GPU_Mode == "Multi":
            LLM_Manager(LLM_Type="Reasoning").start_server()

        self.context_variables = context_variables
        self.Deep_Research_Enabled=Deep_Research_Enabled
        self.Profiler_System_Message="""
        You are a Metadata Profiler Agent. You must profile the data using the Data_Discovery function.
        You must never attempt to analyse or interpret the data yourself.
        """
        self.Deep_Research_System_Message="""
        You are a Deep Research Agent that specialises in developing a detailed report on a given topic to allow
        other Agents to learn more about a topic that they are not familiar with.
        """
        self.Deep_Research_UpdateSystemMessage="""
        **ROLE**
        Your role is to generate research areas that will be used to perform a web search for relevant information.
        You have been provided with metadata about datsaets that are being analysed. Your research areas MUST be relevant to both the
        data that is being analysed as well as the context of the data analysis.

        **Theme**
        High Performance Liquid Chromatography (HPLC) and Mass Spectrometry (MS) data s used inside a Biopharmaceutical
        manufacturing process. You have been provded with metadata about the datasets that are being analysed. The aim is to understand
        what analysis methods are suitable, specific Python approaches and dedicated Python packages that may be
        useful and any information that can be used to assess the quality of the data.
        
        **Task**
        Your task is to call the Deep_Research_Mode function with your detailed research topics. To ensure a data driven
        research approach, it is advised to include relevant information from the metadata in your research topics.
        
        **Illustrative Questions for you to consider when developing your research topics**

        1. **Data Quality and Validation for HPLC and MS data**: Research methods to assess whether 
           experimental data is suitable for analysis. What issues commonly affect chromatographic and 
           spectral data quality? How can these be detected and addressed programmatically?

        2. **HPLC Data Analysis**: Research established methods for analyzing chromatographic data, 
           including how compounds are identified and quantified from chromatograms. What Python tools 
           and resources are available? What databses are available for compound identification?

        3. **Mass Spectrometry Data Analysis**: Research established methods for analyzing mass spectra,
           including how compounds are identified from m/z data. What Python tools, databases, and 
           resources are available for compound identification?

        **Context**
        The metadata about the datasets is provided by {metadata}
        """
        Profiler_Name="Profiler_Agent" # Note: Names here will be used in self. ----, so it doesn't matter that they don't exist after constructor run.
        Deep_Research_Name="Deep_Research_Agent"
        Profiler_llm_config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Deep_Research_llm_config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.6,enable_thinking=False,LLM_Type="Reasoning").build_config()
        Profiling_Chat.Deep_Research_LLM=self.Deep_Research_llm_config
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Profiler=Profiler_Agent(Profiler_Name,
                                     Profiler_llm_config,
                                     self.Profiler_System_Message)
        self.Deep_Research=Deep_Research_Agent(Deep_Research_Name,
                                   self.Deep_Research_llm_config,
                                   self.Deep_Research_System_Message,
                                   self.Deep_Research_UpdateSystemMessage,
                                   )
        register_function(
            Data_Discovery,
            caller=self.Profiler.agent,
            executor=self.Profiler.agent,
            name="Data_Discovery",
            description="Discovers the data in the directory and provides a summary of the data."
        )

        register_function(
            Deep_Research_Mode,
            caller=self.Deep_Research.agent,
            executor=self.Deep_Research.agent,
            name="Deep_Research_Mode",
            description="Start the deep research mode."
        )
    

    def run_Conversation(self):
        # Transforming Message History (To Limit)----
        context_handling = transform_messages.TransformMessages(
            transforms=[transforms.MessageHistoryLimiter(max_messages=3)])
        context_handling.add_to_agent(self.Profiler.agent)
        if self.Deep_Research_Enabled:
            context_handling.add_to_agent(self.Deep_Research.agent)
        #---------------------------
        if self.Deep_Research_Enabled:
            agents=[self.Profiler.agent,
                    self.Deep_Research.agent]
        else:
            agents=[self.Profiler.agent]
        pattern=DefaultPattern(
            initial_agent=self.Profiler.agent,
            agents=agents,
            group_manager_args={"llm_config": self.Chat_Config},
            context_variables=self.context_variables,

        )
        
        if self.Deep_Research_Enabled:

            self.Profiler.agent.handoffs.add_context_condition(
                OnContextCondition(
                    target=AgentTarget(self.Deep_Research.agent),
                    condition=ExpressionContextCondition(
                        expression=ContextExpression("${Profiling_Complete} == True")
                    )
                )
            )
            self.Profiler.agent.handoffs.set_after_work(AgentTarget(self.Profiler.agent))

            self.Deep_Research.agent.handoffs.add_context_condition(
                OnContextCondition(
                    target=TerminateTarget(),
                    condition=ExpressionContextCondition(
                        expression=ContextExpression("${Deep_Research_Report_Count} == 2")
                    )
                )
            )

            self.Deep_Research.agent.handoffs.set_after_work(AgentTarget(self.Deep_Research.agent))
        else:
            self.Profiler.agent.handoffs.add_context_condition(
                OnContextCondition(
                    target=TerminateTarget(),
                    condition=ExpressionContextCondition(
                        expression=ContextExpression("${Profiling_Complete} == True")
                    )
                )
            )
            self.Profiler.agent.handoffs.set_after_work(AgentTarget(self.Profiler.agent))
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Profile the data.",
            max_rounds=50,
        )
        
        return result, ctx


class Planning_Chat: # Chatroom that brings the Profiling and PLanning Chats into 1 chatroom.
    def __init__(self, context_variables: ContextVariables, Deep_Research_Enabled: bool, Single_GPU_Mode: bool, Multi_GPU_Mode: Optional[bool]=None):
        self.context_variables = context_variables
        self.Deep_Research_Enabled = Deep_Research_Enabled
        self.Single_GPU_Mode = Single_GPU_Mode
        self.Multi_GPU_Mode = Multi_GPU_Mode

    def run_Conversation(self):
        #-------------- Data Discovery Chatroom ------------------
        Profiling=Profiling_Chat(context_variables=self.context_variables, Deep_Research_Enabled=self.Deep_Research_Enabled, Single_GPU_Mode=self.Single_GPU_Mode, Multi_GPU_Mode=self.Multi_GPU_Mode)
        Profiling.run_Conversation()
        #---------------------------------------------------------
        #----------- Focus Area Agent ---------------------
        Focus_Area_Chat1=Focus_Chat(context_variables=self.context_variables, Analysis_Type="Usability", Max_Rounds=5)
        Focus_Area_Chat1.run_Conversation()
        Focus_Area_Chat2=Focus_Chat(context_variables=self.context_variables, Analysis_Type="HPLC", Max_Rounds=5)
        Focus_Area_Chat2.run_Conversation()
        Focus_Area_Chat3=Focus_Chat(context_variables=self.context_variables, Analysis_Type="MS", Max_Rounds=5)
        Focus_Area_Chat3.run_Conversation()
        #------------------------------------------------------------
        #-------------- Planning and Reviewer/ Review_Panel Agents ----------------
        Planning1=Planner_Reviewer_Chat(context_variables=self.context_variables,Section_To_Run="Usability", Max_Rounds=30)
        Planning1.run_Conversation()
        Planning2=Planner_Reviewer_Chat(context_variables=self.context_variables,Section_To_Run="HPLC", Max_Rounds=30)
        Planning2.run_Conversation()
        Planning3=Planner_Reviewer_Chat(context_variables=self.context_variables,Section_To_Run="MS", Max_Rounds=30)
        Planning3.run_Conversation()
        #---------------------------------------------------------------------------


#==================== Functions to Pass to Agents =======================

def Data_Discovery(context_variables: ContextVariables) -> ReplyResult:
    """
    This function calls a conversation between agents that can write code to discover what data is available
    inside the Inputs directory. The findings will be summarised in a report and stored in the context variables.
    The functon will also define and then call the Profile_Check function to obtain the metadata (Headings and Indices), as
    before.
    """

    def Profile_Check(context_variables: ContextVariables) -> ReplyResult:
        inputs_dir=Path("./Inputs") # Defining the path:
        def get_headers(csv_file):
            with csv_file.open("r", newline="", encoding="utf-8", errors="replace") as f:
                headers = next(csv.reader(f), None)
            return csv_file.name, headers
        csv_files=[]
        for file in inputs_dir.iterdir():
            if file.suffix.lower() == ".csv":
                csv_files.append(file)
        with ThreadPoolExecutor(max_workers=8) as executor:
            for name, headers in executor.map(get_headers, csv_files):   
                context_variables["metadata"][name]=headers
        print(context_variables["metadata"])
        return ReplyResult(message="Profile_Check Successful",
                        context_variables=context_variables)
    Profile_Check(context_variables)
    data_discovery_room=Data_Discovery_Chat(context_variables=context_variables)
    data_discovery_room.run_Conversation()
    LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
    context_variables["Profiling_Complete"]=True
    return ReplyResult(
        message="Data Discovery and Profiling Successful",
        context_variables=context_variables,
    )

def Deep_Research_Mode(task: Annotated[str, """The detailed research specification to allow for deep research.
    This should include key information fromt the metadata that may be relevant. This should also include specific sites
    such as Arxiv, NIST, Google Scholar, ChemRxiv, IUPAC, PubChem, Wkipedia, etc that may be relevant resources to search."""],
    context_variables: ContextVariables) -> ReplyResult:

    # Setting Up MultiModal Model required for Deep Research:
    LLM_Manager(LLM_Type="VL").Manage_VLLM()


    # ---- Setting up the Singularity Container with Playwright installed. (Based on Docker Image).
    Playwright_Image_Docker="docker://mcr.microsoft.com/playwright/python:v1.57.0-noble" # This image has all of the OS level 
    # dependencies and playwright installations already installed as well as Python. This is useful because the OS level
    # dependencies would require root access, which is not possible on most HPC's. 
    Playwright_Image_Singularity="Playwright_Image.sif"
    # Include in dependencies with BioAgent:
    research_scripts_dir=Path("./Deep_Research_Scripts") # Always exists as it has a .env file and the Deep Research scripts needed.

    # Not required prior to running BioAgent.
    setup_dir=Path("./Singularity_Images") # Path for the Playwright SIF.
    research_dir=Path("./Deep_Research_Reports") # Path for the Deep Research Reports.
    pip_dir=Path("./Pip_Packages_DR") # Path for the Pip Packages. (Only used for when the sessiondir max size is too small).
    temp_dir=Path("./Deep_Research_Temp") # Path for the temporary files. (Delete after use).
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)
    if not research_dir.exists():
        research_dir.mkdir(parents=True, exist_ok=True)
    if not setup_dir.exists():
        setup_dir.mkdir(parents=True, exist_ok=True)
    # pip_dir will be made later in script.

    playwright_singularity_path=setup_dir/f"{Playwright_Image_Singularity}"

    if not playwright_singularity_path.exists():
        logging.info(f"Singularity image not found.  Pulling {Playwright_Image_Docker}...")
        result = subprocess.run(
            ["singularity", "pull", str(playwright_singularity_path), Playwright_Image_Docker],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise ValueError(f"Failed to pull image {Playwright_Image_Docker}: {result.stderr}")

    container_name="Playwright_Container"
    # This container differs from the contianer used in the Singularity COmmand Line Executor script (persistent container),
    # This container is set to be ephemeral to ensure all files are deleted after the research is complete, or if it errors.
    # This will prevent the need for cleanup functions and destroy any possibly malicious files.

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    #print(f"Hostname: {hostname} | IP Address: {ip_address}")

    # If Pip_Packages not setup yet:
    if not pip_dir.exists():
        logging.info(f"Pip packages not found. Installing...")
        pip_dir.mkdir(parents=True, exist_ok=True)
        setup_cmd = [
            "singularity", "exec",
            "--containall",
            "--no-home",
            "--bind", f"{pip_dir.resolve()}:/pip",  # Read-write for installation. Made read only later in script.
            "--env", "TMPDIR=/pip",
            str(playwright_singularity_path),
            "bash", "-c",
            "pip install --no-cache-dir --target=/pip ag2[openai,browser-use]",
        ]
        subprocess.run(setup_cmd)
        logging.info(f"Pip packages installed successfully.")
        try:
             # Now the code can be run with all pip installations already installed.
            instance_start_cmd = [
            "singularity", "exec",
            "--nv", # Enables GPU support.
            "--containall", # Full isolation from the host system.
            "--no-home",    # Full isolation from the home directory.
            "--network=bridge", # Setes a virtual network interface for the container to prevent any network attacks to host.
            "--bind", f"{research_dir.resolve()}:/research",
            "--bind", f"{pip_dir.resolve()}:/pip:ro", # Read only for security.
            "--bind", f"{research_scripts_dir.resolve()}:/research_scripts:ro", # Read only for security.
            "--bind", f"{temp_dir.resolve()}:/temp",
            "--env","PYTHONPATH=/pip",
            #"--env","SINGULARITYENV_PYTHONPATH=/pip",
            "--env", "TMPDIR=/temp",
            #"--env", "PYTHONUSERBASE=/pip",
            str(playwright_singularity_path),
            "python", "/research_scripts/Deep_Research_Singularity_Script.py", f"'{task}'", f"'{ip_address}'",
            ]
            result = subprocess.run(
                instance_start_cmd,
                text=True)

            if result.returncode != 0:
                raise ValueError(f"Failed to start Singularity instance: {result.stderr}")
            else:
                logging.info(f"Singularity instance started successfully.")
            # Setup the Reasoning LLM server for the rest of the Chatroom.
        finally:
            # Setup the Reasoning LLM server for the rest of the Chatroom.
            LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
            # Delete temp files for security.
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logging.info(f"Deleted temp folder: {temp_dir}")
    else:   
        try:
            instance_start_cmd = [
            "singularity", "exec",
            "--nv", # Enables GPU support.
            "--containall", # Full isolation from the host system.
            "--no-home",    # Full isolation from the home directory.
            "--network=bridge", # Setes a virtual network interface for the container to prevent any network attacks to host.
            "--bind", f"{research_dir.resolve()}:/research",
            "--bind", f"{pip_dir.resolve()}:/pip:ro", # Read only for security.
            "--bind", f"{research_scripts_dir.resolve()}:/research_scripts:ro", # Read only for security.
            "--bind", f"{temp_dir.resolve()}:/temp",
            #"--env", "PYTHONDONTWRITEBYTECODE=1", # Prevents Python from writing .pyc files. (Stops any errors).
            "--env","PYTHONPATH=/pip",
            #"--env","SINGULARITYENV_PYTHONPATH=/pip",
            "--env", "TMPDIR=/temp",
            #"--env", "PYTHONUSERBASE=/pip",
            str(playwright_singularity_path),
            "python", "/research_scripts/Deep_Research_Singularity_Script.py", f"'{task}'", f"'{ip_address}'",
            ]
            result = subprocess.run(
                instance_start_cmd,
                text=True)

            if result.returncode != 0:
                raise ValueError(f"Failed to start Singularity instance: {result.stderr}")
            else:
                logging.info(f"Singularity instance started successfully.")
        finally:
            # Setup the Reasoning LLM server for the rest of the Chatroom.
            LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
            # Delete files for security.
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logging.info(f"Deleted temp folder: {temp_dir}")

    
    context_variables["Deep_Research_Report_Count"]+=1
    return ReplyResult(
        message=f"Deep Research report saved successfully.",
        context_variables=context_variables,
    )
#=========================================================================
# main() is for unit testing. This enusres modularity of chatrooms and allows for easy testing.
# It also gives an example of how to use the Planning_Chat class.
def main():
    context_variables=ContextVariables({
        #"metadata": {},
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
        "Mass_Spectrometry_Steps_To_Add": "",
        "Issues": False, # Make sure this is reset after chat has finished to avoid conflicts later.
        # New Context Variables for pre-planning stage.
        "Univariate_Analysis": "",
        "Multivariate_Analysis": "",
        #"Univariate_EDA_Report": "",
        "Multivariate_EDA_Report": "",
        "Domain_Knowledge":[],
        "Univariate_RAG_Used": False,
        "Multivariate_RAG_Used": False,
        "Univariate_Report_Available": False,
        "Multivariate_Report_Available": False,
        "Univariate_Suggestions": "",
        "Multivariate_Suggestions": "",
        "Suggestions_Available": False,
        #-------- Testing New Addtions ----------------
        "metadata": "'metadata chromatography_combined.csv' \n Variables: ['Unnamed: 0', 'start_well', 'end_well', 'Sample_Code', 'process_part', 'run', 'chromatography_stage', 'COLUMN_IN_USE', 'Resin_type', 'column', 'Column_Volume_(ml)', 'Load_volume_(mL)', 'Load_pH', 'Unicorn_server_result_file', 'Exported_run_data_file', 'Unicorn_Method_Name', 'Chromatography_Plate_ID', 'Plate_start', 'Plate_end', 'Chromatography_Plate_ID_2', 'Plate2_start', 'Plate2_end', 'run_no', 'Run_Name', 'Excel_Filename', 'Shortened_Code_for_Bioaccord_Samples', 'Run_variables_file', 'Operator_name_(Full)', 'Date', 'Upstream_sample', 'Aliquot_code', 'Time_samples_removed_from_fridge', 'Time_sample_returned_to_fridge', 'IDBS_experiment_number', 'Sample_start_volume_(mL)', 'Sample_volume_after_dilution_with_HPW_(mL)', 'Sample_final_volume_after_pH_titration_(mL)', 'Titration_agent', 'Start_pH', 'Final_pH', 'Total_protein_concentration_(bradford)_(mg/mL)', 'Conducivity_(mS)', 'Concentration_(mg/mL)', 'Filter_type_used', 'Load_sample_start_ID', 'Load_sample_start_IDBS_name', 'Load_sample_end_IDBS_name', 'EQ_and_wash_buffer', 'EQ_and_Wash_Buffer_Inlet', 'Elution_buffer', 'Elution_Buffer_Inlet', 'Sanitisation_buffer', 'Other_buffer', 'All_columns_loaded_from_same_material', 'Notes', 'fraction_volume', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise'] \n -----'MS_combined.csv' ------ \n ['Unnamed: 0', 'Unique_Peak_ID', 'Unique_MS_Sample_ID', 'Sample_Code', 'run_code', 'process_part', 'run', 'MS_method', 'column', 'chromatography_stage', 'start_well', 'end_well', 'Replicate', 'Type', 'Molecule ID', 'Component', 'Observed_TIC_RT_(mins)', 'Observed_UV_RT_(mins)', 'Observed RT delta (mins)', 'Response', '%_of_response', 'Observed_neutral_mass_(Da)', 'Observed_m/z', 'Spectrum_type', 'Expected_mass_(Da)', 'Mass_error_(ppm)', 'Alternative_assignments', 'fraction_volume', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise']\n ------'chromatography_combined.csv'------ \n ['Unnamed: 0', 'run_no', 'run', 'Fraction_unique_ID', 'column', 'volume_ml', 'UV_1_280_ml', 'UV_1_280_mAU', 'Cond_ml', 'Cond_mS_cm', 'Conc_B_ml', 'Conc_B_%', 'Injection_ml', 'Injection_Injection', 'Run_Log_ml', 'Run_Log_Logbook', 'Fraction_ml', 'Fraction_Fraction', 'UV_1_280_CUT_TEMP_100_BASEM_ml', 'UV_1_280_CUT_TEMP_100_BASEM_mAU', 'UV_2_260_ml', 'UV_2_260_mAU', 'pH_ml', 'pH_pH', 'DeltaC_pressure_ml', 'DeltaC_pressure_MPa', 'System_flow_ml', 'System_flow_ml_min', 'Sample_flow_ml', 'Sample_flow_ml_min', 'Sample_Code', 'chromatography_stage', 'chromatography_stage_order', 'Fraction_number', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise', 'fraction_volume']",
        "Univariate_EDA_Report": "### Dataset Interpretation Report\n\nThis report provides a detailed interpretation of the three files in the dataset: `metadata chromatography_combined.csv`, `MS_combined.csv`, and `chromatography_combined.csv`. Each file is analyzed independently to understand its structure, content, and relevance for downstream analysis.\n\n---\n\n### 1. `metadata chromatography_combined.csv`\n\n**File Structure and Type:**\n- This is a CSV file with a data shape of 354 rows and 58 columns.\n- The file contains metadata related to chromatography runs, including experimental parameters, sample details, buffer information, and operational notes.\n\n**Key Observations:**\n- **Data Types:** The file contains a mix of `int64`, `float64`, and `object` (string) data types. Notably, several columns (e.g., `start_well`, `end_well`, `Sample_Code`) are of type `object`, indicating categorical or textual data.\n- **Missing Values:** Many columns (e.g., `Time_samples_removed_from_fridge`, `Time_sample_returned_to_fridge`, `Notes`) have no observed values (all NaNs), suggesting they may be unused or incomplete.\n- **Dense Data:** The dataset is dense, with most numerical columns having meaningful values. The `Unnamed: 0` column appears to be an index, with values ranging from 0 to 49.\n- **Categorical Variables:** Several columns are highly categorical:\n  - `Resin_type` (4 unique values)\n  - `chromatography_stage` (16 unique values)\n  - `Run_Name`, `Excel_Filename`, `Operator_name_(Full)` (11, 11, and 1 unique values respectively), indicating a small number of experimental runs.\n- **Numerical Variables:** Key experimental parameters include:\n  - `Load_volume_(mL)` (mean ~112 mL, range 35–188 mL)\n  - `Sample_start_volume_(mL)` (mean ~166 mL, range 90–270 mL)\n  - `Total_protein_concentration_(bradford)_(mg/mL)` (mean ~0.16 mg/mL, range 0.10–0.44 mg/mL)\n  - `Start_pH` and `Final_pH` (mean ~7.1 and ~6.2, respectively)\n- **Grouping:** The data is grouped by `run`, `run_no`, and `Sample_Code`, suggesting that each row corresponds to a unique experimental run or sample.\n- **Relevance:** This file serves as a comprehensive metadata source for chromatography experiments, linking sample, process, and operational details. It is essential for contextualizing the other two datasets.\n\n**Conclusion:** This file is a dense, structured metadata file critical for understanding the experimental setup. It should be used to enrich and validate the other datasets.\n\n---\n\n### 2. `MS_combined.csv`\n\n**File Structure and Type:**\n- This is a large CSV file (620.35 MB) with 1,912,029 rows and 30 columns.\n- It contains mass spectrometry (MS) data, likely from peptide or protein identification and quantification.\n\n**Key Observations:**\n- **Data Types:** Mixed types, with `int64`, `float64`, and `object` columns. The `Unnamed: 0` column appears to be an index.\n- **Missing Values:** Several columns (e.g., `Type`, `Molecule ID`, `Component`, `Expected_mass_(Da)`, `Mass_error_(ppm)`, `Alternative_assignments`) have no observed values (all NaNs), indicating they may be unused or incomplete.\n- **Sparse Data:** Despite its large size, the dataset is sparse in certain key columns, particularly those related to molecular identification.\n- **Categorical Variables:**\n  - `Unique_MS_Sample_ID` (283 unique values)\n  - `Sample_Code` (274 unique values)\n  - `chromatography_stage` (19 unique values)\n  - `Spectrum_type` (3 unique values)\n- **Numerical Variables:** Key MS features include:\n  - `Observed_TIC_RT_(mins)` (mean ~2.33 mins, range ~0.62–3.03 mins)\n  - `Observed_UV_RT_(mins)` (mean ~1.98 mins, range ~0.74–2.72 mins)\n  - `Response` (mean ~9,920, range ~4.86–15.25M)\n  - `Observed_neutral_mass_(Da)` (mean ~162.5 kDa, range ~400–314 kDa)\n  - `Observed_m/z` (mean ~733 Da, range ~505–1,263 Da)\n- **Grouping:** Data is grouped by `Unique_MS_Sample_ID`, `Sample_Code`, and `run_code`, suggesting a hierarchical structure where multiple MS spectra are associated with a single sample.\n- **Relevance:** This file contains high-resolution MS data, likely from LC-MS/MS experiments. It is essential for downstream analysis such as peptide identification, quantification, and post-translational modification (PTM) analysis.\n\n**Conclusion:** This file is a large, sparse dataset containing detailed MS data. It is highly relevant for proteomics analysis but requires careful handling due to missing values in key identification columns.\n\n---\n\n### 3. `chromatography_combined.csv`\n\n**File Structure and Type:**\n- This is a large CSV file (257.96 MB) with 1,084,069 rows and 37 columns.\n- It contains chromatographic data, including UV, conductivity, pH, flow, and volume measurements across fractions.\n\n**Key Observations:**\n- **Data Types:** Mixed types, with `int64`, `float64`, and `object` columns. The `Unnamed: 0` column is likely an index.\n- **Missing Values:** Some columns (e.g., `Injection_Injection`, `Fraction_Fraction`) have no observed values (all NaNs), suggesting they may be unused.\n- **Dense Data:** The dataset is dense, with most numerical columns having meaningful values. The `Fraction_number` and `Sample_Code` columns suggest a hierarchical structure where multiple fractions are generated per sample.\n- **Categorical Variables:**\n  - `column` (4 unique values)\n  - `chromatography_stage` (16 unique values)\n  - `Run_Log_Logbook` (17 unique values)\n- **Numerical Variables:** Key chromatographic parameters include:\n  - `volume_ml` (mean ~126 mL, range ~-33.8–335.8 mL)\n  - `UV_1_280_ml` and `UV_1_280_mAU` (mean ~126 mL and ~250 mAU, respectively)\n  - `Cond_ml` and `Cond_mS_cm` (mean ~126 mL and ~15.8 mS/cm)\n  - `pH_ml` and `pH_pH` (mean ~132.5 mL and ~6.28 pH)\n  - `fraction_volume` (mean ~20.9 mL, range ~0.008–66.0 mL)\n- **Grouping:** Data is grouped by `Sample_Code`, `run_no`, and `Fraction_number`, indicating that each row corresponds to a fraction collected during a chromatography run.\n- **Relevance:** This file contains high-resolution chromatographic profiles, essential for understanding elution patterns, peak detection, and fraction collection. It is critical for integrating with MS data to link protein identity to chromatographic behavior.\n\n**Conclusion:** This file is a dense, structured chromatographic dataset that provides detailed fraction-level data. It is indispensable for correlating MS results with chromatographic elution profiles.\n\n---\n\n### Overall Summary\n\n- The dataset consists of three interrelated files:\n  1. `metadata chromatography_combined.csv`: Metadata for experimental runs.\n  2. `chromatography_combined.csv`: High-resolution chromatographic profiles (fraction-level).\n  3. `MS_combined.csv`: High-resolution MS data (peptide/protein-level).\n\n- **Integration Potential:** The three files can be integrated using common keys such as `Sample_Code`, `run_no`, and `chromatography_stage`. This integration enables a comprehensive analysis of protein behavior across chromatography and MS platforms.\n\n- **Gaps and Limitations:**\n  - Several columns in `MS_combined.csv` are entirely missing (all NaNs), which may limit downstream identification.\n  - Some columns in all files (e.g., `Notes`, `Other_buffer`) are unused or incomplete.\n  - Further validation is required to confirm the consistency of `Sample_Code` and `run_no` across files.\n\n- **Recommendations for Further Analysis:**\n  - Perform data merging using `Sample_Code` and `run_no` to align chromatography and MS data.\n  - Investigate the reasons for missing values in `MS_combined.csv`.\n  - Validate the integrity of `fraction_volume` and related columns across files.\n  - Conduct a time-series analysis of chromatographic profiles to identify elution patterns.\n\nThis dataset is well-structured for integrative proteomics analysis, provided that missing data issues are addressed.",
        "Data_Reality_Report":"",
        "Reality_Feedback": False,
        "FA_Feedback": False,
        "FA_Report":"",
        "Focus_Area_Usability":"",
        "Focus_Area_HPLC":"",
        "Focus_Area_MS":"",
        "FA_Updated": False,
        "Usability_Review_Request": False,
        "HPLC_Review_Request": False,
        "MS_Review_Request": False,
        "last_speaker": "",
        #-------------------------------
        "Data_Structure_Advice":"",
        "Structure_Advice_Provided": False,
        "RAG_QA":"",
        "QA_Available": False,
        "RAG_Interpretation":"",
        "RAG_Interpretation_Available": False,
        "Compiled_Review":"",
        "Compilation_Complete": False,
        #----------------------------------
        "Plan_Scoring_1":[],
        "Plan_Scoring_2":[],
        "Plan_Scoring_3":[],
        "Plan_Scoring_4":[],
        "Plan_Scoring_5":[],
        "Plan_Scoring_6":[],
        "Plan_Scoring_7":[],
        "Plan_Scoring_8":[],
        "Plan_Scoring_9":[],
        "Plan_Scoring_10":[],
        "Plan_Scoring_11":[],
        "Plan_Scoring_12":[],
        })
    planning_room=Planning_Chat(context_variables=context_variables, Deep_Research_Enabled=True, Single_GPU_Mode=True, Multi_GPU_Mode=False)
    planning_room.run_Conversation()

# Store the context variables as a JSON file for troubleshooting.
    with open("Planning_Conversation.json", "w") as f:
        json.dump(context_variables.model_dump(), f, indent=2)



if __name__ =="__main__":
    main()