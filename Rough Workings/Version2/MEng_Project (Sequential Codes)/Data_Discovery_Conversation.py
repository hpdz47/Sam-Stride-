# ---------------------Imports -----------------------------
from autogen.agentchat.group import AgentTarget, RevertToUserTarget, OnCondition, StringLLMCondition
from autogen.agentchat.group import OnContextCondition, ExpressionContextCondition, ContextExpression
from autogen.agentchat.group.guardrails import Guardrail, RegexGuardrail, GuardrailResult
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat.group import ContextVariables, ReplyResult
from autogen.agentchat.group import StringContextCondition
from autogen.agentchat import initiate_group_chat
from autogen import GroupChat, GroupChatManager
from autogen.agentchat.group.targets.transition_target import StayTarget, TerminateTarget
from typing import Any, Dict, List, Optional, Annotated, Tuple, Union
from autogen import UpdateSystemMessage
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
import numpy as np
import pandas as pd
import json
import csv
import os

from vLLM_Configuration import VLLM_Config
from vLLM_Manager import LLM_Manager
from dotenv import load_dotenv
from Singularity_Command_Line_Executor import SingularityCommandLineCodeExecutor
from pathlib import Path
from autogen import Agent
import copy
import pprint
import re
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from RAG_Tools import RAG_Tool
#===========================================================
load_dotenv()

#===================== Structured Responses ===============================
class Code_Response(BaseModel):
    Shell_Code: str = Field(..., description="The Shell code to be updated.")
    Python_Code: str = Field(..., description="The Python code to be updated.")

class Summary_Step(BaseModel):
    File_Name: str = Field(..., description = "The name of the file that was discovered.")
    File_Type: str = Field(..., description = "The type of the file that was discovered.")
    File_Summary: str = Field(..., description = "A summary of the file that was discovered.")
class Summary_Response(BaseModel):
    Summary_Steps: List[Summary_Step] = Field(..., description = "The summary for each file discovered.")

class Step(BaseModel):
    Analysis_Type: str = Field(..., description="A clear statement of the analysis type that must be performed by the coder.")
    Variables: List[str] = Field(..., description="A list of variable names from the data file that must be used for the analysis.")
    Context: str = Field(..., description="Any important information that the coder must be aware of when handling the data and important explanations of the data analysis suggested.")

class PlanResponse(BaseModel):
    Data_File: str = Field(..., description="The main data file that must be analysed.")
    Analysis_Suggestions: List[Step] = Field(..., description="A list of analyses to perform using the data file suggested.")

# Note: In this chatroom, the SngularityCommandLineExecutor will not have separate files passed to it, as everything will
# be printed out and stored in the Context Variables. This is because the main purpose is to understand the data available,
# wthout needing to do any analysis on the data.

class Agent_Base():
    def __init__(self,name: str,llm_config: LLMConfig, system_message: str, Update_System_Message: Optional[str] = None):
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
            update_agent_state_before_reply=Updated_Message if Update_System_Message else None
        )

    @property
    def agent(self) -> ConversableAgent:
        return self._agent # Getter for retrieving the agent instance.

# Setup Specific Agents by Inheritance of main Conversable Agent Setup.
class Data_Discovery_Coder_Agent(Agent_Base):
    pass
class Data_Discovery_Reviewer_Agent(Agent_Base):
    pass
class Pre_Planner_Agent(Agent_Base):
    pass
class RAG_Agents(Agent_Base):
    pass
class Suggestion_Agent(Agent_Base):
    pass

class EDA_Analysis:
    Stage:""
    Index:Optional[int]=None
    def __init__(self,context_variables: ContextVariables, Stage:str, Index: Optional[int]=None):
        print("Data Discovery in Progress ... \n")
        LLM_Manager(LLM_Type="Coding").Manage_VLLM() # Set up the vLLM server for coding. Assumes only Single GPU Mode for now.
        self.context_variables = context_variables
        self.Stage=Stage
        EDA_Analysis.Stage=Stage
        EDA_Analysis.Index=Index
        self.Index=Index
        if self.Stage == "Univariate":
            self.Data_Discovery_Coder_System_Message="""
            **ROLE**
            Your role is to write python code (using shell code where necessary) to perform non-graphical Exploratory Data Analysis (EDA) on each data file that is present. The directory is called /inputs/ and is the only directory the code is allowed to explore.
            Your code must implement a univariate EDA approach for each file. Each file must be associated with a separate summary table of the EDA performed. In this table, you must make it clear what the summary statistic is and what variable it refers to and follow the output format precisely. 
            The table must include all column names in the file along with the relevant summary statistics. The statistics
            to be found using EDA must follow the Allowed Summary Themes below. The output must never miss out any information and must never contain an ellipsis (...). The code must be capable of handling large files efficiently.
            You must never attempt to output the raw data from the files or attempt to interpret the data. You will receive reviews of your code and you must always use the suggestions
            from this to update your code so that it better aligns with the requirements.
            """
            self.Data_Discovery_Coder_Updated_System_Message="""
            **ROLE**
            Your role is to write python code (using shell code where necessary) to perform non-graphical Exploratory Data Analysis (EDA) on each data file that is present. The directory is called /inputs/ and is the only directory the code is allowed to explore.
            Your code must implement a univariate EDA approach for each file. Each file must be associated with a separate summary table of the EDA performed. In this table, you must make it clear what the summary statistic is and what variable it refers to and follow the output format precisely. 
            The table must include all column names in the file along with the relevant summary statistics. The statistics
            to be found using EDA must follow the Allowed Summary Themes below. The output must never miss out any information and must never contain an ellipsis (...). The code must be capable of handling large files efficiently.
            You must never attempt to output the raw data from the files or attempt to interpret the data. You will receive reviews of your code and you must always use the suggestions
            from this to update your code so that it better aligns with the requirements.

            **Task**
            - You MUST use the Data_Discovery_Code_Update function to update your code.
            - You will be provided with feedback that you MUST use to update your code.
            - You will be provided with multiple data files and your code must print the file name and then a very concise description of the file structure and any relevant metadata.
            - You may be provided with the results from the first execution of your code. You MUST read this and improve your code based on the requirements.
            - The directory to explore is /inputs/. You must only explore this directory.
            - Your code MUST perform univariate EDA to produce a single summary table for each file analysed. The summary table must include all column names and relevant summary statistics.

            **Allowed Summary Themes**
            - File metadata (File Size)
            - Rows / columns (Size and shape)
            - Column names 
            - Data types 
            - Cardinality & keys
            - Statistical Summaries (mean / median / std / min / max)
            - Additional Univarate Analysis is encouraged.

            **Forbidden Summary Themes** 
            - Multi-variate analysis
            - Graphical methods
            - Correlation or covariance
            - Domain interpretation

            **Output Format** (Example)

            File: sample.csv
            File Type: CSV
            File Size: X MB
            Data Shape (1000 x 20)
            Summary Table:
            | Column Name | Data Type | Mean | Median | Std | Min | Max | Cardinality| 
            |-------------|-----------|------|--------|-----|-----|-----|------------|
            | Column1     |Numeric    | ...  | ...    | ... | ... | ... | ...  |
            | Column2     |Categorical| ...  | ...    | ... | ... | ... | ... |
            | ...         | ...       | ...  | ...    | ... | ... | ... | ... |

            **Context**
            The current code is as follows: {Data_Discovery_Code}
            The Reviewer has provided the following feedback: {Data_Discovery_Code_Reviews}
            """
            self.Data_Discovery_Reviewer_System_Message="""
            **ROLE**
            Your role is to review python code that performs non-graphical Exploratory Data Analysis (EDA) on data files in the /inputs/ directory. This is the only directory that the code is allowed to explore.
            The code must perform univariate EDA to produce a single summary table for each file analysed. The summary table must include all column names and relevant summary statistics.
            The use of any code that could produce an ellipsis (...) in the output is forbidden as this can miss out important information.
            Examples of the methods to be used are given in the Allowed Summary Themes below. The code should never attempt to print out the raw data from the files.
            """
            self.Data_Discovery_Reviewer_Updated_System_Message="""
            ----------- ROLE ---------------------
            Your role is to review python code that performs non-graphical Exploratory Data Analysis (EDA) on data files in the /inputs/ directory. This is the only directory that the code is allowed to explore.
            The code must perform univariate EDA to produce a single summary table for each file analysed. The summary table must include all column names and relevant summary statistics.
            The use of any code that could produce an ellipsis (...) in the output is forbidden as this can miss out important information.
            Examples of the methods to be used are given in the Allowed Summary Themes below. The code should never attempt to print out the raw data from the files.
            ---------------------------------------
            ----------- Task ----------------------
            - You MUST use the Data_Discovery_Code_Review function to provide feedback.
            - If the code does not meet these requirements, you must provide actionable feedback to improve the code. You must only approve the code when the code meets all of the criteria. You must be strict with your enforcement of these rules.
            - If the code meets ALL of the criteria below, you MUST set Approval_Status to True. But you cannot undo this decision and so you must be certain.
            - Never approve code that does not meet all of the criteria.
            - The code must not analyse the data in any way other than to produce metadata about the file structure and other useful information.
            ---------------------------------------

            ---------- Allowed Summary Themes ------------
            - File metadata (File Size)
            - Rows / columns (Size and shape)
            - Column names 
            - Data types 
            - Cardinality & keys
            - Statistical Summaries (mean / median / std / min / max)
            - Unvariate anlysis of numerical columns
            -----------------------------------------------

            --------- Forbidden Summary Themes -----------------
            - Multi-variate analysis
            - Graphical methods
            - Correlation or covariance
            - Domain interpretation
            -----------------------------------------------------

            --------- Context ----------------------------------
            ***** Code to be Improved *******
            {Data_Discovery_Code}

            """
        elif self.Stage=="Multivariate":
            # Obtaining the correct plan section:
            plan = self.context_variables["Univariate_Suggestions"][self.Index]
            self.Plan_Step = (
                f"Data_File: {plan['Data_File']}\n"
                "Analysis_Suggestions:\n"
                + "\n".join(
                    f"- Analysis_Type: {step['Analysis_Type']}\n"
                    f"  Variables: {', '.join(step['Variables'])}\n"
                    f"  Context: {step['Context']}\n"
                    for step in plan["Analysis_Suggestions"]
                )
            )
            #--------------------------------------
            self.Data_Discovery_Coder_System_Message="""
            **ROLE**
            Your role is to write python code (using shell code where necessary) to implement instructions.
            """
            self.Data_Discovery_Coder_Updated_System_Message=f"""
            ------------- ROLE --------------------------
            Your role is to write python code (using any shell code that is necessary for installing packages) to implement the instructions that have been provided to you.
            If numpy or pandas are used, you must ensure that you modify the print default options to allow more rows and columns to be printed to the console adn to increase the linewidth.
            The directory with data is called /inputs/ and is the only directory you have access to. This directory contains the files that are referenced in the instructions that you must follow.
            There is no need to attempt to explore the directory with shell scripts as you have already been provided with the necessary file names.
            The datasets you will be analysing are assumed to be very large (hundreds of MB) and so you must ensure your code efficiently handles the data.
            You must clearly and concisely print the results of this multivariate EDA to the console.
            If your code does not implement the instructions, then you will receive reviews and suggestions that you must use to update your code.
            ---------------------------------------------

            ------------- Task ----------------------------
            - You MUST use the Data_Discovery_Code_Update function to update your code.
            - You will be provided with feedback that you MUST use to update your code.
            - You will be provided with instructions for data analysis that you MUST follow. You must use the data file it suggests and perform the analysis type it requests.
            - The directory to explore is /inputs/. You must only explore this directory.
            - When printing out the results to the console, you must include information about what analysis was performed and on which variables before you print the result.
            -----------------------------------------------

            --------------- Context ------------------------
            **** The Coding Instructions *******
            {self.Plan_Step}
            *************************
            **** Code To Be Improved **** 
            {{Data_Discovery_Code}}
            *************************
            *** Feedback for Improving the Code (Compulsory) *** 
            {{Data_Discovery_Code_Reviews}}
            """
            self.Data_Discovery_Reviewer_System_Message="""
            **ROLE**
            Your role is to review python code that performs non-graphical Exploratory Data Analysis (EDA) on data files in the /inputs/ directory. This is the only directory that the code is allowed to explore.
            The code must perform multivariate EDA to produce a concise summary of the relationships between key variables. You must ensure that the data analysis is non-graphical and it must produce text-based output only.
            You have been provided with information about the univariate data analysis and the instructions for multivariate data analysis that must be followed.
            It is your role to enforce the code to follow these instructions. The code must print the results out to the console but it must never attempt to print out excessive amounts
            of information. The data files that are being analysed may be very large (hundreds of MB) and so the code must efficently handle all data files provided.
            """
            self.Data_Discovery_Reviewer_Updated_System_Message=f"""
            ------------- ROLE ---------------------------
            Your role is to review python code that performs non-graphical Exploratory Data Analysis (EDA) on data files in the /inputs/ directory. This is the only directory that the code is allowed to explore.
            If numpy or pandas are used, you must ensure that you modify the print default options to allow more rows and columns to be printed to the console adn to increase the linewidth.
            The code must perform multivariate EDA to produce a concise summary of the relationships between key variables. You must ensure that the data analysis is non-graphical and it must produce text-based output only.
            You have been provided with information about the univariate data analysis and the instructions for multivariate data analysis that must be followed.
            It is your role to enforce the code to follow these instructions. The code must print the results out to the console but it must never attempt to print out excessive amounts
            of information. The data files that are being analysed may be very large (hundreds of MB) and so the code must efficently handle all data files provided.
            ----------------------------------------------

            ------------- Task ---------------------------
            - You MUST use the Data_Discovery_Code_Review function to provide feedback.
            - If the code does not meet these requirements, you must provide actionable feedback to improve the code. You must only approve the code when the code meets all of the criteria. You must be strict with your enforcement of these rules.
            - If the code meets ALL of the criteria below, you MUST set Approval_Status to True. But you cannot undo this decision and so you must be certain.
            - Never approve code that does not meet all of the criteria.
            ----------------------------------------------
            ---------- Context ----------------------------
            *** The Code that Requires Feedback *** 
            {{Data_Discovery_Code}}
            *** The Instructions that the Code Must Follow (if applicable) ***
            {self.Plan_Step}
            """
            # The key structure of each file is given by: {metadata}  ----- Add this line once the Profile_Check function has been redone.
        else:
            raise ValueError("Invalid Stage specified. Must be 'Univariate' or 'Multivariate'.")
        self.Data_Discovery_Coder_llm_config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Coding").build_config()
        self.Data_Discovery_Reviewer_llm_config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Coding").build_config()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Coding").build_config()
        self.Data_Discovery_Coder=Data_Discovery_Coder_Agent(
            "Data_Discovery_Coder",
            self.Data_Discovery_Coder_llm_config,
            self.Data_Discovery_Coder_System_Message,
            self.Data_Discovery_Coder_Updated_System_Message
        )
        self.Data_Discovery_Reviewer=Data_Discovery_Reviewer_Agent(
            "Data_Discovery_Reviewer",
            self.Data_Discovery_Reviewer_llm_config,
            self.Data_Discovery_Reviewer_System_Message,
            self.Data_Discovery_Reviewer_Updated_System_Message
        )

        # Setup Executor -------------------------
        self.inputs_dir=Path("./Inputs")
        self.work_dir=Path("./LLM_Scripts")
        self.setup_dir=Path("./Singularity_Images")
        self.dummy_dir=Path("./Dummy_Dir")  # Directory to use instead of writable-tmpfs for pip installs.
        self.executor = SingularityCommandLineCodeExecutor(
            image="continuumio/anaconda3",
            timeout=60,
            work_dir=str(self.work_dir),
            setup_dir=str(self.setup_dir),
            inputs_dir=str(self.inputs_dir),
            pip_install_dir=str(self.dummy_dir),
        )
        self.Code_Executor_Agent = ConversableAgent("Code_Executor_Agent",
        llm_config=False,  # Turn off LLM for this agent.
        code_execution_config={"executor": self.executor,
        "last_n_messages": 3},  # Use the docker command line code executor.
        human_input_mode="NEVER",
        )

        register_function(
            Data_Discovery_Code_Update,
            caller=self.Data_Discovery_Coder.agent,
            executor=self.Data_Discovery_Coder.agent,
            name="Data_Discovery_Code_Update",
            description="Update the code based on the feedback."
        )
        register_function(
            Data_Discovery_Code_Review,
            caller=self.Data_Discovery_Reviewer.agent,
            executor=self.Data_Discovery_Reviewer.agent,
            name="Data_Discovery_Code_Review",
            description="Review the code."
        )

        # Setup Handoffs -----

        self.Code_Executor_Agent.handoffs.add_after_work( # This is the only add_after_work needed.
            OnContextCondition(
                target=AgentTarget(self.Data_Discovery_Coder.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Issues} == True")
                )
            )
        )

        self.Code_Executor_Agent.handoffs.add_after_work( # This is the only add_after_work needed.
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Issues} == False")
                )
            )
        )
        self.Data_Discovery_Coder.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Data_Discovery_Reviewer.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Data_Discovery_Code_Updated} == True")
                )
            )
        )

        self.Data_Discovery_Coder.agent.handoffs.set_after_work(AgentTarget(self.Data_Discovery_Coder.agent))

        self.Data_Discovery_Reviewer.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Data_Discovery_Coder.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Data_Discovery_Code_Suggestions_Available} == True & ${Data_Discovery_Code_Approval} == False")
                )
            )
        )
        self.Data_Discovery_Reviewer.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Code_Executor_Agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Data_Discovery_Code_Approval} == True")
                )
            )
        )
        self.Data_Discovery_Reviewer.agent.handoffs.set_after_work(AgentTarget(self.Data_Discovery_Reviewer.agent))

        # Removed chance for coder to see the output as it does not tend to change anything without feedback anyway.
        # Summariser has been removed and replaced with the EDA pre-planner.

        self.Code_Executor_Agent.register_hook("process_message_before_send",Execution_Results)
    def run_Conversation(self):
        # Transforming Message History (To Limit)----
        context_handling = transform_messages.TransformMessages(
            transforms=[transforms.MessageHistoryLimiter(max_messages=3)])
        context_handling.add_to_agent(self.Data_Discovery_Coder.agent)
        context_handling.add_to_agent(self.Data_Discovery_Reviewer.agent)
        #---------------------------
        pattern=DefaultPattern(
        initial_agent=self.Data_Discovery_Coder.agent,
        agents=[self.Data_Discovery_Coder.agent,
                self.Data_Discovery_Reviewer.agent,
                self.Code_Executor_Agent],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )

        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Discover the data in the directory.",
            max_rounds=50,
        )
        return result, ctx
class Pre_Planning_Chat:
    def __init__(self,context_variables: ContextVariables, Stage:str, RAG_Enabled:bool, Suggestions: bool):
        self.Stage=Stage
        self.context_variables = context_variables
        self.RAG_Enabled=RAG_Enabled
        self.Suggestions=Suggestions
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()

        self.Pre_Planner_System_Message="""
        You are a Summariser Agent.
        """
        if self.Stage == "Univariate":
            self.Pre_Planner_Updated_System_Message="""
            --------------- ROLE -------------------------------
            You are a Dataset Interpretation Agent. You interpret the results of Exploratory Data Analysis (EDA) to explain the structure of the dataset and concisely highlight any important results.
            You must explan this to an agent that cannot see the dataset or the EDA results and cannot ask for any clarification. The agent (who you are producing this report for) will use your report to analyse the data, but it cannot do its job if your report is not clear.
            You have been provided with some key questions to illustrate the types of insights that are important to convey (although these are illustrative and not exhaustive).
            Your report must clearly segregate the interpretation of each file present in the dataset.
            If there is not enough information to come to certain interpretations or conclusions, then you must suggest that further analysis is required, or state that you do not have enough information. Never make anything up that is not supported by the data or the EDA.
            ----------------------------------------------------
            -------------- Task ---------------------------------
            - You MUST use the EDA_Report function to provide a concise report.
            - You must provide a detailed report of the EDA performed and specifically outline the 
            main insights about the data that were discovered. This must be done for each file present in the dataset.
            ------------------------------------------------------
            ---------------- Questions to Consider **For Each Data File** (Illustrative NOT Exhaustive) ----------------
            - Does the data file follow a particular structure?
            - Do any variables contain all NaNs?
            - Is the data in the datset files sparse, dense, timeseries data, summary data, etc?
            - Is the data grouped into sections of that are relevant to a specific run/experiment?
            - Are any of the files only metadata that does not require further analysis?
            ------------------------------------------------------
            ------------------ Context ---------------------------
             *****The Univariate Analysis EDA Results ******
            {Univariate_Analysis}
            """
            self.Suggestion_Agent_System_Message="""
            You are a Suggestion Agent.
            """
            self.Suggestion_Agent_Updated_System_Message=f"""
            --------------- ROLE -------------------------------
            You are a suggestion agent that specialises in providing suggestions for **Non-Graphical** Multvariate Exploratory Data Analysis (EDA).
            You have been provided with the findings from a Univariate EDA stage to help you understand what has already been performed on the datset and the key conclusions that have been drawn.
            You must focus on suggesting multvariate EDA that will improve understanding of a dataset structure. The future results of the analysis you suggest will help other agents understand how to use the data. 
            You must only suggest non-graphical methods and you must ensure that your sugestions are relevant to the dataset. You have been provided with some important questions (illustrative not exhaustive) that you should consider when making your suggestions.
            You must focus on making suggestions for multivariate EDA that contrbutes to understanding the dataset and you should not repeat what has already been performed in the univariate EDA stage.
            Your suggestions must be clear and have explainability so that a coder agent (who cannot see the dataset and cannot ask for clarification) can implement them.
            ----------------------------------------------------
            -------------- Task ---------------------------------
            - You MUST use the EDA_Suggestions function to clearly state your multivariate EDA suggestions.
            - You must clearly state the analysis type to be performed.
            - You must clearly outline the data file to be used and list the variables that should be involved in the analysis.
            - You should provide any explanations of the reasoning behind your suggestions.
            - It should be specified that all results must be printed to the console and never attempt to produce graphical output or write to any data files.
            - You are only allowed to suggest an analysis on one file at a time. You can never attempt to suggest analysis that uses multiple files at once. This is because the strucutre of each file depends on the relationshp between variables in the same file.
            -----------------------------------------------------
            ---------- Output Format ----------------------------
            - Data File: You must clearly state one data file at a time.
            - Analysis Type: You must clearly state the analysis type to be performed using a concise name. For example (illustrative not exhaustive): "Correlation", "Covariance", etc.
            - Variables Involved: You must list the variables that this analysis should focus on.
            - Explanation: You must provide a concise explanation of why this analysis is important and how it will help understand the dataset structure.
            For each data file, you can suggest a list of multiple different analysis types and the corresponding variables and explanations.
            You must use these suggestions once per data file. Each data file can have mulitple analysis types listed, but only one plan per data file.

            Correct Format: {{"Suggestions":[{{"Data_File":"X.csv","Analysis_Suggestions":[{{"Analysis_Type":"XYZ","Variables":["var1","var2"],"Context":"ABC"}}]}},{{"Data_File":"Y.csv","Analysis_Suggestions":[{{"Analysis_Type":"XYZ","Variables":["var1","var2"],"Context":"ABC"}}]}}]}}
            ---------------- Context ----------------------------
            ***** The Univariate Analysis EDA Results ******
            {{Univariate_Analysis}}
            ***** Metadata For Each File ******
            {{metadata}}
            """
            self.RAG_System_Message="""
            Your are a RAG agent.
            """
            self.RAG_Updated_System_Message="""
            **ROLE**
            Your role is to query a knowledge database to obtain domain-specific information that will
            enhance the Univariate Exploratory Data Analysis (EDA) report. This topic is aimed at understanding
            the context of Biopharmaceutical manufacturing, specifically focusing on High Performance Liquid
            Chromatography (HPLC) and Mass Spectrometry data files. Your contribution is to ask relevant questions
            that will help gather crucial domain knowledge to enrich the EDA report. Your questions must focus primarly on the 
            multivariate relationships that should be explored to better understand the dataset.

            **Task**
            - You MUST use the RAG function to query the database.
            - You MUST ask a variety of different questions to ensure a sufficient breadth of information is obtained.
            - You must focus on questions that will provide insights into the nature of HPLC and Mass Spectrometry data files
            and their typical characteristics in a Biopharmaceutical manufacturing context.
            - You MUST use some of the variable names to help guide your questioning to retrieve the most relevant information.

            **Context**
            The Univariate Analysis of the data is provided by: {Univariate_Analysis} 
            """
        elif self.Stage=="Multivariate":
            self.Pre_Planner_System_Message="""
            You are a Summariser Agent.
            """
            self.Pre_Planner_Updated_System_Message="""
            --------------- ROLE -------------------------------
            You are a Dataset Interpretation Agent. You interpret the results of Exploratory Data Analysis (EDA) to explain the structure of the dataset and concisely highlight any important results.
            You must explan this to an agent that cannot see the dataset or the EDA results and cannot ask for any clarification. The agent (who you are producing this report for) will use your report to analyse the data, but it cannot do its job if your report is not clear.
            You have been provided with some key questions to illustrate the types of insights that are important to convey (although these are illustrative and not exhaustive).
            You have been provided with the interpretation of a previous EDA stage. You MUST not assume that this report is comprehensive or correct, but it should only act as a guide if observations are consistent with the new EDA results. If you are not ujsre, you must not comment on it.
            Your report must clearly segregate the interpretation of each file present in the dataset.
            If there is not enough information to come to certain interpretations or conclusions, then you must suggest that further analysis is required, or state that you do not have enough information. Never make anything up that is not supported by the data or the EDA.
            ----------------------------------------------------
            -------------- Task ---------------------------------
            - You MUST use the EDA_Report function to provide a concise report.
            - You must provide a detailed report of the EDA performed and specifically outline the 
            main insights about the data that were discovered. This must be done for each file present in the dataset.
            ------------------------------------------------------
            ---------------- Questions to Consider **For Each Data File** (Illustrative NOT Exhaustive) ----------------
            - Does the data file follow a particular structure?
            - Do any variables contain all NaNs?
            - Is the data in the datset files sparse, dense, timeseries data, summary data, etc?
            - Is the data grouped into sections of that are relevant to a specific run/experiment?
            - Are there any variables that are linked together that must be analysed or plotted jointly? (Example: Time series data dependent on time variables).
            - Are any of the files only metadata that does not require further analysis?
            ------------------------------------------------------
            --------------- Context ------------------------------
            ****** The Initial EDA Report ********
            {Univariate_EDA_Report}
            ***** The Results of the New Multivariate EDA *******
            {Multivariate_Analysis}
            """
            
        else:
            raise ValueError("Invalid Stage specified. Must be 'Univariate' or 'Multivariate'.")

        self.Pre_Planner_llm_config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Suggestion_llm_config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.RAG_llm_config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        self.Pre_Planner=Pre_Planner_Agent(
            "Pre_Planner",
            self.Pre_Planner_llm_config,
            self.Pre_Planner_System_Message,
            self.Pre_Planner_Updated_System_Message
        )
        self.Suggestion_Agent=Suggestion_Agent(
            "Suggestion_Agent",
            self.Suggestion_llm_config,
            self.Suggestion_Agent_System_Message,
            self.Suggestion_Agent_Updated_System_Message,
        )
        if self.RAG_Enabled:
            self.RAG_Agent=RAG_Agents(
                "RAG_Agent",
                self.RAG_llm_config,
                self.RAG_System_Message,
                self.RAG_Updated_System_Message
            )

        register_function(
            EDA_Report,
            caller=self.Pre_Planner.agent,
            executor=self.Pre_Planner.agent,
            name="EDA_Report",
            description="Provide a detailed report based on the EDA performed."
        )
        register_function(
            EDA_Suggestions,
            caller=self.Suggestion_Agent.agent,
            executor=self.Suggestion_Agent.agent,
            name="EDA_Suggestions",
            description="Provide suggestions based on the EDA performed."
        )
        if self.RAG_Enabled:
            register_function(
                RAG,
                caller=self.RAG_Agent.agent,
                executor=self.RAG_Agent.agent,
                name="RAG",
                description="Retrieve domain-specific information using RAG."
            )

        # Handoffs -------------------------
        if self.RAG_Enabled:
            if self.Stage=="Univariate":
                self.Pre_Planner.agent.handoffs.add_context_condition(
                    OnContextCondition(
                        target=AgentTarget(self.RAG_Agent.agent),
                        condition=ExpressionContextCondition(
                            expression=ContextExpression("${Univariate_Report_Available} == True")
                        )
                    )
                )
                self.Pre_Planner.agent.handoffs.set_after_work(AgentTarget(self.Pre_Planner.agent))
                self.RAG_Agent.agent.handoffs.add_context_condition(
                    OnContextCondition(
                        target=AgentTarget(self.Pre_Planner.agent),
                        condition=ExpressionContextCondition(
                            expression=ContextExpression("${Univariate_RAG_Used} == True")
                        )
                    )
                )
            elif self.Stage=="Multivariate":
                self.Pre_Planner.agent.handoffs.add_context_condition(
                    OnContextCondition(
                        target=AgentTarget(self.RAG_Agent.agent),
                        condition=ExpressionContextCondition(
                            expression=ContextExpression("${Multivariate_Report_Available} == True")
                        )
                    )
                )
                self.Pre_Planner.agent.handoffs.set_after_work(AgentTarget(self.Pre_Planner.agent))
                self.RAG_Agent.agent.handoffs.add_context_condition(
                    OnContextCondition(
                        target=AgentTarget(self.Pre_Planner.agent),
                        condition=ExpressionContextCondition(
                            expression=ContextExpression("${Multivariate_RAG_Used} == True")
                        )
                    )
                )
            else:
                raise ValueError("Invalid Stage specified. Must be 'Univariate' or 'Multivariate'.")
            self.RAG_Agent.agent.handoffs.set_after_work(AgentTarget(self.RAG_Agent.agent))
        else:
            if self.Suggestions==True:
                self.Pre_Planner.agent.handoffs.set_after_work(AgentTarget(self.Pre_Planner.agent))
                self.Pre_Planner.agent.handoffs.add_context_condition(
                    OnContextCondition(
                        target=AgentTarget(self.Suggestion_Agent.agent),
                        condition=ExpressionContextCondition(
                            expression=ContextExpression("${Univariate_Report_Available} == True or ${Multivariate_Report_Available} == True")
                        )
                    )
                )
                self.Suggestion_Agent.agent.handoffs.set_after_work(AgentTarget(self.Suggestion_Agent.agent))
                self.Suggestion_Agent.agent.handoffs.add_context_condition(
                    OnContextCondition(
                        target=TerminateTarget(),
                        condition=ExpressionContextCondition(
                            expression=ContextExpression("${Suggestions_Available} == True")
                        )
                    )
                )
            else:
                self.Pre_Planner.agent.handoffs.set_after_work(AgentTarget(self.Pre_Planner.agent))
                self.Pre_Planner.agent.handoffs.add_context_condition(
                    OnContextCondition(
                        target=TerminateTarget(),
                        condition=ExpressionContextCondition(
                            expression=ContextExpression("${Univariate_Report_Available} == True or ${Multivariate_Report_Available} == True")
                        )
                    )
                )


    def run_Conversation(self):
        # Transforming Message History (To Limit)----
        context_handling = transform_messages.TransformMessages(
            transforms=[transforms.MessageHistoryLimiter(max_messages=3)])
        context_handling.add_to_agent(self.Pre_Planner.agent)
        if self.RAG_Enabled:
            context_handling.add_to_agent(self.RAG_Agent.agent)
        #---------------------------
        if self.RAG_Enabled:
            agents=[self.Pre_Planner.agent,
                    self.RAG_Agent.agent]
            max_rounds=25
        else:
            agents=[self.Pre_Planner.agent, self.Suggestion_Agent.agent]
            max_rounds=10
        pattern=DefaultPattern(
        initial_agent=self.Pre_Planner.agent,
        agents=agents,
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )

        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Data Analysis stage completed.",
            max_rounds=max_rounds,
        )
        return result, ctx

class Data_Discovery_Chat:
    def __init__(self,context_variables: ContextVariables):
        self.context_variables = context_variables

    def run_Conversation(self):
        eda_analysis=EDA_Analysis(self.context_variables, Stage="Univariate")
        eda_analysis.run_Conversation()
        Univariate_planning=Pre_Planning_Chat(self.context_variables, Stage="Univariate", RAG_Enabled=False, Suggestions=False)
        Univariate_planning.run_Conversation()
        self.context_variables["Multivariate_Suggestions"]=False # Needs to be reset.
        self.context_variables["Domain_Knowledge"]=[]  # Initialize Domain Knowledge for RAG storage.
        self.context_variables["Data_Discovery_Code"]="" # Reset to avoid overlap
        self.context_variables["Data_Discovery_Code_Reviews"]="" # Reset to avoid overlap
        self.context_variables["Univariate_Report_Available"]=False  # Reset to avoid termination immediately.
        #print(self.context_variables["Univariate_Suggestions"])

        #--------------- Multivariate EDA May NOT be required ----------------
        # idx=0
        # for plan in self.context_variables["Univariate_Suggestions"]:
        #     eda_analysis=EDA_Analysis(self.context_variables, Stage="Multivariate",Index=idx)
        #     eda_analysis.run_Conversation()
        #     idx+=1
        # # Assemble the Multivarate EDA Results.
        # for idx in range(0,EDA_Analysis.Index+1):
        #     self.context_variables["Multivariate_Analysis"]+=f"\n *** Results for Data File: {self.context_variables['Univariate_Suggestions'][idx]['Data_File']} *** \n {self.context_variables[f'Multivariate_Analysis_{idx}']}\n"
        # print(self.context_variables["Multivariate_Analysis"])
        # Univariate_planning=Pre_Planning_Chat(self.context_variables, Stage="Multivariate", RAG_Enabled=False)
        # Univariate_planning.run_Conversation()
        #---------------------------------------------------------------------
    


# ============== Functions to Pass to Agents =======================
def Data_Discovery_Code_Update(
    Code: Annotated[Code_Response, "The Python code to be updated."],
    context_variables: ContextVariables) -> ReplyResult:

    """Update the code based on the feedback."""

    context_variables[f"Data_Discovery_Code"] = Code.model_dump()
    context_variables[f"Data_Discovery_Code_Updated"] = True

    context_variables[f"Shell_Code_Data_Discovery"] = f"\n```sh\n{Code.Shell_Code}\n```"
    context_variables[f"Python_Code_Data_Discovery"] = f"\n```python\n{Code.Python_Code}\n```"

    # Resetting Code Reviewer Context variables to be used with the necessary handoffs.
    context_variables[f"Data_Discovery_Code_Suggestions_Available"] = False
    context_variables[f"Data_Discovery_Code_Approval"] = False
    context_variables["Issues"] = False  # Reset Issues flag for code execution checking.

    return ReplyResult(message="Data_Discovery_Code updated successfully.", context_variables=context_variables)

def Data_Discovery_Code_Review(
    Code_Reviews: Annotated[str, "Review suggestions for the code."],
    Approval_Status: Annotated[bool, "Does the code need to be revised (False). Is the code complete and approved (True)."],
    context_variables: ContextVariables) -> ReplyResult:
    
    context_variables[f"Data_Discovery_Code_Reviews"] = Code_Reviews
    context_variables[f"Data_Discovery_Code_Suggestions_Available"] = True
    context_variables[f"Data_Discovery_Code_Revision_Count"] += 1

    if Approval_Status:
        context_variables[f"Data_Discovery_Code_Approval"] = True
        context_variables[f"Iteration_Count"] += 1
        message=f"""Code Approved. The code must be repeated as follows:
    {context_variables[f"Shell_Code_Data_Discovery"]}
    {context_variables[f"Python_Code_Data_Discovery"]}
    """
    else:
        message="Code reviews stored successfully. Ready for update."
    # Resetting Coder Context variables to be used with the necessary handoffs.
    context_variables[f"Data_Discovery_Code_Updated"] = False

    return ReplyResult(message=message, context_variables=context_variables)

def EDA_Report(Report: Annotated[str,"The detailed report based on the EDA performed."],
    context_variables: ContextVariables) -> ReplyResult:
    """Store the EDA report."""
    if EDA_Analysis.Stage == "Univariate":
        context_variables[f"Univariate_EDA_Report"] = Report
        context_variables["Univariate_Report_Available"] = True
        context_variables["Univariate_RAG_Used"] = False  # Reset RAG used flag for new report.
    elif EDA_Analysis.Stage == "Multivariate":
        context_variables["Multivariate_Report_Available"] = True
        context_variables[f"Multivariate_EDA_Report"] = Report
        context_variables["Multivariate_RAG_Used"] = False  # Reset RAG used flag for new report.
    else:
        raise ValueError("Invalid Stage specified. Must be 'Univariate' or 'Multivariate'.")
    return ReplyResult(message="EDA_Report stored successfully.", context_variables=context_variables)

def EDA_Suggestions(Suggestions: Annotated[List[PlanResponse],"Your suggestions for further data analysis."],
    context_variables: ContextVariables) -> ReplyResult:
    """Store the EDA suggestions."""
    context_variables["Univariate_Suggestions"] = Suggestions
    context_variables["Suggestions_Available"]=True # Needs to be reset.
    return ReplyResult(message="EDA_Report stored successfully.", context_variables=context_variables)

def RAG(first_query: Annotated[str, "The query to the RAG"],second_query: Annotated[str, "The query to the RAG"],third_query: Annotated[str, "The query to the RAG"], context_variables: ContextVariables) -> ReplyResult:
    results1 = RAG_Tool(first_query)
    results2 = RAG_Tool(second_query)
    results3 = RAG_Tool(third_query)

    # Limit to most recent 3-5 queries to prevent context overflow
    MAX_RAG_HISTORY = 3
    context_variables["Domain_Knowledge"].append({"Query": first_query, "Results": results1})
    context_variables["Domain_Knowledge"].append({"Query": second_query, "Results": results2})
    context_variables["Domain_Knowledge"].append({"Query": third_query, "Results": results3})
    
    # Keep only the most recent queries
    if len(context_variables["Domain_Knowledge"]) > MAX_RAG_HISTORY:
        context_variables["Domain_Knowledge"] = context_variables["Domain_Knowledge"][-MAX_RAG_HISTORY:]
    
    context_variables["Univariate_RAG_Used"] = True

    #Reset flags for Reporter.
    if EDA_Analysis.Stage == "Univariate":
        context_variables["Univariate_Report_Available"] = False
    elif EDA_Analysis.Stage == "Multivariate":
        context_variables["Multivariate_Report_Available"] = False
    else:
        raise ValueError("Invalid Stage specified. Must be 'Univariate' or 'Multivariate'.")

    return ReplyResult(
        message="RAG results stored successfully.",
        context_variables=context_variables,
    )

def Execution_Results(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    """Hook to check code execution results and update context variables."""
    # Extract content from message
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = str(message)
    
    # Check for execution errors
    if "exitcode: 1" in content or "execution failed" in content.lower() or "error occurred:" in content:
        sender.context_variables["Issues"] = True
        sender.context_variables["Data_Discovery_Code_Approval"] = False # Retracts approval to ensure that the code changes are reviewed in light of the plan to ensure no large scale changes are made.
        # Modify the message to include error info
        if isinstance(message, dict):
            message["content"] = f"CODE EXECUTION FAILED:\n{content}\n\nPlease fix the code."
            return message
        else:
            return f"CODE EXECUTION FAILED:\n{content}\n\nPlease fix the code."
    
    if "exitcode: 0" in content:
        sender.context_variables["Issues"] = False
        # Modify the message to include success info  
        if isinstance(message, dict):
            message["content"] = f"CODE EXECUTION SUCCESSFUL:\n{content}"
        else:
            message = f"CODE EXECUTION SUCCESSFUL:\n{content}"
        if EDA_Analysis.Stage == "Univariate":
            sender.context_variables["Univariate_Analysis"] = content
        elif EDA_Analysis.Stage == "Multivariate":
            sender.context_variables[f"Multivariate_Analysis_{EDA_Analysis.Index}"] = content
        else:
            raise ValueError("Invalid Stage specified. Must be 'Univariate' or 'Multivariate'.")
    
    # Return message unchanged if no execution result found
    return message

#===================================================================


def main():
    context_variables = ContextVariables({
        "Data_Discovery_Code": "",
        "Data_Discovery_Code_Updated": False,
        "Data_Discovery_Code_Reviews": "",
        "Data_Discovery_Code_Suggestions_Available": False,
        "Data_Discovery_Code_Approval": False,
        "Data_Discovery_Code_Revision_Count": 0,
        "Data_Discovery_Summary": "",
        "Iteration_Count": 0,
    })
    data_discovery_room=Data_Discovery_Chat(context_variables=context_variables)
    data_discovery_room.run_Conversation()

    with open("Data_Discovery_Conversation.json", "w") as f:
        json.dump(context_variables.model_dump(), f, indent=2)

if __name__ == "__main__":
    main()