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
from typing import Any, Dict, List, Optional, Annotated, Tuple, Union
from autogen import UpdateSystemMessage
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
import numpy as np
import pandas as pd
import json
import csv
import os
from autogen.tools.experimental import DeepResearchTool
from autogen.tools.experimental import ReliableTool

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
import copy
import pprint
import re
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from autogen import Agent
from Review_Panel import Reality_Chat, Focus_Area_Chat, Review_Compilation, RAG_System, Output_Instruction
#===========================================================
load_dotenv()

# ------------------Structured Output-------------------
class Step(BaseModel):
    Step_Number: int = Field(..., description="The step number in the plan.")
    Analysis_Type: str = Field(..., description="A clear statement of the analysis type that must be performed by the coder.")
    Data_File: str = Field(..., description="Explictly name the one data file that must be used for the analysis.")
    Variables: List[str] = Field(..., description="A list of variable names from the data file that must be used for the analysis.")
    Context: str = Field(..., description="Any important information that the coder must be aware of when handling the data.")
    Output_Format: str = Field(..., description="The output format that must be used for the analysis step. This must be either text-based output appended (never overwrite) to a markdown file OR visualisations saved as image files. It can be a combination of both")
    Output_Details: str=Field(..., description="Detailed instructions on how the output must be formatted. If the output is a graph, specify the type of graph, labels, title, and any other relevant details. If the output is text-based, specify the structure and content required.")


class PlanResponse(BaseModel):
    Plan_Section: List[Step] = Field(..., description="A list of steps for the Plan.")
    Number_of_Steps: int = Field(..., description="The total number of steps in the plan.")
    #HPLC_Analysis_Plan: List[Step] = Field(..., description="A list of steps for the HPLC Analysis Plan.")
    #Mass_Spectrometry_Plan: List[Step] = Field(..., description="A list of steps for the Mass Spectrometry Analysis Plan.")
#-----------------------

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

class Usability_Planner_Agent(Agent_Base):
    pass
class Usability_RAG_Agent(Agent_Base):
    pass
class Usability_Reviewer_Agent(Agent_Base):
    pass
class HPLC_Analysis_Planner_Agent(Agent_Base):
    pass
class HPLC_Analysis_RAG_Agent(Agent_Base):
    pass
class HPLC_Analysis_Reviewer_Agent(Agent_Base):
    pass
class Mass_Spectrometry_Planner_Agent(Agent_Base):
    pass
class Mass_Spectrometry_RAG_Agent(Agent_Base):
    pass
class Mass_Spectrometry_Reviewer_Agent(Agent_Base):
    pass   

class Planner_Reviewer_Chat:
    Usability_Idx=1
    HPLC_Idx=1
    MS_Idx=1
    def __init__(self, context_variables: ContextVariables, Section_To_Run: str, Max_Rounds: int, Max_Plan_Steps: int=3):
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.section_to_run=Section_To_Run
        self.max_rounds=Max_Rounds
        self.context_variables = context_variables
        self.max_plan_steps=Max_Plan_Steps
        self.context_variables["Max_Plan_Steps"]= self.max_plan_steps
        self.Usability_Planner_System_Message="""
        You are a Planner Agent that specialises in planning the data analysis for the usability of the data.
        """
        self.Usability_Planner_Update_System_Message="""
        ---------------  ROLE  ----------------
        You are an expert planner for designing detailed and actionable plans for your designated focus area.
        You develop data-driven plans that instruct a code writer to implement your plan to analyse data.
        You never attempt to solve the problem yourself. Instead you rely on other agents to implement your plan.
        You listen to feedback from reviewers and advice from experts to iteratively improve your current plan.
        You never falsify any information and you must always base your plan on the data available.
        The maximum number of steps you can have in your plan is {Max_Plan_Steps}.
        ----------------------------------------

        ------------ Focus Area ----------------
        {Focus_Area_Usability}
        ----------------------------------------

        ---------------- Task ------------------
        Your task is to create a **DATA-DRIVEN** analysis plan that enables meaningful and reliable interpretation of the future analysis results.
        - To Update your plan, you MUST use the Update_Usability_Plan function to detail your plan.
        - Each plan step must be independently executable as each step will be given to different code writers that work independently.
        - You have been provided with summaries of an Exploratory Data Analysis (EDA) phase that details the structure of the data and the important relationships between variables (not exhaustive). This knowledge will enable your plan to be data-driven.
        - You have been provided with reviews and advice on how to improve your plan. You MUST always use this feedback to improve your plan.
        - You MUST decide whether the data analysis should be mainly text-based output or visualisations.
        ----------------------------------------

        ----------- Output Format ---------------
        You must produce a detailed data-driven plan. The plan must be instructive to ensure effective implementation by a code writer.
        - Analysis: You must be clear what analysis type is to be performed in each step. You must be specific
        - Data: You must explicitly name one data file in each step and the variables that must be used in the analysis.
        - Additional Context: You must provide any additional comments to enable a coder to implement the step effectively. This may include specific python packages that must be used, specific information about the data to be aware of (e.g. data split into batches, data is summary data, data is time-series data, data has peaks that are relevant). You should clearly instruct the coder on how to handle these data characteristics.
        - Output Format: The two output formats allowed are:
                        * Text based outputs MUST be appended (never overwrite) to a markdown file called "Usability.md"
                        * Visualisations MUST be saved as image files.
        - Output Details: Detailed instructions on how the output must be formatted. If the output is a graph, specify the type of graph, labels, title, and any other relevant details. If the output is text-based, specify the structure and content required.
                          You should indicate clearly how the coder should format the output to ensure consistency across all steps. Text based output must only fill a maximum of 20 lines per step.
        These are explanations of how to fill out some of the mandatory felds in this output format:
        {{"Plan": {{"Plan_Section": [{{"Step_Number": 1, "Analysis_Type": "XYZ Analysis", "Data_File": "X.csv", "Variables": ["X","Y","Z"], "Context": "ABC Context", "Output_Format": "XYZ Output", "Output_Details": "XYZ Details"}}, {{"Step_Number": 2, "Analysis_Type": "XYZ Analysis", "Data_File": "X.csv", "Variables": ["X,Y,Z"], "Context": "ABC Context", "Output_Format": "XYZ Output", "Output_Details": "XYZ Details"}}], "Number_of_Steps": "N"}},  "Review_Request": true}}
        -----------------------------------------

        --------- Important Themes ---------------
        The following themes may be important to consider for your data analysis instructions, but they are not exhaustive:
        - Data Completeness Analysis.
        - Machine Learning Methods.
        - Signal integrity and noise level assessment.
        - Studying outliers or anomalies in the data.
        -------------------------------------------


        ---------------- Context ------------------- 
        ***** Information about the datasets (Exploratory Data Analysis reports) *****
        {Univariate_EDA_Report} 
        {Multivariate_EDA_Report}
        ***************

        ****** Current Plan ******
        {Usability_Plan}
        *******************

        ****** Variables Available (Per File) ******
        {metadata}
        ********************

        ******** Improvement Instructions ******
        {Usability_Reviewer_Suggestions}
        {Compiled_Review}
        {Usability_Plan_Review} 
        {Usability_Steps_To_Add}
        """
        self.HPLC_Analysis_Planner_System_Message="""
        Your are a Planner Agent that specialises in planning the data analysis for the High Performance Liquid Chromatography (HPLC) data.
        """
        self.HPLC_Analysis_Planner_Update_System_Message="""
        ---------------  ROLE  ----------------
        You are an expert planner for designing detailed and actionable plans for your designated focus area.
        You develop data-driven plans that instruct a code writer to implement your plan to analyse data. You MUST ONLY use data related to the focus area.
        You never attempt to solve the problem yourself. Instead you rely on other agents to implement your plan.
        You listen to feedback from reviewers and advice from experts to iteratively improve your current plan.
        You never falsify any information and you must always base your plan on the data available.
        The maximum number of steps you can have in your plan is {Max_Plan_Steps}.
        ----------------------------------------

        ------------ Focus Area ----------------
        {Focus_Area_HPLC}
        ----------------------------------------

        ---------------- Task ------------------
        Your task is to create a **DATA-DRIVEN** analysis plan that enables meaningful and reliable interpretation of the future analysis results.
        - To Update your plan, you MUST use the Update_HPLC_Plan function to detail your plan.
        - Each plan step must be independently executable as each step will be given to different code writers that work independently.
        - You have been provided with summaries of an Exploratory Data Analysis (EDA) phase that details the structure of the data and the important relationships between variables (not exhaustive). This knowledge will enable your plan to be data-driven.
        - You have been provided with reviews and advice on how to improve your plan. You MUST always use this feedback to improve your plan.
        - You MUST decide whether the data analysis should be mainly text-based output or visualisations.
        ----------------------------------------

        ----------- Output Format ---------------
        You must produce a detailed data-driven plan. The plan must be instructive to ensure effective implementation by a code writer.
        - Analysis: You must be clear what analysis type is to be performed in each step. You must be specific
        - Data: You must explicitly name one data file in each step and the variables that must be used in the analysis.
        - Additional Context: You must provide any additional comments to enable a coder to implement the step effectively. This may include specific python packages that must be used, specific information about the data to be aware of (e.g. data split into batches, data is summary data, data is time-series data, data has peaks that are relevant). You should clearly instruct the coder on how to handle these data characteristics.
        - Output Format: The two output formats allowed are:
                        * Text based outputs MUST be appended (never overwrite) to a markdown file called "HPLC.md"
                        * Visualisations MUST be saved as image files.
        - Output Details: Detailed instructions on how the output must be formatted. If the output is a graph, specify the type of graph, labels, title, and any other relevant details. If the output is text-based, specify the structure and content required.
                          You should indicate clearly how the coder should format the output to ensure consistency across all steps. Text based output must only fill a maximum of 20 lines per step.
        These are explanations of how to fill out some of the mandatory felds in this output format:
        {{"Plan": {{"Plan_Section": [{{"Step_Number": 1, "Analysis_Type": "XYZ Analysis", "Data_File": "X.csv", "Variables": ["X","Y","Z"], "Context": "ABC Context", "Output_Format": "XYZ Output", "Output_Details": "XYZ Details"}}, {{"Step_Number": 2, "Analysis_Type": "XYZ Analysis", "Data_File": "X.csv", "Variables": ["X,Y,Z"], "Context": "ABC Context", "Output_Format": "XYZ Output", "Output_Details": "XYZ Details"}}], "Number_of_Steps": "N"}},  "Review_Request": true}}
        -----------------------------------------

        --------- Important Themes for Analysis ---------------
        The following themes may be important to consider for your data analysis instructions, but they are not exhaustive:
        - Peak Idenitifcation Techniques
        - Machine Learning Methods.
        - Exploring advanced techniques for peak integration.
        - Exploring advanced techniques for handling noisy data.

        The analysis plan should not attempt to encourage any data quality assessments as this has been given to a different agent to carry out.
        -------------------------------------------
        ---------------- Context ------------------- 
        ***** Information about the datasets (Exploratory Data Analysis reports) *****
        {Univariate_EDA_Report} 
        {Multivariate_EDA_Report}
        ***************

        ****** Current Plan ******
        {HPLC_Analysis_Plan}
        *******************
        ****** Variables Available (Per File) ******
        {metadata}
        ********************
        ******** Suggestions for Improvement ******
        {HPLC_Reviewer_Suggestions} 
        {Compiled_Review}
        {HPLC_Analysis_Plan_Review}
        {HPLC_Analysis_Steps_To_Add}
        """
        self.Mass_Spectrometry_Planner_System_Message="""
        Your are a Planner Agent that specialises in planning the data analysis for the Mass Spectrometry (MS) data.
        """
        self.Mass_Spectrometry_Planner_Update_System_Message="""
        ---------------  ROLE  ----------------
        You are an expert planner for designing detailed and actionable plans for your designated focus area.
        You develop data-driven plans that instruct a code writer to implement your plan to analyse data. You MUST ONLY use data related to the focus area.
        You never attempt to solve the problem yourself. Instead you rely on other agents to implement your plan.
        You listen to feedback from reviewers and advice from experts to iteratively improve your current plan.
        You never falsify any information and you must always base your plan on the data available.
        The maximum number of steps you can have in your plan is {Max_Plan_Steps}.
        ----------------------------------------

        ------------ Focus Area ----------------
        {Focus_Area_MS}
        ----------------------------------------

        ---------------- Task ------------------
        Your task is to create a **DATA-DRIVEN** analysis plan that enables meaningful and reliable interpretation of the future analysis results.
        - To Update your plan, you MUST use the Update_MS_Plan function to detail your plan.
        - Each plan step must be independently executable as each step will be given to different code writers that work independently.
        - You have been provided with summaries of an Exploratory Data Analysis (EDA) phase that details the structure of the data and the important relationships between variables (not exhaustive). This knowledge will enable your plan to be data-driven.
        - You have been provided with reviews and advice on how to improve your plan. You MUST always use this feedback to improve your plan.
        - You MUST decide whether the data analysis should be mainly text-based output or visualisations.
        ----------------------------------------

        ----------- Output Format ---------------
        You must produce a detailed data-driven plan. The plan must be instructive to ensure effective implementation by a code writer.
        - Analysis: You must be clear what analysis type is to be performed in each step. You must be specific
        - Data: You must explicitly name one data file in each step and the variables that must be used in the analysis.
        - Additional Context: You must provide any additional comments to enable a coder to implement the step effectively. This may include specific python packages that must be used, specific information about the data to be aware of (e.g. data split into batches, data is summary data, data is time-series data, data has peaks that are relevant). You should clearly instruct the coder on how to handle these data characteristics.
        - Output Format: The two output formats allowed are:
                        * Text based outputs MUST be appended (never overwrite) to a markdown file called "MS.md"
                        * Visualisations MUST be saved as image files.
        - Output Details: Detailed instructions on how the output must be formatted. If the output is a graph, specify the type of graph, labels, title, and any other relevant details. If the output is text-based, specify the structure and content required.
                          You should indicate clearly how the coder should format the output to ensure consistency across all steps. Text based output must only fill a maximum of 20 lines per step.
        These are explanations of how to fill out some of the mandatory felds in this output format:
        {{"Plan": {{"Plan_Section": [{{"Step_Number": 1, "Analysis_Type": "XYZ Analysis", "Data_File": "X.csv", "Variables": ["X","Y","Z"], "Context": "ABC Context", "Output_Format": "XYZ Output", "Output_Details": "XYZ Details"}}, {{"Step_Number": 2, "Analysis_Type": "XYZ Analysis", "Data_File": "X.csv", "Variables": ["X,Y,Z"], "Context": "ABC Context", "Output_Format": "XYZ Output", "Output_Details": "XYZ Details"}}], "Number_of_Steps": "N"}},  "Review_Request": true}}
        -----------------------------------------

        --------- Important Themes for Analysis ---------------
        The following themes may be important to consider for your data analysis instructions, but they are not exhaustive:
        - Peak Idenitifcation Techniques
        - Machine Learning Methods.
        - Exploring advanced techniques for handling noisy data.

        The analysis plan should not attempt to encourage any data quality assessments as this has been given to a different agent to carry out.
        -------------------------------------------
        ---------------- Context ------------------- 
        ***** Information about the datasets (Exploratory Data Analysis reports) *****
        {Univariate_EDA_Report} 
        {Multivariate_EDA_Report}
        ***************

        ****** Current Plan ******
        {Mass_Spectrometry_Plan}
        *******************
        ****** Variables Available (Per File) ******
        {metadata}
        ********************
        ******** Suggestions for Improvement ******
        {MS_Reviewer_Suggestions} 
        {Compiled_Review}
        {Mass_Spectrometry_Plan_Review}
        {Mass_Spectrometry_Steps_To_Add}
        """
        self.Usability_Reviewer_System_Message="""
        You are a Reviewer Agent. You critically evaluate analysis plans by examining whether proposed methods are appropriate for the actual data structure available.
        The plan is about the Usability of the data and data wrangling techniques that can be applied to High Performance Liquid Chromatography (HPLC) and Mass Spectrometry (MS) data.
        """
        self.Usability_Reviewer_Update_System_Message="""
        -------- ROLE ------------
        You are an expert data science plan critic that suggests feedback to improve analysis plans for a designated focus area.
        You ensure that the plan is comprehensive, data-driven and actionable by a code writer that cannot see the data or ask for clarification.
        You must ensure that only data files related to the focus area are used in the plan.
        You must ensure that the plan is valid and variable selection is valid for any visualisations or analysis suggested.
        You never run out of ideas to improve the plan, but if the plan is valid then you must focus on other areas that are not valid. If all areas are valid, then only minor improvements can be made.
    
        --------- Focus Area ------------
        The focus area for the plan is provided to you to ensure your feedback is relevant.
        {Focus_Area_Usability}
        ----------------------------------
        ----------- Task -----------------
        Your task is to revew and critique the current plan for data analysis ensuring that is is relevant and **DATA-DRIVEN** to enable meaningful and reliable interpretation of the future analysis results.
        - You provide feedback by calling the Submit_Usability_Feedback function.
        - Each step in the analysis plan must be independently executable as each step will be given to different code writers that work independently.
        - You must ensure that the plan is suitable to the focus area provided and ensure that there is a good mixture of basic and advanced analysis techniques.
        Examples (Not Exhaustive) of this are:
                 * Basic statistical analysis techniques to understand data distributions. 
                 * More advanced concepts may suggest machine learning techniques to explore patterns in the data.
        - You MUST ensure that the structure of the dataset is clearly communicated, where relevant, to support the analysis suggestions. This is because the coder will not have access to the data and cannot ask for clarification.
        Examples (Not Exhaustive) of this are:
                 * Analysis type, file name and variable names must be clearly stated in each step.
                 * If the data is time-series data, then the plan must state this and suggest appropriate time-series analysis techniques.
        - You have been provided with summaries of an Exploratory Data Analysis (EDA) phase that details the structure of the data and the important relationships between variables (not exhaustive). This knowledge is essential to ensure a **DATA-DRIVEN** plan.

        ----------- Context -----------------
        ******* Current Plan********
        {Usability_Plan}
        ****** Information about the dataset (Exploratory Data Analysis reports) *****
        {Univariate_EDA_Report} 
        {Multivariate_EDA_Report}
        ****** Variables Available (Per File) ******
        {metadata}
        ********************
        ***** Data Analysis Feedback (Only After Plan has been implemented and outputs reviewed) *****
        {Usability_Plan_Review} 
        {Usability_Steps_To_Add}
        """
        self.HPLC_Analysis_Reviewer_System_Message="""
        You are a Reviewer Agent. You critically evaluate analysis plans by examining whether proposed methods are appropriate for the actual data structure available.
        The plan is about the data analysis of High Performance Liquid Chromatography (HPLC) data.
        """
        self.HPLC_Analysis_Reviewer_Update_System_Message="""
        -------- ROLE ------------
        You are an expert data science plan critic that suggests feedback to improve analysis plans for a designated focus area.
        You ensure that the plan is comprehensive, data-driven and actionable by a code writer that cannot see the data or ask for clarification.
        You must ensure that only data files related to the focus area are used in the plan.
        You must ensure that the plan is valid and variable selection is valid for any visualisations or analysis suggested.
        You never run out of ideas to improve the plan, but if the plan is vald then you must focus on other areas that are not valid. If all areas are valid, then only minor improvements can be made.

        --------- Focus Area ------------
        The focus area for the plan is provided to you to ensure your feedback is relevant.
        {Focus_Area_HPLC}
        ----------------------------------
        ----------- Task -----------------
        Your task is to revew and critique the current plan for data analysis ensuring that is is relevant and **DATA-DRIVEN** to enable meaningful and reliable interpretation of the future analysis results.
        - You provide feedback by calling the Submit_HPLC_Feedback function.
        - Each step in the analysis plan must be independently executable as each step will be given to different code writers that work independently.
        - You must ensure that the plan is suitable to the focus area provided and ensure that there is a good mixture of basic and advanced analysis techniques.
        Examples (Not Exhaustive) of this are:
                 * Basic techniques for calibration curves using regression. 
                 * More advanced concepts may suggest machine learning techniques to identify peaks.
        - You MUST ensure that the structure of the dataset is clearly communicated, where relevant, to support the analysis suggestions. This is because the coder will not have access to the data and cannot ask for clarification.
        Examples (Not Exhaustive) of this are:
                 * Analysis type, file name and variable names must be clearly stated in each step.
                 * If the data is time-series data, then the plan must state this and suggest appropriate time-series analysis techniques.
        - You have been provided with summaries of an Exploratory Data Analysis (EDA) phase that details the structure of the data and the important relationships between variables (not exhaustive). This knowledge is essential to ensure a **DATA-DRIVEN** plan.

        ----------- Context -----------------
        ******* Current Plan********
        {HPLC_Analysis_Plan}
        ****** Information about the dataset (Exploratory Data Analysis reports) *****
        {Univariate_EDA_Report} 
        {Multivariate_EDA_Report}
        ****** Variables Available (Per File) ******
        {metadata}
        ********************
        ***** Data Analysis Feedback (Only After Plan has been implemented and outputs reviewed) *****
        {HPLC_Analysis_Plan_Review} 
        {HPLC_Analysis_Steps_To_Add}
        """
        self.Mass_Spectrometry_Reviewer_System_Message="""
        You are a Reviewer Agent. You critically evaluate analysis plans by examining whether proposed methods are appropriate for the actual data structure available.
        The plan is about the data analysis of Mass Spectrometry (MS) data.
        """
        self.Mass_Spectrometry_Reviewer_Update_System_Message="""
        -------- ROLE ------------
        You are an expert data science plan critic that suggests feedback to improve analysis plans for a designated focus area.
        You ensure that the plan is comprehensive, data-driven and actionable by a code writer that cannot see the data or ask for clarification.
        You must ensure that only data files related to the focus area are used in the plan.
        You must ensure that the plan is valid and variable selection is valid for any visualisations or analysis suggested.
        You never run out of ideas to improve the plan, but if the plan is vald then you must focus on other areas that are not valid. If all areas are valid, then only minor improvements can be made.

        --------- Focus Area ------------
        The focus area for the plan is provided to you to ensure your feedback is relevant.
        {Focus_Area_MS}
        ----------------------------------
        ----------- Task -----------------
        Your task is to revew and critique the current plan for data analysis ensuring that is is relevant and **DATA-DRIVEN** to enable meaningful and reliable interpretation of the future analysis results.
        - You provide feedback by calling the Submit_MS_Feedback function.
        - Each step in the analysis plan must be independently executable as each step will be given to different code writers that work independently.
        - You must ensure that the plan is suitable to the focus area provided and ensure that there is a good mixture of basic and advanced analysis techniques.
        Examples (Not Exhaustive) of this are:
                 * Basic techniques for calibration curves using regression. 
                 * More advanced concepts may suggest machine learning techniques to identify peaks.
        - You MUST ensure that the structure of the dataset is clearly communicated, where relevant, to support the analysis suggestions. This is because the coder will not have access to the data and cannot ask for clarification.
        Examples (Not Exhaustive) of this are:
                 * Analysis type, file name and variable names must be clearly stated in each step.
                 * If the data is time-series data, then the plan must state this and suggest appropriate time-series analysis techniques.
        - You have been provided with summaries of an Exploratory Data Analysis (EDA) phase that details the structure of the data and the important relationships between variables (not exhaustive). This knowledge is essential to ensure a **DATA-DRIVEN** plan.

        ----------- Context -----------------
        ******* Current Plan********
        {Mass_Spectrometry_Plan}
        ****** Information about the dataset (Exploratory Data Analysis reports) *****
        {Univariate_EDA_Report} 
        {Multivariate_EDA_Report}
        ****** Variables Available (Per File) ******
        {metadata}
        ********************
        ***** Data Analysis Feedback (Only After Plan has been implemented and outputs reviewed) *****
        {Mass_Spectrometry_Plan_Review} 
        {Mass_Spectrometry_Steps_To_Add}
        """
        Usability_Planner_Name="Usability_Planner"
        HPLC_Analysis_Planner_Name="HPLC_Analysis_Planner"
        Mass_Spectrometry_Planner_Name="Mass_Spectrometry_Planner"
        Usability_Reviewer_Name="Usability_Reviewer"
        HPLC_Analysis_Reviewer_Name="HPLC_Analysis_Reviewer"
        Mass_Spectrometry_Reviewer_Name="Mass_Spectrometry_Reviewer"

        self.Usability_Planner_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.HPLC_Analysis_Planner_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Mass_Spectrometry_Planner_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Usability_Reviewer_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.HPLC_Analysis_Reviewer_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Mass_Spectrometry_Reviewer_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()

        self.Usability_Planner=Usability_Planner_Agent(Usability_Planner_Name,
                                            self.Usability_Planner_LLM_Config,
                                            self.Usability_Planner_System_Message,
                                            self.Usability_Planner_Update_System_Message)
        self.HPLC_Analysis_Planner=HPLC_Analysis_Planner_Agent(HPLC_Analysis_Planner_Name,
                                            self.HPLC_Analysis_Planner_LLM_Config,
                                            self.HPLC_Analysis_Planner_System_Message,
                                            self.HPLC_Analysis_Planner_Update_System_Message)
        self.Mass_Spectrometry_Planner=Mass_Spectrometry_Planner_Agent(Mass_Spectrometry_Planner_Name,
                                            self.Mass_Spectrometry_Planner_LLM_Config,
                                            self.Mass_Spectrometry_Planner_System_Message,
                                            self.Mass_Spectrometry_Planner_Update_System_Message)
        self.Usability_Reviewer=Usability_Reviewer_Agent(Usability_Reviewer_Name,
                                            self.Usability_Reviewer_LLM_Config,
                                            self.Usability_Reviewer_System_Message,
                                            self.Usability_Reviewer_Update_System_Message)
        self.HPLC_Analysis_Reviewer=HPLC_Analysis_Reviewer_Agent(HPLC_Analysis_Reviewer_Name,
                                            self.HPLC_Analysis_Reviewer_LLM_Config,
                                            self.HPLC_Analysis_Reviewer_System_Message,
                                            self.HPLC_Analysis_Reviewer_Update_System_Message)
        self.Mass_Spectrometry_Reviewer=Mass_Spectrometry_Reviewer_Agent(Mass_Spectrometry_Reviewer_Name,
                                            self.Mass_Spectrometry_Reviewer_LLM_Config,
                                            self.Mass_Spectrometry_Reviewer_System_Message,
                                            self.Mass_Spectrometry_Reviewer_Update_System_Message)
        # Hooks For Error Handling:
        self.Usability_Planner.agent.register_hook("process_message_before_send",Error_Handling_Hook)
        self.HPLC_Analysis_Planner.agent.register_hook("process_message_before_send",Error_Handling_Hook)
        self.Mass_Spectrometry_Planner.agent.register_hook("process_message_before_send",Error_Handling_Hook)
        #---Handoffs -----------------------------------------------------------------------------
        # The Plans to be developed wil happen in teh following Order: Usability, HPLC, MS. Only the MS reviewer will handoff
        # to terminate once the enitre plan has been fully approved.

        # Usability----------------------
        self.Usability_Planner.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Usability_Reviewer.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Usability_Plan_Updated} == True")
                )
            )
        )
        self.Usability_Planner.agent.handoffs.set_after_work(AgentTarget(self.Usability_Planner.agent)) # In case the Planner Agent hasn't called the correct functions.
        
        self.Usability_Reviewer.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Usability_Planner.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Usability_Feedback} == True") # & ${Usability_Plan_Approval} == False")
                )
            )
        )
        self.Usability_Reviewer.agent.handoffs.set_after_work(AgentTarget(self.Usability_Reviewer.agent)) # In case the Reviewer Agent hasn't called the correct functions.

        #========== REMOVED =====================
        #JUSTIFICATION: Plan approval requires inherent understanding of what is not included in the plan. But, agents do not know what they do not know.
        # ie: They don't have enough understanding to know when to stop improving the plan and so the more iterations, the  more likelihood of understanding the braod topic area by exploration.
        #========================================
        # When Plan is approved:
       # self.Usability_Reviewer.agent.handoffs.add_context_condition(
          #  OnContextCondition(
               # target=AgentTarget(self.HPLC_Analysis_Planner.agent),
                #condition=ExpressionContextCondition(
                  #  expression=ContextExpression("${Usability_Plan_Approval} == True")
               # )
            #)
        #)

         # HPLC----------------------

        self.HPLC_Analysis_Planner.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.HPLC_Analysis_Reviewer.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${HPLC_Plan_Updated} == True")
                )
            )
        )
        self.HPLC_Analysis_Planner.agent.handoffs.set_after_work(AgentTarget(self.HPLC_Analysis_Planner.agent)) # In case the Planner Agent hasn't called the correct functions.

        self.HPLC_Analysis_Reviewer.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.HPLC_Analysis_Planner.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${HPLC_Feedback} == True")
                )
            )
        )
        self.HPLC_Analysis_Reviewer.agent.handoffs.set_after_work(AgentTarget(self.HPLC_Analysis_Reviewer.agent)) # In case the Reviewer Agent hasn't called the correct functions.

        #========== REMOVED =====================
        #JUSTIFICATION: Plan approval requires inherent understanding of what is not included in the plan. But, agents do not know what they do not know.
        # ie: They don't have enough understanding to know when to stop improving the plan and so the more iterations, the  more likelihood of understanding the braod topic area by exploration.
        #========================================
        # When Plan is approved:
        #self.HPLC_Analysis_Reviewer.agent.handoffs.add_context_condition(
           # OnContextCondition(
                #target=AgentTarget(self.Mass_Spectrometry_Planner.agent),
               # condition=ExpressionContextCondition(
                   # expression=ContextExpression("${HPLC_Analysis_Plan_Approval} == True")
                #)
           # )
       # )

         # MS ----------------------
        
        self.Mass_Spectrometry_Planner.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Mass_Spectrometry_Reviewer.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${MS_Plan_Updated} == True")
                )
            )
        )
        self.Mass_Spectrometry_Planner.agent.handoffs.set_after_work(AgentTarget(self.Mass_Spectrometry_Planner.agent)) # In case the Planner Agent hasn't called the correct functions.

        self.Mass_Spectrometry_Reviewer.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Mass_Spectrometry_Planner.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${MS_Feedback} == True")
                )
            )
        )
        self.Mass_Spectrometry_Reviewer.agent.handoffs.set_after_work(AgentTarget(self.Mass_Spectrometry_Reviewer.agent)) # In case the Reviewer Agent hasn't called the correct functions.

        #========== REMOVED =====================
        #JUSTIFICATION: Plan approval requires inherent understanding of what is not included in the plan. But, agents do not know what they do not know.
        # ie: They don't have enough understanding to know when to stop improving the plan and so the more iterations, the  more likelihood of understanding the braod topic area by exploration.
        #========================================
        # When Plan is approved:
        #self.Mass_Spectrometry_Reviewer.agent.handoffs.add_context_condition(
            #OnContextCondition(
                #target=TerminateTarget(),
                #condition=ExpressionContextCondition(
                    #expression=ContextExpression("${Mass_Spectrometry_Plan_Approval} == True")
               # )
          #  )
        #)

        
    def run_Conversation(self):
        # Reset Re-Used Context Variables:
        self.context_variables["RAG_Skipped"]=0
        self.context_variables["Reviewer_Skipped"]=0
       # Transforming Message History (To Limit)----
        context_handling = transform_messages.TransformMessages(
            transforms=[transforms.MessageHistoryLimiter(max_messages=3)])
        context_handling.add_to_agent(self.Usability_Planner.agent)
        context_handling.add_to_agent(self.Usability_Reviewer.agent)
        context_handling.add_to_agent(self.HPLC_Analysis_Planner.agent)
        context_handling.add_to_agent(self.HPLC_Analysis_Reviewer.agent)
        context_handling.add_to_agent(self.Mass_Spectrometry_Planner.agent)
        context_handling.add_to_agent(self.Mass_Spectrometry_Reviewer.agent)
        #--------------------------------------------
        register_function(
            Update_Usability_Plan,
            caller=self.Usability_Planner.agent,
            executor=self.Usability_Planner.agent,
            name="Update_Usability_Plan",
            description="Update the usability plan in context variables. You MUST follow the format of the PlanResponse object. You must not forget to incude the argument, Plan."
        )
        register_function(
            Update_HPLC_Plan,
            caller=self.HPLC_Analysis_Planner.agent,
            executor=self.HPLC_Analysis_Planner.agent,
            name="Update_HPLC_Plan",
            description="Update the HPLC analysis plan in context variables."
        )
        register_function(
            Update_MS_Plan,
            caller=self.Mass_Spectrometry_Planner.agent,
            executor=self.Mass_Spectrometry_Planner.agent,
            name="Update_MS_Plan",
            description="Update the Mass Spectrometry analysis plan in context variables."
        )
        register_function(
            Submit_Usability_Feedback,
            caller=self.Usability_Reviewer.agent,
            executor=self.Usability_Reviewer.agent,
            name="Submit_Usability_Feedback",
            description="Submit usability feedback and indicate if revision is needed."
        )
        register_function(
            Submit_HPLC_Feedback,
            caller=self.HPLC_Analysis_Reviewer.agent,
            executor=self.HPLC_Analysis_Reviewer.agent,
            name="Submit_HPLC_Feedback",
            description="Submit HPLC analysis feedback and indicate if revision is needed."
        )
        register_function(
            Submit_MS_Feedback,
            caller=self.Mass_Spectrometry_Reviewer.agent,
            executor=self.Mass_Spectrometry_Reviewer.agent,
            name="Submit_MS_Feedback",
            description="Submit Mass Spectrometry analysis feedback and indicate if revision is needed."
        )

        if self.section_to_run=="Usability":
            agents=[self.Usability_Planner.agent,
        self.Usability_Reviewer.agent]
        elif self.section_to_run=="HPLC":
            agents=[self.HPLC_Analysis_Planner.agent,
        self.HPLC_Analysis_Reviewer.agent]
        elif self.section_to_run=="MS":
            agents=[self.Mass_Spectrometry_Planner.agent,
        self.Mass_Spectrometry_Reviewer.agent]
        else:
            print("Please select a valid section to run: Usability, HPLC, MS.")

        pattern=DefaultPattern(
        initial_agent=agents[0],
        agents=agents,
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,)

        result, ctx, _ = initiate_group_chat(
        pattern=pattern,
        messages="Using the information available, come up with a plan for the data.",
        max_rounds=self.max_rounds)
    
        return result, ctx

def Update_Usability_Plan(
    Plan: Annotated[PlanResponse, "The full structured plan response. You must follow the format provided and include the argument, Plan, as well as the full response format with the Number_of_Steps and Plan_Section keys."],
    context_variables: ContextVariables,
) -> ReplyResult:
    """Update the plan in context variables and set flags for handoff."""
    context_variables["Usability_Plan"] = Plan.model_dump() # Stores full Plan in context variables.

    message="Plan updated successfully."
    context_variables["Usability_Plan_Updated"] = True # Plan has been updated by the Planner Agent. 
    
    context_variables["last_speaker"]="Usability_Planner"
        
    # Resetting Reviewer Context Variables. This will ensure that the Reviewer Agent CAN'T handoff to Planner
    #without doing anything. This MUST be combined with the Reviewer Agent handoffs, where it has to stay with
    #the Reviewer until it calls the function to submit reviews.
    context_variables["Usability_Reviewer_Suggestions"]=""
    context_variables["Usability_Feedback"]=False
    context_variables["Usability_RAG_Used"]=False
   
    return ReplyResult(
        message=message,
        context_variables=context_variables,
    )

def Update_HPLC_Plan(
    Plan: Annotated[PlanResponse, "The full structured plan response. You must follow the format provided and include the argument, Plan, as well as the full response format with the Number_of_Steps and Plan_Section keys."],
    context_variables: ContextVariables,
) -> ReplyResult:
    """Update the plan in context variables and set flags for handoff."""
    context_variables["HPLC_Analysis_Plan"] = Plan.model_dump() # Stores full Plan in context variables.

    message="Plan updated successfully."
    context_variables["HPLC_Plan_Updated"] = True # Plan has been updated by the Planner Agent. 
    
    context_variables["last_speaker"]="HPLC_Analysis_Planner"
        
    # Resetting Reviewer Context Variables. This will ensure that the Reviewer Agent CAN'T handoff to Planner
    #without doing anything. This MUST be combined with the Reviewer Agent handoffs, where it has to stay with
    #the Reviewer until it calls the function to submit reviews.
    context_variables["HPLC_Reviewer_Suggestions"]=""
    context_variables["HPLC_Feedback"]=False
    context_variables["HPLC_RAG_Used"]=False
   
    return ReplyResult(
        message=message,
        context_variables=context_variables,
    )

def Update_MS_Plan(
    Plan: Annotated[PlanResponse, "The full structured plan response. You must follow the format provided and include the argument, Plan, as well as the full response format with the Number_of_Steps and Plan_Section keys."],
    context_variables: ContextVariables,
) -> ReplyResult:
    """Update the plan in context variables and set flags for handoff."""
    context_variables["Mass_Spectrometry_Plan"] = Plan.model_dump() # Stores full Plan in context variables.

    message="Plan updated successfully."
    context_variables["MS_Plan_Updated"] = True # Plan has been updated by the Planner Agent. 
    
    context_variables["last_speaker"]="Mass_Spectrometry_Planner"
        
    # Resetting Reviewer Context Variables. This will ensure that the Reviewer Agent CAN'T handoff to Planner
    #without doing anything. This MUST be combined with the Reviewer Agent handoffs, where it has to stay with
    #the Reviewer until it calls the function to submit reviews.
    context_variables["MS_Reviewer_Suggestions"]=""
    context_variables["MS_Feedback"]=False
    context_variables["MS_RAG_Used"]=False
   
    return ReplyResult(
        message=message,
        context_variables=context_variables,
    )
def Submit_Usability_Feedback(
    feedback: Annotated[Optional[str], "Detailed feedback on the plan with specific suggestions for improvement."],
    #Approve_Plan: Annotated[bool, "True if the plan is approved, False if the plan is not approved."],
    context_variables: ContextVariables,
) -> ReplyResult:
    """Store reviewer feedback and indicate if revision is needed."""

    #if Approve_Plan:
       # message="Usability Plan is approved."
        #context_variables["Usability_Plan_Approval"]=True
       
    #else:
        #message="""Feedback submitted. Usability Plan requires revision.
        #"""
    message="""Feedback submitted. Usability Plan requires revision."""
    context_variables["Usability_Feedback"]=True
    context_variables["Usability_Reviewer_Suggestions"]=feedback
    context_variables["last_speaker"]="Usability_Reviewer"

    # Resetting Planner and RAG Agent Context Variables. This will ensure that the Planner Agent CAN'T handoff
    # to reviewer without doing anything. This does not enforce the Planner Agent to produce
    # a notably different plan however. (This is for Reviewer to decide on).
    context_variables["Usability_Plan_Updated"]=False
    context_variables["Usability_RAG_Used"]=False
    context_variables["Usability_Review_Request"]=False

    #----------- Review Panel -------------- Note: Context variables are injected automatcally into the planner system message. There may be the need to have a final agent that consolidates all feedback. Currently, this runs after the reviewer agent has generated its general feedback.
    idx=Planner_Reviewer_Chat.Usability_Idx

    Reality_Chat1=Reality_Chat(context_variables=context_variables, Analysis_Type="Usability", Max_Rounds=10, Iteration=idx)
    Reality_Chat1.run_Conversation()
    FA1=Focus_Area_Chat(context_variables=context_variables, Analysis_Type="Usability", Max_Rounds=5, Iteration=idx)
    FA1.run_Conversation()
    RAG1=RAG_System(context_variables=context_variables, Analysis_Type="Usability", Max_Rounds=10)
    RAG1.run_Conversation()
    OP1=Output_Instruction(context_variables=context_variables, Analysis_Type="Usability", Max_Rounds=10, Iteration=idx)
    OP1.run_Conversation()
    Compile1=Review_Compilation(context_variables=context_variables, Analysis_Type="Usability", Max_Rounds=5, Iteration=idx)
    Compile1.run_Conversation()

    Planner_Reviewer_Chat.Usability_Idx+=1
    #---------------------------------------
    return ReplyResult(
        message=message,
        context_variables=context_variables,
    )
def Submit_HPLC_Feedback(
    feedback: Annotated[Optional[str], "Detailed feedback on the plan with specific suggestions for improvement."],
    #Approve_Plan: Annotated[bool, "True if the plan is approved, False if the plan is not approved."],
    context_variables: ContextVariables,
) -> ReplyResult:
    """Store reviewer feedback and indicate if revision is needed."""

    #if Approve_Plan:
       # message="Usability Plan is approved."
        #context_variables["Usability_Plan_Approval"]=True
       
    #else:
        #message="""Feedback submitted. Usability Plan requires revision.
        #"""
    message="""Feedback submitted. Usability Plan requires revision."""
    context_variables["HPLC_Feedback"]=True
    context_variables["HPLC_Reviewer_Suggestions"]=feedback
    context_variables["last_speaker"]="HPLC_Reviewer"

    # Resetting Planner and RAG Agent Context Variables. This will ensure that the Planner Agent CAN'T handoff
    # to reviewer without doing anything. This does not enforce the Planner Agent to produce
    # a notably different plan however. (This is for Reviewer to decide on).
    context_variables["HPLC_Plan_Updated"]=False
    context_variables["HPLC_RAG_Used"]=False
    context_variables["HPLC_Review_Request"]=False
    #----------- Review Panel -------------- Note: Context variables are injected automatcally into the planner system message. There may be the need to have a final agent that consolidates all feedback. Currently, this runs after the reviewer agent has generated its general feedback.
    idx=Planner_Reviewer_Chat.HPLC_Idx

    Reality_Chat1=Reality_Chat(context_variables=context_variables, Analysis_Type="HPLC", Max_Rounds=10, Iteration=idx)
    Reality_Chat1.run_Conversation()
    FA1=Focus_Area_Chat(context_variables=context_variables, Analysis_Type="HPLC", Max_Rounds=5, Iteration=idx)
    FA1.run_Conversation()
    RAG1=RAG_System(context_variables=context_variables, Analysis_Type="HPLC", Max_Rounds=10)
    RAG1.run_Conversation()
    OP1=Output_Instruction(context_variables=context_variables, Analysis_Type="HPLC", Max_Rounds=10, Iteration=idx)
    OP1.run_Conversation()
    Compile1=Review_Compilation(context_variables=context_variables, Analysis_Type="HPLC", Max_Rounds=5, Iteration=idx)
    Compile1.run_Conversation()

    Planner_Reviewer_Chat.HPLC_Idx+=1
    #---------------------------------------
    return ReplyResult(
        message=message,
        context_variables=context_variables,
    )
def Submit_MS_Feedback(
    feedback: Annotated[Optional[str], "Detailed feedback on the plan with specific suggestions for improvement."],
    #Approve_Plan: Annotated[bool, "True if the plan is approved, False if the plan is not approved."],
    context_variables: ContextVariables,
) -> ReplyResult:
    """Store reviewer feedback and indicate if revision is needed."""

    #if Approve_Plan:
       # message="Usability Plan is approved."
        #context_variables["Usability_Plan_Approval"]=True
       
    #else:
        #message="""Feedback submitted. Usability Plan requires revision.
        #"""
    message="""Feedback submitted. Usability Plan requires revision."""
    context_variables["MS_Feedback"]=True
    context_variables["MS_Reviewer_Suggestions"]=feedback
    context_variables["last_speaker"]="MS_Reviewer"

    # Resetting Planner and RAG Agent Context Variables. This will ensure that the Planner Agent CAN'T handoff
    # to reviewer without doing anything. This does not enforce the Planner Agent to produce
    # a notably different plan however. (This is for Reviewer to decide on).
    context_variables["MS_Plan_Updated"]=False
    context_variables["MS_RAG_Used"]=False
    context_variables["MS_Review_Request"]=False
    #----------- Review Panel -------------- Note: Context variables are injected automatcally into the planner system message. There may be the need to have a final agent that consolidates all feedback. Currently, this runs after the reviewer agent has generated its general feedback.
    idx=Planner_Reviewer_Chat.MS_Idx

    Reality_Chat1=Reality_Chat(context_variables=context_variables, Analysis_Type="MS", Max_Rounds=10, Iteration=idx)
    Reality_Chat1.run_Conversation()
    FA1=Focus_Area_Chat(context_variables=context_variables, Analysis_Type="MS", Max_Rounds=5, Iteration=idx)
    FA1.run_Conversation()
    RAG1=RAG_System(context_variables=context_variables, Analysis_Type="MS", Max_Rounds=10)
    RAG1.run_Conversation()
    OP1=Output_Instruction(context_variables=context_variables, Analysis_Type="MS", Max_Rounds=10, Iteration=idx)
    OP1.run_Conversation()
    Compile1=Review_Compilation(context_variables=context_variables, Analysis_Type="MS", Max_Rounds=5, Iteration=idx)
    Compile1.run_Conversation()

    Planner_Reviewer_Chat.MS_Idx+=1
    #---------------------------------------
    return ReplyResult(
        message=message,
        context_variables=context_variables,
    )

def Error_Handling_Hook(sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    """Hook to check code execution results and update context variables."""

    # Extract content from message
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = str(message)
    
    correct_format="""{"Plan": {"Plan_Section": [{"Step_Number": 1, "Analysis_Type": "XYZ Analysis", "Data_File": "X.csv", "Variables": ["X","Y","Z"], "Context": "ABC Context", "Output_Format": "XYZ Output"}, {"Step_Number": 2, "Analysis_Type": "XYZ Analysis", "Data_File": "X.csv", "Variables": ["X,Y,Z"], "Context": "ABC Context", "Output_Format": "XYZ Output"}], "Number_of_Steps": "N"},  "Review_Request": true}"""

    if sender.context_variables["last_speaker"] == "Usability_Planner":
        if isinstance(message, dict):
            message["content"]=f"""{content} \n\n Reminder of the correct response format: {correct_format}"""
            return message
        else:
            return f"""{content} \n\n Reminder of the correct response format: {correct_format}"""
    elif sender.context_variables["last_speaker"] == "HPLC_Planner":
        if isinstance(message, dict):
            message["content"]=f"{content} \n\nReminder of the correct response format: {correct_format}"
            return message
        else:
            return f"{content} \n\nReminder of the correct response format: {correct_format}"
    elif sender.context_variables["last_speaker"] == "MS_Planner":
        if isinstance(message, dict):
            message["content"]=f"{content} \n\nReminder of the correct response format: {correct_format}"
            return message
        else:
            return f"{content} \n\nReminder of the correct response format: {correct_format}"
    else:
        pass
    return message

def main():
    context_variables=ContextVariables({
        "metadata": [],
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
        "Usability_Review_Request": False,
        "HPLC_Review_Request": False,
        "MS_Review_Request": False,
        "RAG_Skipped": 0,
        "last_speaker": "",
        "Reviewer_Skipped": 0,
        "Focus_Area_Usability": """ The focus area is assessing the usability and quality of High Performance Liquid Chromatography (HPLC) and mass spectrometry (MS) data for downstream analysis. This includes evaluating data completeness, structural consistency, signal integrity, noise characteristics, and run-to-run or batch-level reliability. The assessment should determine whether the data is fit for intended analytical use, identify quality risks or limitations, and define criteria that guide reliable interpretation of future analysis results. """,
        "Focus_Area_HPLC": """ The focus area is to analyze the High Performance Liquid Chromatography (HPLC) data to extract meaningful insights about the chromatographic separation process. This includes evaluating retention times, peak shapes, resolution between peaks, and overall chromatographic performance. It is crucial that an abilty to identify chemical compounds is achievable from the future analysis results. """,
        "Focus_Area_MS": """ The focus area is to analyze the mass spectrometry """,
        "Multivariate_EDA_Report": """ The univariate exploratory data analysis (EDA) has been conducted on three datasets: metadata chromatography_combined.csv, MS_combined.csv, and chromatography_combined.csv. The analysis reveals key characteristics of each dataset and provides insights into the experimental data collection process.\n\n1. **metadata chromatography_combined.csv**:\n   - The dataset contains 354 rows and 58 columns, with a mix of numerical and categorical variables.\n   - Key numerical variables include 'Load_volume_(mL)', 'Load_pH', 'Sample_start_volume_(mL)', 'Sample_volume_after_dilution_with_HPW_(mL)', 'Sample_final_volume_after_pH_titration_(mL)', 'Total_protein_concentration_(bradford)_(mg/mL)', 'Conducivity_(mS)', 'fraction_volume', and 'fraction_volume_ml_min_precise'. These variables exhibit a wide range of values, with 'fraction_volume' showing high variability (mean = 33.38, std = 52.99), suggesting diverse fraction collection volumes across runs.\n   - Categorical variables such as 'Resin_type', 'column', 'chromatography_stage', 'Titration_agent', 'EQ_and_wash_buffer', 'Elution_buffer', and 'Filter_type_used' indicate a structured experimental design with multiple process steps and material choices. The presence of 'Start_pH' and 'Final_pH' suggests pH titration was a critical step in sample preparation.\n   - The 'Sample_Code' and 'Run_Name' variables show high cardinality (344 and 11 unique values, respectively), indicating a large number of unique samples and runs, which aligns with a high-throughput biopharmaceutical manufacturing environment.\n\n2. **MS_combined.csv**:\n   - This dataset is significantly larger (1,912,029 rows) and contains 30 columns, primarily related to mass spectrometry (MS) data.\n   - Key numerical variables include 'Response', '%_of_response', 'Observed_neutral_mass_(Da)', 'Observed_m/z', 'Observed_TIC_RT_(mins)', 'Observed_UV_RT_(mins)', 'Observed RT delta (mins)'. The 'Response' variable has a very high mean (9,920.20) and standard deviation (79,605.62), indicating a broad dynamic range in signal intensity, typical of MS data.\n   - The 'Unique_MS_Sample_ID' and 'Sample_Code' variables suggest that MS data is linked to specific samples, with 283 and 274 unique IDs, respectively. The 'Replicate' variable (values 1, 2, 3) indicates that multiple replicates were collected for each sample, supporting robustness in the data.\n   - The 'Spectrum_type' variable has three unique values, which may correspond to different types of MS spectra (e.g., MS1, MS2, MS3), suggesting a multi-stage MS analysis.\n\n3. **chromatography_combined.csv**:\n   - This dataset contains 1,084,069 rows and 37 columns, focusing on chromatographic fractionation data.\n   - Key numerical variables include 'volume_ml', 'UV_1_280_ml', 'UV_1_280_mAU', 'Cond_ml', 'Cond_mS_cm', 'Conc_B_ml', 'Conc_B_%', 'Injection_ml', 'Run_Log_ml', 'Fraction_ml', 'UV_1_280_CUT_TEMP_100_BASEM_ml', 'UV_1_280_CUT_TEMP_100_BASEM_mAU', 'UV_2_260_ml', 'UV_2_260_mAU', 'pH_ml', 'pH_pH', 'DeltaC_pressure_ml', 'DeltaC_pressure_MPa', 'System_flow_ml', 'System_flow_ml_min', 'Sample_flow_ml', 'Sample_flow_ml_min', 'Fraction_number', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise', and 'fraction_volume'.\n   - The 'fraction_volume' variable has a mean of 20.93 and a high standard deviation (26.11), indicating variability in fraction collection volumes, which may reflect differences in flow rates, column conditions, or elution profiles.\n   - The 'chromatography_stage' and 'chromatography_stage_order' variables suggest a multi-step chromatography process, with stage order ranging from 2 to 20, indicating a complex purification workflow.\n   - The 'Fraction_number' variable ranges from 0 to 490, suggesting that fractions were collected in a systematic manner across multiple runs.\n\nInsights into Data Collection:\n- The experimental design appears to be highly structured, with multiple process steps (e.g., load, equilibrate, wash, elute) and material choices (e.g., resin type, column type).\n- The presence of replicate data (Replicate in MS_combined.csv) and multiple runs (run_no, run) indicates that the experiments were conducted in a controlled, repeatable manner to ensure data reliability.\n- The use of pH titration (Start_pH and Final_pH) and buffer selection (EQ_and_wash_buffer, Elution_buffer) highlights the importance of pH and ionic strength in the purification process.\n- The large number of unique samples and fractions suggests a high-throughput screening or production environment, typical of biopharmaceutical manufacturing.\n\nOverall, the univariate analysis confirms that the datasets are rich in experimental metadata, chromatographic data, and MS data, all of which are essential for downstream analysis such as quality control, process optimization, and biomolecule characterization.""",
        "Univariate_EDA_Report": "",
    })
    planning_room1=Planner_Reviewer_Chat(context_variables=context_variables, Section_To_Run="Usability", Max_Rounds=50) # Options are Usability, HPLC, MS
    planning_room1.run_Conversation()
    planning_room2=Planner_Reviewer_Chat(context_variables=context_variables, Section_To_Run="HPLC", Max_Rounds=50) # Options are Usability, HPLC, MS
    planning_room2.run_Conversation()
    planning_room3=Planner_Reviewer_Chat(context_variables=context_variables, Section_To_Run="MS", Max_Rounds=50) # Options are Usability, HPLC, MS
    planning_room3.run_Conversation()

# Store the context variables as a JSON file for troubleshooting.
    with open("Planner_Reviewer.json", "w") as f:
        json.dump(context_variables.model_dump(), f, indent=2)



if __name__ =="__main__":
    main()