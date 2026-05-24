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
#===========================================================
load_dotenv()

# ------------------Structured Output-------------------
class Step(BaseModel):
    Step_Number: int = Field(..., description="The step number in the plan.")
    Variables_Feedback: str = Field(..., description="Feedback on the variables suggested for this step.")
    Scope: str = Field(..., description="Detailed information for improving the plan base on the focus area provided.")
    Research: str = Field(..., description="Domain specific knowledge and instructions for how to improve the plan based on retrieved Q&A.")
    Output: str = Field(..., description="The expected output format for this step after improvements. You must not change the files requested by the plan, but you must suggest extra details to improve the output format selected.")

class PlanResponse(BaseModel):
    Steps: List[Step] = Field(..., description="A list of steps for the Plan.")
    Number_of_Steps: int = Field(..., description="The total number of steps in the plan.")

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

#============ Agent Notes ==========================
# The following agents in this file are agents that are meant to enhance the plan quality. They have
# been developed in a separate file so that the reviewer in Planner_Reviewer.py can call a function
# to instantiate these agents when needed.

# Before the conversation passes back to the planner, these agents will be called (Potentialy in parallel by having different classes)
# and the planner should have access to the information generated to enhance the plan quality, as well as the feedback given by the original
# reviewer agent.

class Reality_Checker_Agent(Agent_Base):
    pass
class Structure_Advisor_Agent(Agent_Base):
    pass
class Focus_Area_Assessor_Agent(Agent_Base):
    pass
class RAG_Questioner_Agent(Agent_Base):
    pass
class RAG_Interpreter_Agent(Agent_Base):
    pass
class Review_Compiler_Agent(Agent_Base):
    pass
class Output_Instructor_Agent(Agent_Base):
    pass
class Output_Structure_Advisor_Agent(Agent_Base):
    pass

# 1-------------------
class Reality_Chat():
    Type: ""
    Iteration: int
    def __init__(self, context_variables: ContextVariables, Analysis_Type: str, Max_Rounds: int, Iteration: int):
        self.context_variables=context_variables
        self.Analysis_Type=Analysis_Type
        Reality_Chat.Type=Analysis_Type
        self.Max_Rounds=Max_Rounds
        Reality_Chat.Iteration=Iteration
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()

        if self.Analysis_Type=="Usability":
            self.plan_key="{Usability_Plan}"
        elif self.Analysis_Type=="HPLC":
            self.plan_key="{HPLC_Analysis_Plan}"
        elif self.Analysis_Type=="MS":
            self.plan_key="{Mass_Spectrometry_Plan}"
        else:
            raise ValueError(f"Invalid Analysis_Type: {self.Analysis_Type}. Must be one of 'Usability', 'HPLC', 'MS'.")
        
        self.Reality_Checker_System_Message="""
        You are a reality checker that ensures that the data analysis suggested is well algned with the actual data available.
        """
        self.Reality_Checker_Update_System_Message=f"""
        -------- ROLE -----------------
        You are an expert in checking that data analysis plans are well aligned with the data available. Your main focus is on the variables tat have been suggested for each analysis step. These variables must be suitable for the analysis type suggested.
        You have been provided with information about the dataset from an exploratory data analysis (EDA) phase as well as metadata containing all variables names present in each data file.
        Each analysis step is only allowed to use one data file. Variables MUST NEVER be mixed from other data files within a single analysis step.
        You have been provided with the focus area statement which defines the main aims for the data analysis. This should help to guide your reviews.
        You have been provided with the current plan that is to be reviewed.
        The plan contains information about the analysis type, data file, variable names and the output format that are to be used. You must ensure that the intended outputs are achievable with the variables suggested. You are encouraged to try to predict what the output will look like to help understand if the analysis is valid.
        It is possible to improve plans by suggesting that different variables in the data files might be more suitable for the suggested analysis. This doesn't mean changing all variables, it might be just one or two that are misaligned.
        You are only allowed to use varables form the dataset suggested in the plan. You must NEVER suggest using variables from other data files that are not relevant to the analysis type.
        Example 1: (Illustrative not Exhaustive): If a visualsation is suggested for time series data to identify trends over time, then the x axis should be a time variable. A time varibale does not have to explicitly say "time", but it should represent the progression of obtaining data.
        Example 2: (Illustrative not Exhaustive): If a data analysis suggests identifying noise levels of a signal, then the context of the data may change the analysis. If data is expected to contain sudden peaks, then noise analysis should be applied to the baseline and not the peaks. This avoids misinterpretation of noise levels. When peaks are expected, they must never be assumed to be noise or outliers. However, small fluctuations in the baseline can be considered noise even if they may look like peaks. A good data analysis will distingush between the two cases.
        You must never falsify any information about the dataset. All comments must be grounded in the information provided.
        You NEVER attempt to solve the problem yourself. You NEVER attempt to change the plan being carried out. Your SOLE purpose is to identify any misalignments between the data analysis suggested and the variables that the analysis step relies upon.
        You will recieve knowledge from a dataset strcuture expert that is crucial to effectve use of variables. You must incorporate this knowledge into your reviews to ensure that the plan is well aligned with the structure of the variables suggested and the dataset.
        Your role is to provide a score to support your feedback to help frame the importance of your suggestions. If the plan is not suitable, you must be critical and this must be reflected in your scoring. You must not give a hgh score (>6) if there are any issues with the plan. See the scoring rules for more information.

        ----------- Task -----------------
        - You must call the Reality_Report function to provide your feedback.
        - You must use the information provded to you about the dataset to inform your reviews of the current analysis plan.
        - You must ensure that the analysis suggested is achievable with the data available.
        - Your feedback must be clear and actionable.
        -----------------------------------
        ------- Output Format -------------
        Your output must follow the following format:
        ** Step Number and Analysis Type **
        <Your Feedback Here>

        [Repeat for each step in the plan]
        - Score: The score you try to provide a numeric score out of 10 to accompany your feedback. This score must be for the overall plan, not individual steps. 
                 10 is a perfect score and means the plan is very good. 1 is a very poor score indicating an unusable plan with major issues. You do not have to provide integer scores, but it must be between 1 and 10.
        -----------------------------------
        --------- Scoring Rules -----------
        - Score=10: The plan is perfectly aligned with the data available and the choice of variables is optimal for the analysis suggested and the data available.
        - Score=7-9: The plan is mostly aligned with the data available, but there are minor issues with the choice of variables that could be improved. The analysis suggested is achievable with the data available.
        - Score=4-6: The plan has some misalignments with the data available, and there are several issues with the choice of variables that need to be addressed. The analysis suggested may be partially achievable with the data available, but significant improvements are needed.
        - Score=1-3: The plan is poorly aligned with the data available, and there are major issues with the choice of variables that render the analysis suggested unachievable with the data available.
        -----------------------------------
        ------ Context -------------
        ** Current Analysis Plan **
        {self.plan_key}
        *************
        ** Focus Area Statement **
        {{Focus_Area_{self.Analysis_Type}}}
        *************
        ** Dataset Structure Advice for Improving your Reviews **
        {{Data_Structure_Advice}}
        *************
        ** Dataset Information **
        {{Univariate_EDA_Report}}
        *************

        ** Dataset Metadata **
        {{metadata}}
        """
        self.Structure_Advisor_System_Message="""
        You are a dataset structure advisor.
        """
        self.Structure_Advisor_Update_System_Message=f"""
        --------- ROLE -------------------
        You are an expert in understanding dataset structures and how they impact data analysis plans.
        You have been provided with information about the dataset from an exploratory data analysis (EDA) and interpretation phase. You have also been provided with metadata containing all variables names present in each data file.
        You have been provided with the current plan for data data analysis. You must provide a commentary of the areas of the plan that should handle certain variables carefully depending on the strcuture of the dataset.
        **For Example** (Illustrative not Exhaustive): If the datset contains data from many experments with a time series structure then the plan should contain explicit instructions to perform the suggested data analysis on each experimental run independently. The data analysis should use an identifier variable to group the data by batch or run to avoid mixing data from different experiments.
        You must never falsify any information about the dataset. All comments must be grounded in the information provided.
        The variables provided in a gven data analysis step are only allowed to be from the single data file suggested for that plan step. Variables MUST NEVER be mixed from other data files within a single analysis step.
        You have been provided with the current report from the Reality Checker agent. Your commentary should ensure that the reviews provided take the structure of the dataset into account.
        You must never falsify any information about the dataset. All comments must be grounded in the information provided.
        You NEVER attempt to solve the problem yourself. You NEVER attempt to change the plan being carried out. Your SOLE purpose is to identify any misalignments between the data analysis suggested and the variables that the analysis step relies upon.
        ---------------------------------
        ----------- Task -----------------
        - You MUST provide your advice using the Structure_Advice function.
        - You must use all of the information provided to inform your advice.
        - You must ensure that the analysis suggested is achievable with the data available.
        -----------------------------------
        ------- Output Format -------------
        Your output must follow the following format:
        ** Step Number and Analysis Type **
        <Your Feedback Here>

        [Repeat for each step in the plan]
        -----------------------------------
        -------- Context -------------------
        ** Current Analysis Plan **
        {self.plan_key}
        **********
        ** Dataset Information **
        {{Univariate_EDA_Report}}
        ***********
        ** Dataset Metadata **
        {{metadata}}
        ***********
        ** Current Review of the Plan (From Reality Checker) **
        {{Data_Reality_Report}}
        """
        
        #-----------------
        self.Reality_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.Structure_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Reality_Checker=Reality_Checker_Agent(
            name="Reality_Checker",
            llm_config=self.Reality_LLM_Config,
            system_message=self.Reality_Checker_System_Message,
            Update_System_Message=self.Reality_Checker_Update_System_Message)
        self.Structure_Advisor=Structure_Advisor_Agent(
            name="Structure_Advisor",
            llm_config=self.Structure_LLM_Config,
            system_message=self.Structure_Advisor_System_Message,
            Update_System_Message=self.Structure_Advisor_Update_System_Message)
        
        # Handoffs ---------------
        self.Reality_Checker.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Structure_Advisor.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Reality_Feedback} == True")
                )
            )
        )

        self.Reality_Checker.agent.handoffs.set_after_work(AgentTarget(self.Reality_Checker.agent))

        self.Structure_Advisor.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Reality_Checker.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Structure_Advice_Provided} == True")
                )
            )
        )
        self.Structure_Advisor.agent.handoffs.set_after_work(AgentTarget(self.Structure_Advisor.agent))

        # Functions-------
        register_function(
            Reality_Report,
            caller=self.Reality_Checker.agent,
            executor=self.Reality_Checker.agent,
            name="Reality_Report",
            description="Generates a concise focus area statement for data analysis based on the provided context."
            )
        register_function(
            Structure_Advice,
            caller=self.Structure_Advisor.agent,
            executor=self.Structure_Advisor.agent,
            name="Structure_Advice",
            description="Provides advice on the structure of the data analysis plan based on the provided context."
            )
        
    def run_Conversation(self):
        # Reset Context Variables:
        self.context_variables["Reality_Feedback"]=False
        self.context_variables["Structure_Advice_Provided"]=False
        self.context_variables["Data_Reality_Report"]=""
        self.context_variables["Data_Structure_Advice"]=""
        #-------------------------
        agents=[self.Reality_Checker.agent, self.Structure_Advisor.agent]
        pattern=DefaultPattern(
        initial_agent=agents[0],
        agents=agents,
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,)

        result, ctx, _ = initiate_group_chat(
        pattern=pattern,
        messages="Using the information available, suggest feedback for each step in the plan to ensure that the data analysis suggested is well aligned with the actual data available.",
        max_rounds=self.Max_Rounds)
    
        return result, ctx

def Reality_Report(RR: Annotated[str,"Suggest feedback for each step in the plan to ensure that the data analysis suggested is well aligned with the actual data available."],
Score: Annotated[float,"You must provide a numeric score between 1 and 10 (1=Poor, 10=Perfect) to accompany your feedback." ],
context_variables: ContextVariables) -> ReplyResult:

    context_variables[f"Data_Reality_Report"]=RR
    context_variables["Reality_Feedback"]=True

    context_variables[f"{Reality_Chat.Type}_Plan_Scoring_{Reality_Chat.Iteration}"][0]=Score

    #----- Reset Context Variables
    context_variables["Structure_Advice_Provided"]= False

    return ReplyResult(
        message=f"Reality Checks Complete.",
        context_variables=context_variables,
    )

def Structure_Advice(SA: Annotated[str,"Suggest advice for how to improve the plan reviews based on the structure of the data."], context_variables: ContextVariables) -> ReplyResult:

    context_variables[f"Data_Structure_Advice"]=SA
    context_variables["Structure_Advice_Provided"]= True
    #----- Reset Context Variables
    context_variables["Reality_Feedback"]= False
    return ReplyResult(
        message=f"Reality Checks Complete.",
        context_variables=context_variables,
    )
#--------------------------------------------------------------------------------------

# 2--------------------
class Focus_Area_Chat():
    Type: ""
    Iteration: int
    def __init__(self, context_variables: ContextVariables, Analysis_Type: str, Max_Rounds: int, Iteration: int):
        self.context_variables=context_variables
        self.Analysis_Type=Analysis_Type
        Focus_Area_Chat.Type=Analysis_Type
        self.Max_Rounds=Max_Rounds
        Focus_Area_Chat.Iteration=Iteration
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()

        if self.Analysis_Type=="Usability":
            self.plan_key="{Usability_Plan}"
            self.FA_key="{Focus_Area_Usability}"
        elif self.Analysis_Type=="HPLC":
            self.plan_key="{HPLC_Analysis_Plan}"
            self.FA_key="{Focus_Area_HPLC}"
        elif self.Analysis_Type=="MS":
            self.plan_key="{Mass_Spectrometry_Plan}"
            self.FA_key="{Focus_Area_MS}"
        else:
            raise ValueError(f"Invalid Analysis_Type: {self.Analysis_Type}. Must be one of 'Usability', 'HPLC', 'MS'.")
        
        self.Focus_Area_Assessor_System_Message="""
        You are the focus area assessment agent.
        """
        self.Focus_Area_Assessor_Update_System_Message=f"""
        -------- ROLE -----------------
        You are a Focus Area Assessor that ensures that data analysis plans are well aligned with the main focus area idenitified.
        The focus area that you have been provided with highlights what data analysis means in the context of the datasets available.
        You have been provided with the current plan that needs to be reviewed based on how well it aligns with the focus area and the main objectves of the data analysis.
        The focus area contains suggestions about suitable data analysis techniques, but this is not exhaustive. The purpose is to indicate that arbitrary summary statistics such as mean, median, etc are not sufficient at this level.
        You must never falsify any information about the dataset. All comments must be grounded in the information provided.
        You NEVER attempt to solve the problem yourself. You NEVER attempt to change the plan being carried out. Your SOLE purpose is to identify any misalignments between the data analysis suggested and the focus area provided.
        Your role is to provide a score to support your feedback to help frame the importance of your suggestions. If the plan is not suitable, you must be critical and this must be reflected in your scoring. You must not give a hgh score (>6) if there are any issues with the plan. See the scoring rules for more information.
        -------------------------------
        ------------ Focus Area -----------------
        {self.FA_key}
        -----------------------------------------
        ----------- Task -----------------
        - You must call the FA_Review function to provide your feedback.
        - You must use the information provded to you about the focus area to inform your reviews of the current analysis plan.
        - Your feedback must be clear and actionable.
        -----------------------------------
        ------- Output Format -------------
        Your output must follow the following format:
        ** Step Number and Analysis Type **
        <Your Feedback Here>

        [Repeat for each step in the plan]
        - Score: The score you try to provide a numeric score out of 10 to accompany your feedback and indicate how well the plan follows the focus area. This score must be for the overall plan, not individual steps. 
                 10 is a perfect score and means the plan is very good. 1 is a very poor score indicating an unusable plan with major issues. You do not have to provide integer scores, but it must be between 1 and 10.
        -----------------------------------
        --------- Scoring Rules -----------
        - Score=10: The plan is perfectly aligned with the focus area and the main goals outlined.
        - Score=7-9: The plan is mostly aligned with the focus area and the main goals outlined. There are minor issues that could be addressed to improve alignment.
        - Score=4-6: The plan has some misalignments with the focus area and main goals. There are several issues that should be addressed. The plan may have partial alignment but is far from optimal.
        - Score=1-3: The plan is poorly aligned with the focus area and main goals. There are critical issues that render the plan unsuitable in ths context.
        -----------------------------------
        ------ Context -------------
        ** Current Analysis Plan **
        {self.plan_key}
        """
        #-----------------
        self.FA_Assessor_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Focus_Area_Assessor=Focus_Area_Assessor_Agent(
            name="Focus_Area_Assessor",
            llm_config=self.FA_Assessor_LLM_Config,
            system_message=self.Focus_Area_Assessor_System_Message,
            Update_System_Message=self.Focus_Area_Assessor_Update_System_Message)
        
        # Handoffs ---------------
        self.Focus_Area_Assessor.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${FA_Feedback} == True")
                )
            )
        )

        self.Focus_Area_Assessor.agent.handoffs.set_after_work(AgentTarget(self.Focus_Area_Assessor.agent))

        # Functions-------
        register_function(
            FA_Review,
            caller=self.Focus_Area_Assessor.agent,
            executor=self.Focus_Area_Assessor.agent,
            name="FA_Review",
            description="Suggest feedback on how well the analysis plan aligns with the focus area provided."
            )
        
    def run_Conversation(self):
        # Reset Context Variables:
        self.context_variables["FA_Feedback"]=False
        self.context_variables["FA_Report"]=""
        #------------------------------------------
        agents=[self.Focus_Area_Assessor.agent]
        pattern=DefaultPattern(
        initial_agent=agents[0],
        agents=agents,
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,)

        result, ctx, _ = initiate_group_chat(
        pattern=pattern,
        messages="Using the information available, suggest feedback for each step in the plan to ensure that the data analysis suggested is well aligned with the focus area provided.",
        max_rounds=self.Max_Rounds)
    
        return result, ctx

def FA_Review(FA: Annotated[str,"Suggest feedback for each step in the plan to ensure that the data analysis suggested is well aligned with the focus area provided."],
Score: Annotated[float,"You must provide a numeric score between 1 and 10 (1=Poor, 10=Perfect) to accompany your feedback." ],
context_variables: ContextVariables) -> ReplyResult:

    context_variables[f"FA_Report"]=FA
    context_variables[f"{Focus_Area_Chat.Type}_Plan_Scoring_{Focus_Area_Chat.Iteration}"][1]=Score

    context_variables["FA_Feedback"]=True
    return ReplyResult(
        message=f"Focus Area Alignment Checks Complete.",
        context_variables=context_variables,
    )
# 3-------------------
class RAG_System():
    Type: ""
    def __init__(self, context_variables: ContextVariables, Analysis_Type: str, Max_Rounds: int):
        self.context_variables=context_variables
        self.Analysis_Type=Analysis_Type
        RAG_System.Type=Analysis_Type
        self.Max_Rounds=Max_Rounds
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()

        if self.Analysis_Type=="Usability":
            self.plan_key="{Usability_Plan}"
        elif self.Analysis_Type=="HPLC":
            self.plan_key="{HPLC_Analysis_Plan}"
        elif self.Analysis_Type=="MS":
            self.plan_key="{Mass_Spectrometry_Plan}"
        else:
            raise ValueError(f"Invalid Analysis_Type: {self.Analysis_Type}. Must be one of 'Usability', 'HPLC', 'MS'.")
        
        self.RAG_Questioner_System_Message="""
        You are a research agent.
        """
        self.RAG_Questioner_Update_System_Message=f"""
        -------- ROLE ------------------
        You are an expert research agent that specialises in asking the right questions to gather domain specfic information that is crucial for effective data analysis.
        You have been provided with the current plan that needs to be reinforced with domain specific knowledge. You MUST ask relevant questions that will help you to gather crucial information required to support the plan.
        You have been provided with the focus area statement which defines the main aims of the data analysis that is intended to be addressed by the plan.
        The questions that you ask can relate to the use of specific variables, understanding more about analysis types and common approaches in the domain area, relevant visualisations or outputs that are commonly used, and any other information that will enhance the plan.
        You must never falsify any information about the dataset or the plan. All comments must be grounded in the information provided.
        -------------------------------
        ----------- Task -----------------
        - You must call the RAG_Questions function with your domain specific questions.
        - Your questions must be clear and concise. There should be no ambiguity in what you are asking. You must ensure that each question explores a different area of the plan or focus area to ensure adequate coverage.
        - You wll be able to ask 6 questions in total.
        -----------------------------------
        ------------- Context -------------
        ** Current Plan for Data Analysis **
        {self.plan_key}
        *************
        ** Focus Area Statement **
        {{Focus_Area_{self.Analysis_Type}}}
        """
        self.RAG_Interpreter_System_Message="""
        You are a research interpreter agent.
        """
        self.RAG_Interpreter_Updated_System_Message=f"""
        --------- ROLE -------------------
        You are an expert research interpreter that specalises in interpreting domain specific infromation to enhance data analysis plans.
        You have been provided with the current plan for data analysis as well as the focus area statement that defines the main aims of the data analysis.
        You have been provided with domain specific questions and answers that have been retrieved to provide you with factual information that you must use to provide instructions on how to improve the plan.
        You must clearly state which aspects of the plan are along the correctly lines and provide domain specific knowlegde to support this. You must also identify if any aspects of the plan are misaligned and not sutable for the data analysis intended.
        You must never falsify any information about the dataset or the plan. All comments must be grounded in the information provided. If the information retrieved is not relevant to certan parts of the plan, then you must not comment on these sections.
        ----------------------------------
        ----------- Task -----------------
        - You must call the RAG_Interpretation function to provide information about the domain and instructions for how to improve the plan.
        - You must clearly state which step number of the plan you are referring to when providing your instructions and domain knowledge.
        ----------------------------------
        ------- Output Format -------------
        Your output must follow the following format:
        ** Step Number and Analysis Type **
        <Your Feedback Here>
        [Repeat for each step in the plan that you have information about]
        -----------------------------------
        -------- Context -------------------
        ** Current Analysis Plan **
        {self.plan_key}
        **********
        ** Focus Area Statement **
        {{Focus_Area_{self.Analysis_Type}}}
        ***********
        ** Retrieved Domain Specific Q&A **
        {{RAG_QA}}
        """
        
        #-----------------
        self.RAG_QA_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.RAG_Interpreter=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.RAG_Questioner=RAG_Questioner_Agent(
            name="RAG_Questioner",
            llm_config=self.RAG_QA_LLM_Config,
            system_message=self.RAG_Questioner_System_Message,
            Update_System_Message=self.RAG_Questioner_Update_System_Message)
        self.RAG_Interpreter=RAG_Interpreter_Agent(
            name="RAG_Interpreter",
            llm_config=self.RAG_Interpreter,
            system_message=self.RAG_Interpreter_System_Message,
            Update_System_Message=self.RAG_Interpreter_Updated_System_Message)
        
        # Handoffs ---------------
        self.RAG_Questioner.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.RAG_Interpreter.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${QA_Available} == True")
                )
            )
        )

        self.RAG_Questioner.agent.handoffs.set_after_work(AgentTarget(self.RAG_Questioner.agent))

        self.RAG_Interpreter.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${RAG_Interpretation_Available} == True")
                )
            )
        )
        self.RAG_Interpreter.agent.handoffs.set_after_work(AgentTarget(self.RAG_Interpreter.agent))
        # Functions-------
        register_function(
            RAG_Questions,
            caller=self.RAG_Questioner.agent,
            executor=self.RAG_Questioner.agent,
            name="RAG_Questions",
            description="Generate 6 domain specific questions to gather crucial information required to support the data analysis plan."
            )
        register_function(
            RAG_Interpretation,
            caller=self.RAG_Interpreter.agent,
            executor=self.RAG_Interpreter.agent,
            name="RAG_Interpretation",
            description="Provide domain specific knowledge and instructions for how to improve the data analysis plan based on the retrieved Q&A."
            )
        
    def run_Conversation(self):
        # Reset Context Variables:
        self.context_variables["Reality_Feedback"]=False
        self.context_variables["Data_Reality_Report"]=""
        #-------------------------
        agents=[self.RAG_Questioner.agent, self.RAG_Interpreter.agent]
        pattern=DefaultPattern(
        initial_agent=agents[0],
        agents=agents,
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,)

        result, ctx, _ = initiate_group_chat(
        pattern=pattern,
        messages="Using the information available, ask 6 domain specific questions to gather crucial information required to support the data analysis plan.",
        max_rounds=self.Max_Rounds)
    
        return result, ctx

def RAG_Questions(Q1: Annotated[str,"Question for research database"],Q2: Annotated[str,"Question for research database"], Q3: Annotated[str,"Question for research database"], Q4: Annotated[str,"Question for research database"], Q5: Annotated[str,"Question for research database"], Q6: Annotated[str,"Question for research database"], context_variables: ContextVariables) -> ReplyResult:
    R1 = RAG_Tool(Q1)
    R2 = RAG_Tool(Q2)
    R3 = RAG_Tool(Q3)
    R4 = RAG_Tool(Q4)
    R5 = RAG_Tool(Q5)
    R6 = RAG_Tool(Q6)
    context_variables["RAG_QA"] = f""" Query: {Q1} | Result:{R1} \n Query: {Q2} | Result:{R2} \n Query: {Q3} | Result:{R3} \n Query: {Q4} | Result:{R4} \n Query: {Q5} | Result:{R5} \n Query: {Q6} | Result:{R6} """
    context_variables["QA_Available"]=True

    #----- Reset Context Variables
    context_variables["RAG_Interpretation_Available"]= False

    return ReplyResult(
        message=f"RAG Questions Completed.",
        context_variables=context_variables,
    )

def RAG_Interpretation(Interpretation: Annotated[str,"Interpretation of RAG results"], context_variables: ContextVariables) -> ReplyResult:

    context_variables["RAG_Interpretation"]=Interpretation
    context_variables["RAG_Interpretation_Available"]= True
    #----- Reset Context Variables
    context_variables["QA_Available"]= False
    return ReplyResult(
        message=f"RAG Interpretation Complete.",
        context_variables=context_variables,
    )
# 4--------------------
class Output_Instruction():
    Type: ""
    Iteration: int
    def __init__(self, context_variables: ContextVariables, Analysis_Type: str, Max_Rounds: int, Iteration: int):
        self.context_variables=context_variables
        self.Analysis_Type=Analysis_Type
        Output_Instruction.Type=Analysis_Type
        self.Max_Rounds=Max_Rounds
        Output_Instruction.Iteration=Iteration
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()

        if self.Analysis_Type=="Usability":
            self.plan_key="{Usability_Plan}"
            self.FA_key="{Focus_Area_Usability}"
        elif self.Analysis_Type=="HPLC":
            self.plan_key="{HPLC_Analysis_Plan}"
            self.FA_key="{Focus_Area_HPLC}"
        elif self.Analysis_Type=="MS":
            self.plan_key="{Mass_Spectrometry_Plan}"
            self.FA_key="{Focus_Area_MS}"
        else:
            raise ValueError(f"Invalid Analysis_Type: {self.Analysis_Type}. Must be one of 'Usability', 'HPLC', 'MS'.")
        
        self.Output_Instructor_System_Message="""
        You are an output instruction agent.
        """
        self.Output_Instructor_Update_System_Message=f"""
        You are an expert in providing clear instructions for the output formats required for data analysis plans.
        You have been provided with the current analysis plan which includes the expected output formats for each data analysis step.
        You MUST respect the use of the markdown file usage and the image files that have been specified. Your role is to suggest additional details that enable these output formats to be generated effectively.
        Markdown files should be used for text based output and must be called "{Output_Instruction.Type}.md". No step should ever overwrite this file. Appending to ths fle is the only allowed operation.
        **For Example** (Illustrative not Exhaustive): If there is an instruction to plot visualsations, you might suggest plotting several details on the same plot to enhance interpretablity and reduce redundant output.
        **For Example** (Illustrative not Exhaustive): If the plan is to plot a graph, then there should be clear instruction to be careful of joining data points with lines to prevent a messy visualisation. Points that should be connected should be clearly identified.
        **For Example** (Illustrative not Exhaustive): If text based output is advcies and saved to a markdown (.md) file, then you might suggest providing section headers and other formatting details that would ensure that a reader can easly intepret the results.
        You MUST consider the use of visualisations and ensure that only a small number of visualsations are generated. It is essential that the output does not overload the user with too many images. Note that the use of subplots is encouraged to reduce the number of image files generated.
        You must never falsify any information about the dataset. All comments must be grounded in the information provided.
        You will have information provided to you form a data structure expert that is crucial for effective outputs and reducing redundancy. You must adjust your review of the output formats accordingly to ensure relevant suggestions to improve the plan.
        Your role is to provide a score to support your feedback to help frame the importance of your suggestions. If the plan is not suitable, you must be critical and this must be reflected in your scoring. You must not give a hgh score (>6) if there are any issues with the plan. See the scoring rules for more information.
        -------------------------------
        -----------Task-----------------
        - You must call the Output_Instructions function to provide your feedback.
        - You must use all of the information provided to generate clear instructions for improving the output formats specified in the plan.
        - Your instructions must be clear and actionable.
        -------------------------------
        ------- Output Format -------------
        Your output must follow the following format:
        ** Step Number and Analysis Type **
        <Your Feedback Here>
        [Repeat for each step in the plan]

        - Score: The score you try to provide a numeric score out of 10 to accompany your feedback and indicate how well the plan details the output requirements. This score must be for the overall plan, not individual steps. 
                 10 is a perfect score and means the plan is very good. 1 is a very poor score indicating an unusable plan with major issues. You do not have to provide integer scores, but t must be between 1 and 10.
        -----------------------------------
        --------- Scoring Rules -----------
        - Score=10: The output prompts are clear and perfectly align with the data analysis plan. The output formats would enable effective communication of the results.
        - Score=7-9: The output prompts are relatively clear and mostly align with the data analysis plan. There are minor issues that could be addressed to improve clarity and alignment from either a visual or text based output perspective.
        - Score=4-6: The output prompts have some misalignments with the data analysis plan. There are several issues that should be addressed. The output formats may have partial alignment but are far from optimal. The results could be hard to read or interpret from either a visual or text based output perspective.
        - Score=1-3: The output prompts are poor. They make little sense for the analysis type and would not enable effective communication of the results. There are critical issues that render the output prompts unsuitable in this context. The output is likely to be confusing or misleading from either a visual or text based output perspective.
        -----------------------------------
        ------- Context -------------
        ** Current Analysis Plan **
        {self.plan_key}
        *************
        ** Dataset Structure Advice for Improving Outputs **
        {{Output_Data_Structure_Advice}}
        *************
        """
        self.Output_Data_Structure_Advisor_System_Message="""
        You are a dataset structure advisor for output instructions.
        """
        self.Output_Data_Structure_Advisor_Update_System_Message=f"""
        --------- ROLE -------------------
        You are an expert in understandng dataset structures and understanding how they impact the output formats required for data analysis plans.
        You have been provided with information about the dataset from an exploratory data analysis (EDA) and interpretation phase.
        You have been provided with the current plan for analysis and the focus area statement that defines the main aims of the data analysis.
        You work closely with the output instruction agent and make sure that their suggestions for mproving the output are feasible given the structure of the dataset.
        **For Example** (Illustrative not Exhaustive): If the dataset contains data from many experiments with a time series structure then output instructions should ensure that results are clearly grouped by experimental run. This avoids mixing data from different experiments in the output visualisations or summaries. This may mean suggesting legends for graphs or clear section headers in text based outputs.
        You must never falsify any information about the dataset. All comments must be grounded in the information provided.
        ---------------------------------
        ----------- Task -----------------
        - You MUST provide your advice using the Output_Structure_Advice function.
        - You must use all of the information provided to inform your advice.
        -----------------------------------
        ------- Output Format -------------
        Your output must follow the following format:
        ** Step Number and Analysis Type **
        <Your Feedback Here>

        [Repeat for each step in the plan]
        -----------------------------------
        -------- Context -------------------
        ** Current Analysis Plan **
        {self.plan_key}
        **********
        ** Focus Area Statement **
        {{Focus_Area_{self.Analysis_Type}}}
        ***********
        ** Dataset Information **
        {{Univariate_EDA_Report}}
        ***********
        ** Dataset Metadata **
        {{metadata}}
        ***********
        ** Current Output Instructions (From Output Instruction Agent) **
        {{Output_Instruction_Review}}
        ***********
        """
        #-----------------
        self.OP_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.OP_Structure_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Output_Instructor=Output_Instructor_Agent(
            name="Output_Instructor",
            llm_config=self.OP_LLM_Config,
            system_message=self.Output_Instructor_System_Message,
            Update_System_Message=self.Output_Instructor_Update_System_Message)
        self.Output_Structure_Advisor=Output_Structure_Advisor_Agent(
            name="Output_Structure_Advisor",
            llm_config=self.OP_Structure_LLM_Config,
            system_message=self.Output_Data_Structure_Advisor_System_Message,
            Update_System_Message=self.Output_Data_Structure_Advisor_Update_System_Message)
        
        # Handoffs ---------------
        self.Output_Instructor.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Output_Structure_Advisor.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${OP_Instruction_Available} == True")
                )
            )
        )

        self.Output_Instructor.agent.handoffs.set_after_work(AgentTarget(self.Output_Instructor.agent))

        self.Output_Structure_Advisor.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Output_Instructor.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${OP_Structure_Advice_Available} == True")
                )
            )
        )
        self.Output_Structure_Advisor.agent.handoffs.set_after_work(AgentTarget(self.Output_Structure_Advisor.agent))
        # Functions-------
        register_function(
            Output_Instructions,
            caller=self.Output_Instructor.agent,
            executor=self.Output_Instructor.agent,
            name="Output_Instructions",
            description="Provide clear instructions for improving the output formats specified in the data analysis plan."
            )
        register_function(
            Output_Structure_Advice,
            caller=self.Output_Structure_Advisor.agent,
            executor=self.Output_Structure_Advisor.agent,
            name="Output_Structure_Advice",
            description="Provide advice on how the structure of the dataset impacts the output formats required for the data analysis plan."
        )
        
    def run_Conversation(self):
        # Reset Context Variables:
        self.context_variables["OP_Instruction_Available"]=False
        self.context_variables["OP_Structure_Advice_Available"]=False
        self.context_variables["Output_Instruction_Review"]=""
        self.context_variables["Output_Data_Structure_Advice"]=""
        #------------------------------------------
        agents=[self.Output_Instructor.agent, self.Output_Structure_Advisor.agent]
        pattern=DefaultPattern(
        initial_agent=agents[0],
        agents=agents,
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,)

        result, ctx, _ = initiate_group_chat(
        pattern=pattern,
        messages="Using the information available, suggest clear instructions for improving the output formats specified in the data analysis plan.",
        max_rounds=self.Max_Rounds)
    
        return result, ctx

def Output_Instructions(OI: Annotated[str,"Provide clear instructions for improving the output formats specified in the data analysis plan."],
Score: Annotated[float,"You must provide a numeric score between 1 and 10 (1=Poor, 10=Perfect) to accompany your feedback." ],
context_variables: ContextVariables) -> ReplyResult:
    context_variables["Output_Instruction_Review"]=OI
    context_variables["OP_Instruction_Available"]=True

    context_variables[f"{Output_Instruction.Type}_Plan_Scoring_{Output_Instruction.Iteration}"][2]=Score

    #----- Reset Context Variables
    context_variables["OP_Structure_Advice_Available"]= False
    return ReplyResult(
        message=f"Output Instructions Provided.",
        context_variables=context_variables,
    )
def Output_Structure_Advice(OSA: Annotated[str,"Provide advice on how the structure of the dataset impacts the output formats required for the data analysis plan."], context_variables: ContextVariables) -> ReplyResult:
    context_variables["Output_Data_Structure_Advice"]=OSA
    context_variables["OP_Structure_Advice_Available"]=True

    #----- Reset Context Variables
    context_variables["OP_Instruction_Available"]= False
    return ReplyResult(
        message=f"Output Structure Advice Provided.",
        context_variables=context_variables,
    )

# 5--------------------
class Review_Compilation():
    Type: ""
    Iteration: int
    def __init__(self, context_variables: ContextVariables, Analysis_Type: str, Max_Rounds: int, Iteration:int):
        self.context_variables=context_variables
        self.Analysis_Type=Analysis_Type
        Review_Compilation.Type=Analysis_Type
        self.Max_Rounds=Max_Rounds
        Review_Compilation.Iteration=Iteration
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()

        if self.Analysis_Type=="Usability":
            self.plan_key="{Usability_Plan}"
            self.FA_key="{Focus_Area_Usability}"
        elif self.Analysis_Type=="HPLC":
            self.plan_key="{HPLC_Analysis_Plan}"
            self.FA_key="{Focus_Area_HPLC}"
        elif self.Analysis_Type=="MS":
            self.plan_key="{Mass_Spectrometry_Plan}"
            self.FA_key="{Focus_Area_MS}"
        else:
            raise ValueError(f"Invalid Analysis_Type: {self.Analysis_Type}. Must be one of 'Usability', 'HPLC', 'MS'.")
        
        self.Review_Compiler_System_Message="""
        You are the review compilation agent.
        """
        self.Review_Compiler_Update_System_Message=f"""
        -------- ROLE -----------------
        You are a review compilation agent that combines all feedback provided by individual specalists into detailed instructions for improving the data analysis plan.
        You have been provided with the feedback from all of the specialist reviewers that you must use to generate clear instructions for improving the plan.
        You have been provided with the focus area statement for the plan which defines the main aims of the data analysis.
        You have also been provided with the plan that has been reviewed by all of the specialist agents. This is the plan that you must provide instructions to improve.
        You must never falsify any information about the dataset. All comments must be grounded in the information provided.
        Your role is to provide a score to support your feedback to help frame the importance of your suggestions. If the plan is not suitable, you must be critical and this must be reflected in your scoring. You must not give a hgh score (>6) if there are any issues with the plan. See the scoring rules for more information.
        -------------------------------
        --------- Task -----------------
        - You must call the Review function to provide your compiled feedback.
        - You must use all of the feedback provided to generate clear and actionable instructions for improving the plan.
        - You must provide explanations for why the changes are necessary to ensure that the planner agent understands how to improve the plan effectively.
        -----------------------------------
        ------- Output Format -------------
        Your output must follow the following format (For each step in the plan):
        - Step Number: Important to clearly indicate which step you are referring to.
           * Variables_Feedback: You must provide feedback on the suitablility of variables suggested for this analysis step. You must include any relevant information about the dataset strcuture that impacts the choice of variables and how they are used.
           * Scope: You must use information about the main aims of the data analysis from teh focus area and provide instructions on how to ensure that the analysis step is well aligned with these aims.
           * Research: You must provide any domain specific knowledge that is crucial for effectively carrying out this analysis step. This must be grounded in the information provided by the research agent.
           * Output_Instructions: You must provide clear instructions for what is expected as the output format from the analysis step. This must be detailed and unambiguous to ensure that the planner agent can effectively implement the changes.
        You **MUST** repeat this exact format for each step in the plan that requires changes.
        For the overall plan, you must also provide the following:
        - Score: Based on the overall feedback you have been provided with, you must provide a numeric score out of 10 to accompany your feedback to indicate the overall quality of the plan. This score must be for the overall plan, not individual steps. 
                 10 is a perfect score and means the plan is very good. 1 is a very poor score indicating an unusable plan with major issues. You do not have to provide integer scores, but it must be between 1 and 10.
        --------------------------------------
        --------- Scoring Rules -----------
        - Score=10: The feedback indicates a high quality plan with no improvements suggested and perfect alignement with the domain knowledge.
        - Score=7-9: The feedback indicates a good quality plan with minor improvements suggested and mostly good alignement with the domain knowledge.
        - Score=4-6: The feedback indicates a plan with several issues that need to be addressed. There are multiple suggestions for improvement and partial misalignments with the domain knowledge. The issues are not minor and need to be addressed to ensure the plan is effective.
        - Score=1-3: The feedback indicates a poor quality plan with major issues that render it unusable. There are critical misalignments with the domain knowledge and the suggested improvements are substantial. The plan requires significant revisions to be effective.
        -----------------------------------
        ------ Context -------------
        ** Current Analysis Plan **
        {self.plan_key}
        ****
        ** Focus Area Statement **
        {self.FA_key}
        ******
        ** Specialist Reviews **
        Focus Area Feedback: {{FA_Report}}
        Variable Choice and Usage Feedback:{{Data_Reality_Report}}
        Research Feedback: {{RAG_Interpretation}}
        Output Instruction Feedback: {{Output_Instruction_Review}}
        ****
        """
        #-----------------
        self.Compiler_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Review_Compiler=Review_Compiler_Agent(
            name="Review_Compiler",
            llm_config=self.Compiler_LLM_Config,
            system_message=self.Review_Compiler_System_Message,
            Update_System_Message=self.Review_Compiler_Update_System_Message)
        
        # Handoffs ---------------
        self.Review_Compiler.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Compilation_Complete} == True")
                )
            )
        )

        self.Review_Compiler.agent.handoffs.set_after_work(AgentTarget(self.Review_Compiler.agent))

        # Functions-------
        register_function(
            Review,
            caller=self.Review_Compiler.agent,
            executor=self.Review_Compiler.agent,
            name="Review",
            description="Combine all feedback provided by individual specalists into detailed instructions for improving the data analysis plan."
            )
        
    def run_Conversation(self):
        # Reset Context Variables:
        self.context_variables["Compilation_Complete"]=False
        self.context_variables["Compiled_Review"]=""
        #------------------------------------------
        agents=[self.Review_Compiler.agent]
        pattern=DefaultPattern(
        initial_agent=agents[0],
        agents=agents,
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,)

        result, ctx, _ = initiate_group_chat(
        pattern=pattern,
        messages="Using the information available, combine all feedback provided by individual specalists into detailed instructions for improving the data analysis plan.",
        max_rounds=self.Max_Rounds)
    
        return result, ctx

def Review(R: Annotated[str,"You must use the variable 'R' and make sure to use the strcutured response to provide your feedback."],
Score: Annotated[float,"You must provide a numeric score between 1 and 10 (1=Poor, 10=Perfect) to accompany your feedback." ],
context_variables: ContextVariables) -> ReplyResult:
    #Final_Review=""
    #for s in R.Steps:
        #x="""\n The analysis step for improvement: {s.Step_Number} \n Instructions for Improment of Variable Choice and Usage: {s.Variables_Feedback} \n Instructions for Improving Focus Area Alignment: {s.Scope} \n Research Informed Instructions and Information: {s.Research} \n Output Format Instructions: {s.Output} \n """
        #Final_Review+=x
    Final_Review=R
    context_variables["Compiled_Review"]=Final_Review
    context_variables["Compilation_Complete"]= True
    context_variables[f"{Review_Compilation.Type}_Plan_Scoring_{Review_Compilation.Iteration}"][3]=Score
    return ReplyResult(
        message=f"Combined Feedback Completed.",
        context_variables=context_variables,
    )
#--------------------------------------------------------------------------------------

def main():
    context_variables=ContextVariables({
        # Context Variables for the Reality Checker Agent
        "metadata": "'metadata chromatography_combined.csv' \n Variables: ['Unnamed: 0', 'start_well', 'end_well', 'Sample_Code', 'process_part', 'run', 'chromatography_stage', 'COLUMN_IN_USE', 'Resin_type', 'column', 'Column_Volume_(ml)', 'Load_volume_(mL)', 'Load_pH', 'Unicorn_server_result_file', 'Exported_run_data_file', 'Unicorn_Method_Name', 'Chromatography_Plate_ID', 'Plate_start', 'Plate_end', 'Chromatography_Plate_ID_2', 'Plate2_start', 'Plate2_end', 'run_no', 'Run_Name', 'Excel_Filename', 'Shortened_Code_for_Bioaccord_Samples', 'Run_variables_file', 'Operator_name_(Full)', 'Date', 'Upstream_sample', 'Aliquot_code', 'Time_samples_removed_from_fridge', 'Time_sample_returned_to_fridge', 'IDBS_experiment_number', 'Sample_start_volume_(mL)', 'Sample_volume_after_dilution_with_HPW_(mL)', 'Sample_final_volume_after_pH_titration_(mL)', 'Titration_agent', 'Start_pH', 'Final_pH', 'Total_protein_concentration_(bradford)_(mg/mL)', 'Conducivity_(mS)', 'Concentration_(mg/mL)', 'Filter_type_used', 'Load_sample_start_ID', 'Load_sample_start_IDBS_name', 'Load_sample_end_IDBS_name', 'EQ_and_wash_buffer', 'EQ_and_Wash_Buffer_Inlet', 'Elution_buffer', 'Elution_Buffer_Inlet', 'Sanitisation_buffer', 'Other_buffer', 'All_columns_loaded_from_same_material', 'Notes', 'fraction_volume', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise'] \n -----'MS_combined.csv' ------ \n ['Unnamed: 0', 'Unique_Peak_ID', 'Unique_MS_Sample_ID', 'Sample_Code', 'run_code', 'process_part', 'run', 'MS_method', 'column', 'chromatography_stage', 'start_well', 'end_well', 'Replicate', 'Type', 'Molecule ID', 'Component', 'Observed_TIC_RT_(mins)', 'Observed_UV_RT_(mins)', 'Observed RT delta (mins)', 'Response', '%_of_response', 'Observed_neutral_mass_(Da)', 'Observed_m/z', 'Spectrum_type', 'Expected_mass_(Da)', 'Mass_error_(ppm)', 'Alternative_assignments', 'fraction_volume', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise']\n ------'chromatography_combined.csv'------ \n ['Unnamed: 0', 'run_no', 'run', 'Fraction_unique_ID', 'column', 'volume_ml', 'UV_1_280_ml', 'UV_1_280_mAU', 'Cond_ml', 'Cond_mS_cm', 'Conc_B_ml', 'Conc_B_%', 'Injection_ml', 'Injection_Injection', 'Run_Log_ml', 'Run_Log_Logbook', 'Fraction_ml', 'Fraction_Fraction', 'UV_1_280_CUT_TEMP_100_BASEM_ml', 'UV_1_280_CUT_TEMP_100_BASEM_mAU', 'UV_2_260_ml', 'UV_2_260_mAU', 'pH_ml', 'pH_pH', 'DeltaC_pressure_ml', 'DeltaC_pressure_MPa', 'System_flow_ml', 'System_flow_ml_min', 'Sample_flow_ml', 'Sample_flow_ml_min', 'Sample_Code', 'chromatography_stage', 'chromatography_stage_order', 'Fraction_number', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise', 'fraction_volume']",
        "Univariate_EDA_Report": "### Dataset Interpretation Report\n\nThis report provides a detailed interpretation of the three files in the dataset: `metadata chromatography_combined.csv`, `MS_combined.csv`, and `chromatography_combined.csv`. Each file is analyzed independently to understand its structure, content, and relevance for downstream analysis.\n\n---\n\n### 1. `metadata chromatography_combined.csv`\n\n**File Structure and Type:**\n- This is a CSV file with a data shape of 354 rows and 58 columns.\n- The file contains metadata related to chromatography runs, including experimental parameters, sample details, buffer information, and operational notes.\n\n**Key Observations:**\n- **Data Types:** The file contains a mix of `int64`, `float64`, and `object` (string) data types. Notably, several columns (e.g., `start_well`, `end_well`, `Sample_Code`) are of type `object`, indicating categorical or textual data.\n- **Missing Values:** Many columns (e.g., `Time_samples_removed_from_fridge`, `Time_sample_returned_to_fridge`, `Notes`) have no observed values (all NaNs), suggesting they may be unused or incomplete.\n- **Dense Data:** The dataset is dense, with most numerical columns having meaningful values. The `Unnamed: 0` column appears to be an index, with values ranging from 0 to 49.\n- **Categorical Variables:** Several columns are highly categorical:\n  - `Resin_type` (4 unique values)\n  - `chromatography_stage` (16 unique values)\n  - `Run_Name`, `Excel_Filename`, `Operator_name_(Full)` (11, 11, and 1 unique values respectively), indicating a small number of experimental runs.\n- **Numerical Variables:** Key experimental parameters include:\n  - `Load_volume_(mL)` (mean ~112 mL, range 35–188 mL)\n  - `Sample_start_volume_(mL)` (mean ~166 mL, range 90–270 mL)\n  - `Total_protein_concentration_(bradford)_(mg/mL)` (mean ~0.16 mg/mL, range 0.10–0.44 mg/mL)\n  - `Start_pH` and `Final_pH` (mean ~7.1 and ~6.2, respectively)\n- **Grouping:** The data is grouped by `run`, `run_no`, and `Sample_Code`, suggesting that each row corresponds to a unique experimental run or sample.\n- **Relevance:** This file serves as a comprehensive metadata source for chromatography experiments, linking sample, process, and operational details. It is essential for contextualizing the other two datasets.\n\n**Conclusion:** This file is a dense, structured metadata file critical for understanding the experimental setup. It should be used to enrich and validate the other datasets.\n\n---\n\n### 2. `MS_combined.csv`\n\n**File Structure and Type:**\n- This is a large CSV file (620.35 MB) with 1,912,029 rows and 30 columns.\n- It contains mass spectrometry (MS) data, likely from peptide or protein identification and quantification.\n\n**Key Observations:**\n- **Data Types:** Mixed types, with `int64`, `float64`, and `object` columns. The `Unnamed: 0` column appears to be an index.\n- **Missing Values:** Several columns (e.g., `Type`, `Molecule ID`, `Component`, `Expected_mass_(Da)`, `Mass_error_(ppm)`, `Alternative_assignments`) have no observed values (all NaNs), indicating they may be unused or incomplete.\n- **Sparse Data:** Despite its large size, the dataset is sparse in certain key columns, particularly those related to molecular identification.\n- **Categorical Variables:**\n  - `Unique_MS_Sample_ID` (283 unique values)\n  - `Sample_Code` (274 unique values)\n  - `chromatography_stage` (19 unique values)\n  - `Spectrum_type` (3 unique values)\n- **Numerical Variables:** Key MS features include:\n  - `Observed_TIC_RT_(mins)` (mean ~2.33 mins, range ~0.62–3.03 mins)\n  - `Observed_UV_RT_(mins)` (mean ~1.98 mins, range ~0.74–2.72 mins)\n  - `Response` (mean ~9,920, range ~4.86–15.25M)\n  - `Observed_neutral_mass_(Da)` (mean ~162.5 kDa, range ~400–314 kDa)\n  - `Observed_m/z` (mean ~733 Da, range ~505–1,263 Da)\n- **Grouping:** Data is grouped by `Unique_MS_Sample_ID`, `Sample_Code`, and `run_code`, suggesting a hierarchical structure where multiple MS spectra are associated with a single sample.\n- **Relevance:** This file contains high-resolution MS data, likely from LC-MS/MS experiments. It is essential for downstream analysis such as peptide identification, quantification, and post-translational modification (PTM) analysis.\n\n**Conclusion:** This file is a large, sparse dataset containing detailed MS data. It is highly relevant for proteomics analysis but requires careful handling due to missing values in key identification columns.\n\n---\n\n### 3. `chromatography_combined.csv`\n\n**File Structure and Type:**\n- This is a large CSV file (257.96 MB) with 1,084,069 rows and 37 columns.\n- It contains chromatographic data, including UV, conductivity, pH, flow, and volume measurements across fractions.\n\n**Key Observations:**\n- **Data Types:** Mixed types, with `int64`, `float64`, and `object` columns. The `Unnamed: 0` column is likely an index.\n- **Missing Values:** Some columns (e.g., `Injection_Injection`, `Fraction_Fraction`) have no observed values (all NaNs), suggesting they may be unused.\n- **Dense Data:** The dataset is dense, with most numerical columns having meaningful values. The `Fraction_number` and `Sample_Code` columns suggest a hierarchical structure where multiple fractions are generated per sample.\n- **Categorical Variables:**\n  - `column` (4 unique values)\n  - `chromatography_stage` (16 unique values)\n  - `Run_Log_Logbook` (17 unique values)\n- **Numerical Variables:** Key chromatographic parameters include:\n  - `volume_ml` (mean ~126 mL, range ~-33.8–335.8 mL)\n  - `UV_1_280_ml` and `UV_1_280_mAU` (mean ~126 mL and ~250 mAU, respectively)\n  - `Cond_ml` and `Cond_mS_cm` (mean ~126 mL and ~15.8 mS/cm)\n  - `pH_ml` and `pH_pH` (mean ~132.5 mL and ~6.28 pH)\n  - `fraction_volume` (mean ~20.9 mL, range ~0.008–66.0 mL)\n- **Grouping:** Data is grouped by `Sample_Code`, `run_no`, and `Fraction_number`, indicating that each row corresponds to a fraction collected during a chromatography run.\n- **Relevance:** This file contains high-resolution chromatographic profiles, essential for understanding elution patterns, peak detection, and fraction collection. It is critical for integrating with MS data to link protein identity to chromatographic behavior.\n\n**Conclusion:** This file is a dense, structured chromatographic dataset that provides detailed fraction-level data. It is indispensable for correlating MS results with chromatographic elution profiles.\n\n---\n\n### Overall Summary\n\n- The dataset consists of three interrelated files:\n  1. `metadata chromatography_combined.csv`: Metadata for experimental runs.\n  2. `chromatography_combined.csv`: High-resolution chromatographic profiles (fraction-level).\n  3. `MS_combined.csv`: High-resolution MS data (peptide/protein-level).\n\n- **Integration Potential:** The three files can be integrated using common keys such as `Sample_Code`, `run_no`, and `chromatography_stage`. This integration enables a comprehensive analysis of protein behavior across chromatography and MS platforms.\n\n- **Gaps and Limitations:**\n  - Several columns in `MS_combined.csv` are entirely missing (all NaNs), which may limit downstream identification.\n  - Some columns in all files (e.g., `Notes`, `Other_buffer`) are unused or incomplete.\n  - Further validation is required to confirm the consistency of `Sample_Code` and `run_no` across files.\n\n- **Recommendations for Further Analysis:**\n  - Perform data merging using `Sample_Code` and `run_no` to align chromatography and MS data.\n  - Investigate the reasons for missing values in `MS_combined.csv`.\n  - Validate the integrity of `fraction_volume` and related columns across files.\n  - Conduct a time-series analysis of chromatographic profiles to identify elution patterns.\n\nThis dataset is well-structured for integrative proteomics analysis, provided that missing data issues are addressed.",

        "Usability_Plan":
        {"Plan_Section": [{"Step_Number": 1, "Analysis_Type": "Data Completeness and Missingness Analysis with Pattern Detection and Cross-Tabulation", 
        "Data_File": "metadata chromatography_combined.csv", 
        "Variables": ["Sample_Code", "Run_Name", "Resin_type", "column", "chromatography_stage", "Load_volume_(mL)", "Load_pH", "Total_protein_concentration_(bradford)_(mg/mL)", "fraction_volume"], 
        "Context": "This dataset contains experimental metadata for HPLC runs. Perform a comprehensive missingness analysis by computing missingness per variable and per experimental group (e.g., by 'chromatography_stage' and 'Resin_type') to detect systematic gaps. Use the 'missingno' library to generate a heatmap of missingness. Additionally, use pivot tables or cross-tabulations to examine missingness patterns across 'chromatography_stage' and 'Resin_type' to detect systematic data gaps (e.g., certain resin types missing data in later stages). Flag variables with >5% missingness. Use pandas for missingness computation and seaborn for visualization. Include a summary table of missingness rates and flagged variables. Ensure that 'Sample_Code' and 'Run_Name' are used to validate the integrity of the metadata linkage.",
        "Output_Format": "Text-based output appended to Usability.md"}, 
        {"Step_Number": 2, "Analysis_Type": "Signal Intensity and Noise Level Assessment with Per-Group Noise Analysis and Baseline Drift Quantification", 
        "Data_File": "MS_combined.csv", "Variables": ["Response", "%_of_response", "Observed_TIC_RT_(mins)", "Spectrum_type", "Fraction_number"], 
        "Context": "MS data is large and contains signal intensity (Response) and relative abundance (%_of_response). Compute the signal-to-noise ratio (SNR) as mean(Response)/std(Response) per 'Spectrum_type'. Assess noise by computing the standard deviation of 'Response' values where Response < 100, grouped by 'Spectrum_type' and 'Fraction_number'. Additionally, compute the percentage of 'Response' values below 10 to quantify baseline drift or instrument noise. Use log(Response) for visualization. Perform a Shapiro-Wilk test on the log-transformed 'Response' to assess normality. Use a violin plot to visualize noise distribution and a histogram of log(Response) for signal distribution. Use scipy.stats for Shapiro-Wilk test and seaborn for plotting.", 
        "Output_Format": "Visualisation saved as 'signal_noise_assessment.png'"}]},

        "HPLC_Analysis_Plan": {"Plan_Section": [{"Step_Number": 1, "Analysis_Type": "Data Quality Assessment and Baseline Correction", 
        "Data_File": "chromatography_combined.csv", 
        "Variables": ["UV_1_280_mAU", "UV_2_260_mAU", "Fraction_number", "chromatography_stage"], 
        "Context": "Prior to any chromatographic analysis, perform a preliminary data quality check to identify outliers, noise, and baseline drift in UV absorbance signals. Use the 'UV_1_280_mAU' and 'UV_2_260_mAU' signals, grouped by 'chromatography_stage', to detect anomalies. Apply the Asymmetric Least Squares (ALS) method for baseline correction, which is a standard technique in HPLC data processing. This step ensures that subsequent peak detection and integration are performed on cleaned, stable signals. Use 'Fraction_number' as the x-axis for baseline correction. Save the corrected signals as a new column in a temporary dataset for use in later steps.", 
        "Output_Format": "Text-based output appended to HPLC.md"}, 
        {"Step_Number": 2, "Analysis_Type": "Exploratory Data Analysis (EDA) on Chromatography Data with Corrected Signals", 
        "Data_File": "chromatography_combined.csv", 
        "Variables": ["Fraction_number", "UV_1_280_mAU", "UV_2_260_mAU", "fraction_volume_ml_min_precise", "chromatography_stage", "chromatography_stage_order"], 
        "Context": "Using the baseline-corrected UV signals from Step 0, perform a refined EDA. Compute descriptive statistics (mean, median, std, min, max) for 'fraction_volume_ml_min_precise' and the corrected UV signals, grouped by 'chromatography_stage'. Generate histograms and boxplots to visualize the distribution of fraction volumes and UV signal intensities. This step will reveal stage-specific trends and potential data anomalies. Use 'fraction_volume_ml_min_precise' for precision in volume representation.", 
        "Output_Format": "Text-based output appended to HPLC.md"}]},

        "Mass_Spectrometry_Plan": "",

        "Data_Reality_Report":"",
        "Reality_Feedback": False,
        #-----------------------------
        # Context Variables for the Focus Area Assessor Agent (Additional to Above):
        "Focus_Area_Usability": "The focus area is the assessment of data quality, reliability, and relevance of High Performance Liquid Chromatography (HPLC) and Mass Spectrometry (MS) data generated during a biopharmaceutical manufacturing process. Data analysis in this context involves evaluating the integrity of chromatographic and MS datasets—specifically, identifying anomalies such as sensor drift, noise, missing values, and inconsistent metadata—to ensure that the data accurately reflects the biological and process characteristics of the product. The primary goal is to verify that the data is fit for purpose in supporting critical quality attributes (CQAs), such as purity, identity, and structural integrity of the biopharmaceutical product. This includes validating the consistency of sample tracking across files, detecting outliers in elution profiles or MS response, and ensuring that MS identifications are supported by reliable chromatographic co-elution. Appropriate data analysis includes SNR Calculation, Chromatographic Peak Shape Analysis, MS Signal Consistency Check, Metadata Alignment Validation, Missing Data Pattern Analysis, Chromatography-MS Integration, Sensor Drift Detection, Outlier Detection in Response Intensity, RT Alignment Validation, and Data Completeness Assessment. The context is that the datasets are large, high-resolution, and hierarchical, with multiple levels of sample and run metadata; therefore, analysis must account for the expected variability in elution times and response intensities due to process dynamics, while distinguishing true anomalies from normal process variation.",
        "Focus_Area_HPLC": "**Statement**\nHPLC data analysis in the context of biopharmaceutical manufacturing focuses on characterizing chromatographic profiles from fraction-level data to assess product quality, purity, and consistency. The primary objective is to identify and quantify target protein peaks, detect impurities or degradation products, and ensure process reproducibility across runs. This analysis relies on the integrity, quality, and relevance of chromatographic data, including UV absorbance, conductivity, pH, and volume measurements, to support process understanding, release testing, and regulatory compliance.\n\n**Suggested Data Analysis**\nPeak Detection, Peak Integration, Chromatographic Alignment, Time-Series Analysis, Multivariate Analysis (PCA), Process Variability Assessment, Fraction Classification, Elution Profile Comparison, Outlier Detection, Run-to-Run Consistency Analysis\n\n**Context**\nThe chromatography data is collected at the fraction level, with each row representing a fraction from a specific run and sample. Data is grouped by `Sample_Code`, `run_no`, and `Fraction_number`, enabling analysis at the fraction, run, and process stage levels. The data is dense and high-resolution, making it suitable for detailed peak profiling and comparison across stages (e.g., capture, polishing). Analysis should account for variations in load volume, pH, buffer composition, and resin type, as these are captured in the metadata and can influence elution behavior. Integration with metadata (e.g., `chromatography_stage`, `Resin_type`, `Start_pH`) is essential for contextual interpretation.",
        "Focus_Area_MS": "**Statement**\nMass spectrometry (MS) data analysis in biopharmaceutical manufacturing aims to identify and quantify proteins and peptides, assess their structural integrity, and detect critical quality attributes (CQAs) such as post-translational modifications (PTMs), deamidation, oxidation, and truncations. This analysis is essential for ensuring product consistency, safety, and efficacy, with the primary result being a reliable characterization of the biologic’s molecular identity and purity. Data quality, reliability, and relevance are ensured by validating spectral integrity, alignment across runs, and consistency with chromatographic elution profiles.\n\n**Suggested Data Analysis**\n- Peak Detection and Integration\n- Protein/Peptide Identification (via database search)\n- Quantification (label-free or label-based)\n- Post-Translational Modification (PTM) Analysis\n- Mass Error Assessment\n- Fragmentation Pattern Analysis\n- Multivariate Analysis (PCA, PLS-DA)\n- Batch Effect Correction\n- Temporal Profile Analysis (by chromatography stage)\n- Spectral Quality Control (SQC)\n\n**Context**\nThe MS data is collected across multiple chromatography stages and runs, with samples linked via `Sample_Code`, `run_code`, and `chromatography_stage`. Integration with chromatography data enables correlation of MS signals with elution behavior, enhancing confidence in identifications. The presence of missing values in key identification columns (e.g., `Molecule ID`, `Expected_mass_(Da)`) necessitates careful data filtering and quality control. Analysis must account for run-to-run variability and ensure that only high-quality, reproducible spectra are used for downstream characterization.",
        "FA_Feedback": False,
        "FA_Report":"",
        #----- Iter2 ----------
        "Data_Structure_Advice":"",
        "Structure_Advice_Provided": False,
        #----- Iter3 -----------
        "RAG_QA":"",
        "QA_Available": False,
        "RAG_Interpretation":"",
        "RAG_Interpretation_Available": False,
        "Compiled_Review":"",
        "Compilation_Complete": False,
        #------- Iter 4 -----------
        "Output_Instruction_Review":"",
        "OP_Instruction_Available": False,
        "Output_Data_Structure_Advice":"",
        "OP_Structure_Advice_Available": False,
    })
    # Chat to be tested:
    Reality_Chat1=Reality_Chat(context_variables=context_variables, Analysis_Type="Usability", Max_Rounds=5)
    Reality_Chat1.run_Conversation()

    FA1=Focus_Area_Chat(context_variables=context_variables, Analysis_Type="Usability", Max_Rounds=5)
    FA1.run_Conversation()

    R1=Review_Compilation(context_variables=context_variables, Analysis_Type="Usability", Max_Rounds=5)
    R1.run_Conversation()
    #--------------------

if __name__=="__main__":
    main()