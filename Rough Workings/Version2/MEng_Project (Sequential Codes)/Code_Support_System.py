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
# The following agents are used to enhance the data analysis plan by leveraging factual information by using a RAG questioner
# and a RAG interpreter agent. The aim IS NOT to change the plan, but provide domain specific nowledge opn implementation
# guidelines and possibly suggest python packages that should be installed to support effective data analysis.

class RAG_Questioning_Agent(Agent_Base):
    pass
class RAG_Enhancement_Agent(Agent_Base):
    pass
class Enforcer_Agent(Agent_Base):
    pass
class Error_Checker_Agent(Agent_Base):
    pass
# 1-------------------
class RAG_Enhancer():
    Type: ""
    def __init__(self, context_variables: ContextVariables, Analysis_Type: str, Max_Rounds: int,Iter:int, Step_Group:list):
        self.context_variables=context_variables
        self.Analysis_Type=Analysis_Type
        self.Step_Group=Step_Group
        RAG_Enhancer.Type=Analysis_Type
        self.Max_Rounds=Max_Rounds
        self.Iter=Iter
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        #----- Obtaining Correct Section of the Plan -----
        self.start_index=self.Step_Group[Iter][0]
        self.end_index=self.Step_Group[Iter][1] +1 #Makes the end inclusive.

        def format_plan_steps(plan_group):
            return "\n".join(
                "Step {Step_Number}\n"
                "- Analysis_Type: {Analysis_Type}\n"
                "- Data_File: {Data_File}\n"
                "- Variables: {Variables}\n"
                "- Context: {Context}\n"
                "- Output_Format: {Output_Format}\n"
                .format(**step)
                for step in plan_group
            )
        if self.Analysis_Type=="Usability":
            self.Plan_Group= self.context_variables["Usability_Plan"]["Plan_Section"][self.start_index:self.end_index]
            steps = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in self.Plan_Group
            ]
            self.plan_key = format_plan_steps(steps)
        elif self.Analysis_Type=="HPLC":
            self.Plan_Group= self.context_variables["HPLC_Analysis_Plan"]["Plan_Section"][self.start_index:self.end_index]
            steps = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in self.Plan_Group
            ]
            self.plan_key = format_plan_steps(steps)
        elif self.Analysis_Type=="MS":
            self.Plan_Group= self.context_variables["Mass_Spectrometry_Plan"]["Plan_Section"][self.start_index:self.end_index]
            steps = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in self.Plan_Group
            ]
            self.plan_key = format_plan_steps(steps)
        else:
            raise ValueError(f"Invalid Analysis_Type: {self.Analysis_Type}. Must be one of 'Usability', 'HPLC', 'MS'.")
        #--------------------------
        #----- Obtaining the Correct Code to Review ------
        self.Code = f"""{{{self.Analysis_Type}_Code_{self.Iter}}}"""
        #-------------------------------------------------
        self.RAG_Questioner_System_Message="""
        You are a research agent.
        """
        self.RAG_Questioner_Update_System_Message=f"""
        -------- ROLE ------------------
        You are an expert research agent that specialises in asking the right questions to gather domain specfic information that is crucial for effective data analysis.
        You have been provided with the current plan and your questions must be desgned to gather crucial implementation information to help a coder agent effectively implement the plan.
        The plan is fixed and must never be changed in any way. Instead, you focus on gathering the information to allow the coder to understand how to carry out the plan.
        You have been provided with the focus area statement which defines the main aims of the data analysis that is intended to be addressed by the plan.
        You have been provded with the current code implementation to help you understand what has already been implemented and where gaps in knowledge may exist.
        The questions you ask will allow the correct domain specific information to be retrieved from a research database. You should focus on asking questions related to understanding how to implement the plan. 
        Your questions can include asking if any pre-processing steps are necessary, what python packages are best suted to carry out the analysis suggested in the plan, and any other domain specific information that is crucial for effective implementation of the plan.
        You must never falsify any information about the dataset or the plan. All comments must be grounded in the information provided.
        -------------------------------
        ----------- Task -----------------
        - You must call the RAG_Query function with your domain specific questions.
        - Your questions must be clear and concise. There should be no ambiguity in what you are asking. You must ensure that each question explores a different area of the plan or focus area to ensure adequate coverage.
        - You wll be able to ask 6 questions in total.
        -----------------------------------
        ------------- Context -------------
        ** Current Plan for Data Analysis **
        {self.plan_key}
        *************
        ** Current Code Implementation **
        {self.Code}
        *************
        ** Focus Area Statement **
        {{Focus_Area_{self.Analysis_Type}}}
        """
        self.RAG_Interpreter_System_Message="""
        You are a research interpreter agent.
        """
        self.RAG_Interpreter_Updated_System_Message=f"""
        --------- ROLE -------------------
        You are an expert research interpreter that specalises in advising a coder agent on how to implement a given data analysis plan based on retrieved domain specific information.
        You have been provided with the current plan and the focus area statement that defines the main aims of the data analysis.
        You have been provded with the current code implementation to help you understand what has already been implemented and where gaps in knowledge may exist.
        You MUST NEVER attempt to change the plan in any way. Your role is ONLY to provide domain specific advice to help a coder agent effectively implement the plan.
        You have been provided with research questions and answers. You are only allowed to use this information to provide your advice as ths information has come form a research database.
        You must never falsify any information and if the retrieved research does not provide relevant information to the plan, then you must not comment on that specific area.
        You must focus on providing clear advice that may help with pre-processing steps, suggest relevant python packages and advice on how to use them, and any other domain specific information that will help the coder agent effectively implement the plan.
        ----------------------------------
        ----------- Task -----------------
        - You must call the RAG_Enhancement function to provide information about the domain and instructions for how to improve the plan.
        - You must focus on providing a clear and concise interpretation of the retrieved questions and answers to help a coder agent to effectively impplement the plan.
        - Never change the plan, the varables used or the output format in any way.
        ----------------------------------
        ------- Output Format -------------
        Your interpreation must be clear and concise. You must try to use bullet points where possible to enusre clarity. You must make the main points clear to the coder agent.
        -----------------------------------
        -------- Context -------------------
        ** Current Analysis Plan **
        {self.plan_key}
        **********
        ** Current Code Implementation **
        {self.Code}
        ***********
        ** Focus Area Statement **
        {{Focus_Area_{self.Analysis_Type}}}
        ***********
        ** Retrieved Domain Specific Q&A **
        {{Coding_RAG_QA}}
        """
        
        #-----------------
        self.RAG_QA_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.RAG_Interpreter_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.RAG_Questioner=RAG_Questioning_Agent(
            name="RAG_Questioner",
            llm_config=self.RAG_QA_LLM_Config,
            system_message=self.RAG_Questioner_System_Message,
            Update_System_Message=self.RAG_Questioner_Update_System_Message)
        self.RAG_Interpreter=RAG_Enhancement_Agent(
            name="RAG_Interpreter",
            llm_config=self.RAG_Interpreter_LLM_Config,
            system_message=self.RAG_Interpreter_System_Message,
            Update_System_Message=self.RAG_Interpreter_Updated_System_Message)
        
        # Handoffs ---------------
        self.RAG_Questioner.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.RAG_Interpreter.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Code_QA_Available} == True")
                )
            )
        )

        self.RAG_Questioner.agent.handoffs.set_after_work(AgentTarget(self.RAG_Questioner.agent))

        self.RAG_Interpreter.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${RAG_Enhancement_Available} == True")
                )
            )
        )
        self.RAG_Interpreter.agent.handoffs.set_after_work(AgentTarget(self.RAG_Interpreter.agent))
        # Functions-------
        register_function(
            RAG_Query,
            caller=self.RAG_Questioner.agent,
            executor=self.RAG_Questioner.agent,
            name="RAG_Query",
            description="Generate 6 domain specific questions to gather crucial information required to advise a coder agent to implement the data analysis plan."
            )
        register_function(
            RAG_Enhancement,
            caller=self.RAG_Interpreter.agent,
            executor=self.RAG_Interpreter.agent,
            name="RAG_Enhancement",
            description="Provide domain specific knowledge to help a coder agent effectively implement the data analysis plan based on the retrieved research questions and answers. Never change the plan in any way."
            )
        
    def run_Conversation(self):
        # Reset Context Variables:
        self.context_variables["Code_QA_Available"]=False
        self.context_variables["RAG_Enhancement_Available"]=False
        self.context_variables["Coding_RAG_QA"]=""
        self.context_variables["RAG_Enhancements"]=""
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

def RAG_Query(Q1: Annotated[str,"Question for research database"],Q2: Annotated[str,"Question for research database"], Q3: Annotated[str,"Question for research database"], Q4: Annotated[str,"Question for research database"], Q5: Annotated[str,"Question for research database"], Q6: Annotated[str,"Question for research database"], context_variables: ContextVariables) -> ReplyResult:
    R1 = RAG_Tool(Q1)
    R2 = RAG_Tool(Q2)
    R3 = RAG_Tool(Q3)
    R4 = RAG_Tool(Q4)
    R5 = RAG_Tool(Q5)
    R6 = RAG_Tool(Q6)
    context_variables["Coding_RAG_QA"] = f""" Query: {Q1} | Result:{R1} \n Query: {Q2} | Result:{R2} \n Query: {Q3} | Result:{R3} \n Query: {Q4} | Result:{R4} \n Query: {Q5} | Result:{R5} \n Query: {Q6} | Result:{R6} """
    context_variables["Code_QA_Available"]=True

    #----- Reset Context Variables
    context_variables["RAG_Enhancement_Available"]= False

    return ReplyResult(
        message=f"RAG Questions Completed.",
        context_variables=context_variables,
    )

def RAG_Enhancement(RE: Annotated[str,"Interpretation of RAG results"], context_variables: ContextVariables) -> ReplyResult:

    context_variables["RAG_Enhancements"]=RE
    context_variables["RAG_Enhancement_Available"]= True
    #----- Reset Context Variables
    context_variables["Code_QA_Available"]= False
    return ReplyResult(
        message=f"RAG Interpretation Complete.",
        context_variables=context_variables,
    )
#--------------------------------------------------------------------------------------

#2-----------------------------------
class Plan_Enforcer():
    Type: ""
    Iteration: int
    def __init__(self, context_variables: ContextVariables, Analysis_Type: str, Max_Rounds: int, Iteration: int, Step_Group:list, Iter:int):
        # Iteration is used to track the number of times a section of the plan has been reviewed.
        # Iter is used to identify which section of the plan is being reviewed depending on the number of plan steps to implement at once.
        self.context_variables=context_variables
        self.Analysis_Type=Analysis_Type
        Plan_Enforcer.Type=Analysis_Type
        self.Max_Rounds=Max_Rounds
        Plan_Enforcer.Iteration=Iteration
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.Step_Group=Step_Group
        self.Iter=Iter
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()

        #----- Obtaining Correct Section of the Plan -----
        self.start_index=self.Step_Group[Iter][0]
        self.end_index=self.Step_Group[Iter][1] +1 #Makes the end inclusive.

        def format_plan_steps(plan_group):
            return "\n".join(
                "Step {Step_Number}\n"
                "- Analysis_Type: {Analysis_Type}\n"
                "- Data_File: {Data_File}\n"
                "- Variables: {Variables}\n"
                "- Context: {Context}\n"
                "- Output_Format: {Output_Format}\n"
                .format(**step)
                for step in plan_group
            )
        if self.Analysis_Type=="Usability":
            self.Plan_Group= self.context_variables["Usability_Plan"]["Plan_Section"][self.start_index:self.end_index]
            steps = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in self.Plan_Group
            ]
            self.plan_key = format_plan_steps(steps)
        elif self.Analysis_Type=="HPLC":
            self.Plan_Group= self.context_variables["HPLC_Analysis_Plan"]["Plan_Section"][self.start_index:self.end_index]
            steps = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in self.Plan_Group
            ]
            self.plan_key = format_plan_steps(steps)
        elif self.Analysis_Type=="MS":
            self.Plan_Group= self.context_variables["Mass_Spectrometry_Plan"]["Plan_Section"][self.start_index:self.end_index]
            steps = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in self.Plan_Group
            ]
            self.plan_key = format_plan_steps(steps)
        else:
            raise ValueError(f"Invalid Analysis_Type: {self.Analysis_Type}. Must be one of 'Usability', 'HPLC', 'MS'.")
        #--------------------------
        #----- Obtaining the Correct Code to Review ------
        self.Code = f"""{{{self.Analysis_Type}_Code_{self.Iter}}}"""
        #-------------------------------------------------
        self.Plan_Enforcer_Agent_System_Message="""
        You are a Plan Enforcer agent.
        """
        self.Plan_Enforcer_Agent_Update_System_Message=f"""
        --------- ROLE -------------------
        You are a Plan Enforcement agent that specialises in reviewing code and providing feedback to enforce that the code follows the plan provided.
        You have been provided with the current plan section that needs to be analysed. You have been provided with the code that has been implemented to carry out the plan.
        You have been given information about the dataset avalable from a data discovery agent. This report covers all data available, but the plan will only mention one of these datasets at a time.
        You have been provided with this information to help you understand the context behind the plan. This is important to ensuring the correct handling of the data during analysis.
        You MUST NEVER attempt to change the plan in any way. Your role is ONLY to provide feedback on how well the code follows the plan and provide suggestions on how to improve alignment with the plan.
        You must be critcal in your review when the code does not follow the plan. You must provide a score between 1 and 10 (1=Poor, 10=Perfect) to accompany your feedback whch helps to indcate how well the code follows the plan.
        ----------------------------------
        ----------- Task -----------------
        You must call the Plan_Enforce function to provide your feedback on how well the code follows the plan.
        You must use all information available to you to provide clear feedback on how well the code follows the plan.
        You must provide a numeric score between 1 and 10 (1=Poor, 10=Perfect) to accompany your feedback.
        ----------------------------------
        ------- Output Format -------------
        - Feedback: You must try to use bullet points where possible to ensure clarity. You must make the main points clear to the coder agent.
        - Score: A numeric score between 1 and 10 (1=Poor, 10=Perfect) to accompany your feedback. You must use the Scoring Rules to help you determine the correct numeric score to provide. You do not have to use integers.
        -----------------------------------
        --------- Scoring Rules -----------
        - Score=10: The code is perfectly algned with the plan section provided, with no issues in the choice of variables or analysis suggested. The analysis suggested is fully achievable with the data available.
        - Score=7-9: The code is mostly algned with the plan. Minor issues may be present but it does not change the overall alignment with the plan. The analysis suggested is achievable with the data available.
        - Score=4-6: The code has misalignments with the plan that need to be addressed. The data analysis would be poor give the current code provided.
        - Score=1-3: The code either attempts to change the plan or is completely misaligned with the plan provided. Ths would result in completely incorrect data analysis.
        -----------------------------------
        -------- Context -------------------
        ** Current Analysis Plan Section **
        {self.plan_key}
        **************
        ** Implemented Code to Review **
        {self.Code}
        **************
        ** Dataset Information (From Dataset Discovery Agent) **
        {{Univariate_EDA_Report}}
        """
        #-----------------
        self.Enforcer_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Enforcer=Enforcer_Agent(
            name="Enforcer",
            llm_config=self.Enforcer_LLM_Config,
            system_message=self.Plan_Enforcer_Agent_System_Message,
            Update_System_Message=self.Plan_Enforcer_Agent_Update_System_Message)

        # Handoffs ---------------
        self.Enforcer.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Enforcer_Used} == True")
                )
            )
        )

        self.Enforcer.agent.handoffs.set_after_work(AgentTarget(self.Enforcer.agent))

        # Functions-------
        register_function(
            Plan_Enforce,
            caller=self.Enforcer.agent,
            executor=self.Enforcer.agent,
            name="Plan_Enforce",
            description="Generates a code review to enforce that the implementation of the data analysis plan is closely following the plan provided, and provides feedback on how to improve alignment with the plan based on the provided context."
            )
        
    def run_Conversation(self):
        # Reset Context Variables:
        self.context_variables["Enforcer_Used"]=False
        self.context_variables["Enforcer_Feedback"]=""
        #-------------------------
        agents=[self.Enforcer.agent]
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

def Plan_Enforce(Feedback: Annotated[str,"Generates a code review to enforce that the implementation of the data analysis plan is closely following the plan provided, and provides feedback on how to improve alignment with the plan based on the provided context."],
Score: Annotated[float,"You must provide a numeric score between 1 and 10 (1=Poor, 10=Perfect) to accompany your feedback." ],
context_variables: ContextVariables) -> ReplyResult:

    context_variables["Enforcer_Used"]=True
    context_variables["Enforcer_Feedback"]=Feedback
    # Do Scoring at a later stage....

    return ReplyResult(
        message=f"Plan Enforcement for Code Completed.",
        context_variables=context_variables,
    )

#-------------------------------------------------------------------
#3--------------------------
class Error_Checking_System():
    Type: ""
    Iteration: int
    def __init__(self, context_variables: ContextVariables, Analysis_Type: str, Max_Rounds: int, Iteration: int, Step_Group:list, Iter:int):
        # Iteration is used to track the number of times a section of the plan has been reviewed.
        # Iter is used to identify which section of the plan is being reviewed depending on the number of plan steps to implement at once.
        self.context_variables=context_variables
        self.Analysis_Type=Analysis_Type
        Error_Checking_System.Type=Analysis_Type
        self.Max_Rounds=Max_Rounds
        Error_Checking_System.Iteration=Iteration
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.Step_Group=Step_Group
        self.Iter=Iter
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()

        #----- Obtaining Correct Section of the Plan -----
        self.start_index=self.Step_Group[Iter][0]
        self.end_index=self.Step_Group[Iter][1] +1 #Makes the end inclusive.

        def format_plan_steps(plan_group):
            return "\n".join(
                "Step {Step_Number}\n"
                "- Analysis_Type: {Analysis_Type}\n"
                "- Data_File: {Data_File}\n"
                "- Variables: {Variables}\n"
                "- Context: {Context}\n"
                "- Output_Format: {Output_Format}\n"
                .format(**step)
                for step in plan_group
            )
        if self.Analysis_Type=="Usability":
            self.Plan_Group= self.context_variables["Usability_Plan"]["Plan_Section"][self.start_index:self.end_index]
            steps = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in self.Plan_Group
            ]
            self.plan_key = format_plan_steps(steps)
        elif self.Analysis_Type=="HPLC":
            self.Plan_Group= self.context_variables["HPLC_Analysis_Plan"]["Plan_Section"][self.start_index:self.end_index]
            steps = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in self.Plan_Group
            ]
            self.plan_key = format_plan_steps(steps)
        elif self.Analysis_Type=="MS":
            self.Plan_Group= self.context_variables["Mass_Spectrometry_Plan"]["Plan_Section"][self.start_index:self.end_index]
            steps = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in self.Plan_Group
            ]
            self.plan_key = format_plan_steps(steps)
        else:
            raise ValueError(f"Invalid Analysis_Type: {self.Analysis_Type}. Must be one of 'Usability', 'HPLC', 'MS'.")
        #--------------------------
        #----- Obtaining the Correct Code to Review ------
        self.Code = f"""{{{self.Analysis_Type}_Code_{self.Iter}}}"""
        #-------------------------------------------------
        self.Error_Checker_System_Message="""
        You are an Error Checker agent.
        """
        self.Error_Checker_Update_System_Message=f"""
        --------- ROLE -------------------
        You are an error checking agent that specalises in reviewing Python code to identify any errors that 
        would prevent the code from running successfully.
        You have been provided with the code that has been implemented to carry out a data analysis plan.
        You MUST NEVER attempt to change the plan in any way. Your role is ONLY to identify any errors in the code that would prevent it from running successfully.
        You must be critcal in your review to identify any errors in the code provided.
        ----------------------------------
        ------------ Task -----------------
        You must call the Error_Check function to provide your feedback on any errors in the code that would prevent it from running successfully, and provide suggestions on how to fix these issues.
        You must use all information available to you to provide clear feedback on any errors in the code that would prevent it from running successfully, and provide suggestions on how to fix these issues.
        Never attempt to change the plan in any way.
        ----------------------------------
        ------- Output Format -------------
        - Feedback: You must try to use bullet points where possible to ensure clarity. You must make the main points clear to the coder agent.
        - Score: A numeric score between 1 and 10 (1=Poor, 10=Perfect) to accompany your feedback. You must use the Scoring Rules to help you determine the correct numeric score to provide. You do not have to use integers.
        -----------------------------------
        --------- Scoring Rules -----------
        - Score=10: The code has no errors and would run successfully without any issues.
        - Score=7-9: The code has minor errors that would not prevent it from running successfully, but these issues should be fixed to ensure the code runs as effectively as possible.
        - Score=4-6: The code has errors that would prevent it from running successfully, but these issues could be fixed with some effort.
        - Score=1-3: The code has major errors that would prevent it from running successfully, and these issues would be difficult to fix. The code would not run successfully in its current state.
        -----------------------------------
        -------- Context -------------------
        ** Code To Review **
        {self.Code}
        **************
        """
        #-----------------
        self.Error_Checker_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Error_Checker=Error_Checker_Agent(
            name="Error_Checker",
            llm_config=self.Error_Checker_LLM_Config,
            system_message=self.Error_Checker_System_Message,
            Update_System_Message=self.Error_Checker_Update_System_Message)
        # Handoffs ---------------
        self.Error_Checker.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Errors_Checked} == True")
                )
            )
        )

        self.Error_Checker.agent.handoffs.set_after_work(AgentTarget(self.Error_Checker.agent))

        # Functions-------
        register_function(
            Error_Check,
            caller=self.Error_Checker.agent,
            executor=self.Error_Checker.agent,
            name="Error_Check",
            description="Generates a code review to identify any errors in the implementation of the data analysis plan, and provides feedback on how to fix any issues found."
            )
        
    def run_Conversation(self):
        # Reset Context Variables:
        self.context_variables["Errors_Checked"]=False
        self.context_variables["Errors_Spotted"]=""
        #-------------------------
        agents=[self.Error_Checker.agent]
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

def Error_Check(Feedback: Annotated[str,"Generates a code review to identify any errors in the implementation of the data analysis plan, and provides feedback on how to fix any issues found."],
Score: Annotated[float,"You must provide a numeric score between 1 and 10 (1=Poor, 10=Perfect) to accompany your feedback." ],
context_variables: ContextVariables) -> ReplyResult:

    context_variables["Errors_Checked"]=True
    context_variables["Errors_Spotted"]=Feedback
    # Do Scoring at a later stage....

    return ReplyResult(
        message=f"Error Checking for Code Completed.",
        context_variables=context_variables,
    )
#-------------------------------------------------------------------

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

        "Focus_Area_Usability": "The focus area is the assessment of data quality, reliability, and relevance of High Performance Liquid Chromatography (HPLC) and Mass Spectrometry (MS) data generated during a biopharmaceutical manufacturing process. Data analysis in this context involves evaluating the integrity of chromatographic and MS datasets—specifically, identifying anomalies such as sensor drift, noise, missing values, and inconsistent metadata—to ensure that the data accurately reflects the biological and process characteristics of the product. The primary goal is to verify that the data is fit for purpose in supporting critical quality attributes (CQAs), such as purity, identity, and structural integrity of the biopharmaceutical product. This includes validating the consistency of sample tracking across files, detecting outliers in elution profiles or MS response, and ensuring that MS identifications are supported by reliable chromatographic co-elution. Appropriate data analysis includes SNR Calculation, Chromatographic Peak Shape Analysis, MS Signal Consistency Check, Metadata Alignment Validation, Missing Data Pattern Analysis, Chromatography-MS Integration, Sensor Drift Detection, Outlier Detection in Response Intensity, RT Alignment Validation, and Data Completeness Assessment. The context is that the datasets are large, high-resolution, and hierarchical, with multiple levels of sample and run metadata; therefore, analysis must account for the expected variability in elution times and response intensities due to process dynamics, while distinguishing true anomalies from normal process variation.",
        "Focus_Area_HPLC": "**Statement**\nHPLC data analysis in the context of biopharmaceutical manufacturing focuses on characterizing chromatographic profiles from fraction-level data to assess product quality, purity, and consistency. The primary objective is to identify and quantify target protein peaks, detect impurities or degradation products, and ensure process reproducibility across runs. This analysis relies on the integrity, quality, and relevance of chromatographic data, including UV absorbance, conductivity, pH, and volume measurements, to support process understanding, release testing, and regulatory compliance.\n\n**Suggested Data Analysis**\nPeak Detection, Peak Integration, Chromatographic Alignment, Time-Series Analysis, Multivariate Analysis (PCA), Process Variability Assessment, Fraction Classification, Elution Profile Comparison, Outlier Detection, Run-to-Run Consistency Analysis\n\n**Context**\nThe chromatography data is collected at the fraction level, with each row representing a fraction from a specific run and sample. Data is grouped by `Sample_Code`, `run_no`, and `Fraction_number`, enabling analysis at the fraction, run, and process stage levels. The data is dense and high-resolution, making it suitable for detailed peak profiling and comparison across stages (e.g., capture, polishing). Analysis should account for variations in load volume, pH, buffer composition, and resin type, as these are captured in the metadata and can influence elution behavior. Integration with metadata (e.g., `chromatography_stage`, `Resin_type`, `Start_pH`) is essential for contextual interpretation.",
        "Focus_Area_MS": "**Statement**\nMass spectrometry (MS) data analysis in biopharmaceutical manufacturing aims to identify and quantify proteins and peptides, assess their structural integrity, and detect critical quality attributes (CQAs) such as post-translational modifications (PTMs), deamidation, oxidation, and truncations. This analysis is essential for ensuring product consistency, safety, and efficacy, with the primary result being a reliable characterization of the biologic’s molecular identity and purity. Data quality, reliability, and relevance are ensured by validating spectral integrity, alignment across runs, and consistency with chromatographic elution profiles.\n\n**Suggested Data Analysis**\n- Peak Detection and Integration\n- Protein/Peptide Identification (via database search)\n- Quantification (label-free or label-based)\n- Post-Translational Modification (PTM) Analysis\n- Mass Error Assessment\n- Fragmentation Pattern Analysis\n- Multivariate Analysis (PCA, PLS-DA)\n- Batch Effect Correction\n- Temporal Profile Analysis (by chromatography stage)\n- Spectral Quality Control (SQC)\n\n**Context**\nThe MS data is collected across multiple chromatography stages and runs, with samples linked via `Sample_Code`, `run_code`, and `chromatography_stage`. Integration with chromatography data enables correlation of MS signals with elution behavior, enhancing confidence in identifications. The presence of missing values in key identification columns (e.g., `Molecule ID`, `Expected_mass_(Da)`) necessitates careful data filtering and quality control. Analysis must account for run-to-run variability and ensure that only high-quality, reproducible spectra are used for downstream characterization.",
        #-------- Context Variables for Coding Support System ----------------
        "Code_QA_Available": False,
        "Coding_RAG_QA": "",
        "RAG_Enhancements": "",
        "RAG_Enhancement_Available": False,

        #---------------------------------------------------------------------
    })
    # Chat to be tested:
    #--------------------

if __name__=="__main__":
    main()