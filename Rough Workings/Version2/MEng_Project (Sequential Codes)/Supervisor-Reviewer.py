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
import base64
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
import copy
import pprint
import re
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from autogen import Agent
#===========================================================
load_dotenv()
#================= Structured Responses ==============================
class Review_Step(BaseModel):
    Step_Number: int = Field(..., description="The step number within that plan section (e.g., 1, 2, 3...)")
    What_to_Improve: str = Field(..., description="A detailed description of what needs to be improved in this specific plan step. Reference any important context from the image summaries.")
    Critical_Comments: str = Field(..., description="Critical comments about this step. Should it be removed, modified, or is it missing? Detail the critiques here with specific recommendations.")
class Add_Step(BaseModel):
    Step_Number: int = Field(..., description="The step number within that plan section (e.g., 1, 2, 3...)")
    Description: str = Field(..., description="A detailed description of the step to be performed.")

class Reasoning_Review(BaseModel):
    Plan_Score: float = Field(..., description="Score out of 10 based on the quality of the plan and how well it was followed")
    Step_Reviews: List[Review_Step] = Field(..., description="List of reviews for specific plan steps that need improvement. Include all steps that have issues or need modification.")
    Steps_To_Add: List[Add_Step] = Field(..., description="List of steps to add to the plan. Include all steps that are missing from the plan.")
#=====================================================================

class Agent_Base():
    def __init__(self,name: str,llm_config: LLMConfig, system_message: str, Update_System_Message: Optional[str] = None, context_variables: Optional[ContextVariables] = None):
        self.name=name
        self.llm_config=llm_config # Composition Not Used here in case strucutred responses are needed for specific agents.
        self.system_message=system_message
        self.human_input_mode="NEVER" # Hard Coded as this is an AUTONMOUS system.
        if Update_System_Message:
            Updated_Message = [UpdateSystemMessage(Update_System_Message)]
        self.context_variables=context_variables
        self._agent=MultimodalConversableAgent(
            name=self.name,
            llm_config=self.llm_config,
            system_message=self.system_message,
            human_input_mode=self.human_input_mode,
            update_agent_state_before_reply=Updated_Message if Update_System_Message else None,
            context_variables=context_variables
        )

    @property
    def agent(self) -> MultimodalConversableAgent:
        return self._agent # Getter for retrieving the agent instance.

# Setup Specific Agents by Inheritance of main Conversable Agent Setup.
class VL_Reviewer_Agent(Agent_Base): # Visual Language model to review the data analysis results and provide a score out of 10.
    pass
class Reasoning_Reviewer_Agent(Agent_Base): # Reasoning model to review the plan and provide a score out of 10.
    pass
class RAG_User_Agent(Agent_Base): # RAG user agent to provide context to the VL reviewer.
    pass

# The Supervisor will be implemented as Logic, not as an agent. This is because there will be thresholds set based on the 
# LLM-as-a-Judge scores for the plan and data analysis results.

class VL_Reviewer_Chat:
    Current_Image_Idx=0
    Analysis_Types=""
    def __init__(self,context_variables: ContextVariables, Current_Image_Index: int, Analysis_Type: str, Max_Rounds: int):
        self.Max_Rounds=Max_Rounds
        self.context_variables=context_variables
        self.Current_Image_Index=Current_Image_Index
        VL_Reviewer_Chat.Current_Image_Idx=Current_Image_Index
        self.Analysis_Type=Analysis_Type
        VL_Reviewer_Chat.Analysis_Types=Analysis_Type

        LLM_Manager(LLM_Type="VL").Manage_VLLM()

        self.VL_Reviewer_Agent_System_Message="""
        You are a Reviewer of visual outputs from data analsyis.
        """
        if self.Analysis_Type=="Usability":
            self.VL_Reviewer_Agent_Updated_System_Message=f"""
            **ROLE**
            Your role is to review the visual outputs from data analysis that focuses on assessing the Usability and Quality of data.
            The output is from either High Performance Liquid Chromatography (HPLC) or Mass Spectrometry (MS) data analysis which are crucial in the 
            manufacturing and quality control of biopharmaceuticals. You MUST keep this context in minwhen reviewing the images.

            **Task**
            - You must provide a summary of the images that descibes what the image is, any interesting insights that can be drawn from the image about the 
            data quality and the usability of the data. Your comments can be constructive or crtical, but you must always justify your comments based on the data analysis shown in the images.
            You MUST try to identify any conclusions that can be drawn from the data analysis shown in the images about the proccess used to generate the data.
            - You MUST indicate what could be improved in further work.

            - Whilst you are not an expert in chemistry or biopharmaceutics, you will be provided with context about the subject area that you must use to 
            update your review.

            - You MUST provide a score out of 10 based on how well the visualisations have been produced and if it allows meaningful conclusions to be drawn about the usability and quality of the data.
            You must adapt the score in light of the context provided by the RAG agent. Any high score must be justified, so you must set a lower score until you have enough evidence to increase the score.
            The score must adhere to the scoring rubric that you have been provided with.

            - You MUST use the Score_Reviewer function to provide your summary and score.

            Scoring Rubric:
            Score = 10: The data analysis is perfect and is appropriate. It allows meaningful conclusions to be drawn from the data.
            Score = 7: The data analysis is reasonable and mostly appropriate. Some meaningful conclusions can be drawn, but there may be some minor improvements that could be made.
            Score = 4: The data analysis is completely superficial, but the visualisations are acceptable if applied to the correct data.
            Score = 1: The data analysis is completely invalid and inappropriate. It does not allow meaningful conclusions to be drawn from the data. The data analysis is superficial and no meaningful conclusions can be drawn from the data.
            Additionally the visualisations are poor and do not communicate any insights into the data.

            The scoring rubric is a guide only. It indicates the releative quality of the data analysis and visualisations. You must use your judgement to adapt the score based on the context provided by the RAG agent.
            The score can be set anywhere between 1 and 10.

            **Context**
            The current summary of the images that requires a re-write is: {{{self.Analysis_Type}_Image_Summary{self.Current_Image_Index}}}
            Factual Information that should be used to update the summary and score: {{Reviewer_RAG_Results{self.Current_Image_Index}}}
            The current score that needs to be reviewed is: {{{self.Analysis_Type}_Score{self.Current_Image_Index}}} out of 10.
            """
        elif self.Analysis_Type=="HPLC":
            self.VL_Reviewer_Agent_Updated_System_Message=f"""
            **ROLE**
            Your role is to review the visual outputs from data analysis that analysis High Peformance Liquid Chromatography (HPLC) data.
            This data analysis is crucial in the manufacturing of biopharmaceuticals. You MUST keep this context in minwhen reviewing the images.

            **Task**
            - You must provide a summary of the images that descibes what the image is, any interesting insights that can be drawn from the image about the 
            HPLC data and the manufacturing process. Your comments can be constructive or crtical, but you must always justify your comments based on the data analysis shown in the images.
            You MUST try to identify any conclusions that can be drawn from the data analysis shown in the images about the proccess used to generate the data.
            - You MUST indicate what could be improved in further work.

            - Whilst you are not an expert in chemistry or biopharmaceutics, you will be provided with context about the subject area that you must use to 
            update your review.

            - You MUST provide a score out of 10 based on how well the visualisations have been produced and if it allows meaningful conclusions to be drawn about the HPLC data.
            You must adapt the score in light of the context provided by the RAG agent. Any high score must be justified, so you must set a lower score until you have enough evidence to increase the score.
            The score must adhere to the scoring rubric that you have been provided with.

            - You MUST use the Score_Reviewer function to provide your summary and score.

            Scoring Rubric:
            Score = 10: The data analysis is perfect and is appropriate. It allows meaningful conclusions to be drawn from the data.
            Score = 7: The data analysis is reasonable and mostly appropriate. Some meaningful conclusions can be drawn, but there may be some minor improvements that could be made.
            Score = 4: The data analysis is completely superficial, but the visualisations are acceptable if applied to the correct data.
            Score = 1: The data analysis is completely invalid and inappropriate. It does not allow meaningful conclusions to be drawn from the data. The data analysis is superficial and no meaningful conclusions can be drawn from the data.
            Additionally the visualisations are poor and do not communicate any insights into the data.

            The scoring rubric is a guide only. It indicates the releative quality of the data analysis and visualisations. You must use your judgement to adapt the score based on the context provided by the RAG agent.
            The score can be set anywhere between 1 and 10.

            **Context**
            The current summary of the images that requires a re-write is: {{{self.Analysis_Type}_Image_Summary{self.Current_Image_Index}}}
            Factual Information that should be used to update the summary and score: {{Reviewer_RAG_Results{self.Current_Image_Index}}}
            The current score that needs to be reviewed is: {{{self.Analysis_Type}_Score{self.Current_Image_Index}}} out of 10.
            """
        elif self.Analysis_Type=="MS":
            self.VL_Reviewer_Agent_Updated_System_Message=f"""
            **ROLE**
            Your role is to review the visual outputs from data analysis that analysis Mass Spectrometry (MS) data.
            This data analysis is crucial in the manufacturing of biopharmaceuticals. You MUST keep this context in minwhen reviewing the images.

            **Task**
            - You must provide a summary of the images that descibes what the image is, any interesting insights that can be drawn from the image about the 
            MS data and the manufacturing process. Your comments can be constructive or crtical, but you must always justify your comments based on the data analysis shown in the images.
            You MUST try to identify any conclusions that can be drawn from the data analysis shown in the images about the proccess used to generate the data.
            - You MUST indicate what could be improved in further work.

            - Whilst you are not an expert in chemistry or biopharmaceutics, you will be provided with context about the subject area that you must use to 
            update your review.

            - You MUST provide a score out of 10 based on how well the visualisations have been produced and if it allows meaningful conclusions to be drawn about the MS data.
            You must adapt the score in light of the context provided by the RAG agent. Any high score must be justified, so you must set a lower score until you have enough evidence to increase the score.
            The score must adhere to the scoring rubric that you have been provided with.

            - You MUST use the Score_Reviewer function to provide your summary and score.

            Scoring Rubric:
            Score = 10: The data analysis is perfect and is appropriate. It allows meaningful conclusions to be drawn from the data.
            Score = 7: The data analysis is reasonable and mostly appropriate. Some meaningful conclusions can be drawn, but there may be some minor improvements that could be made.
            Score = 4: The data analysis is completely superficial, but the visualisations are acceptable if applied to the correct data.
            Score = 1: The data analysis is completely invalid and inappropriate. It does not allow meaningful conclusions to be drawn from the data. The data analysis is superficial and no meaningful conclusions can be drawn from the data.
            Additionally the visualisations are poor and do not communicate any insights into the data.

            The scoring rubric is a guide only. It indicates the releative quality of the data analysis and visualisations. You must use your judgement to adapt the score based on the context provided by the RAG agent.
            The score can be set anywhere between 1 and 10.

            **Context**
            The current summary of the images that requires a re-write is: {{{self.Analysis_Type}_Image_Summary{self.Current_Image_Index}}}
            Factual Information that should be used to update the summary and score: {{Reviewer_RAG_Results{self.Current_Image_Index}}}
            The current score that needs to be reviewed is: {{{self.Analysis_Type}_Score{self.Current_Image_Index}}} out of 10.
            """
        else:
            raise ValueError("Invalid Analysis_Type provided. Must be 'Usability', 'HPLC', or 'MS'.")
    
        self.RAG_User_Agent_System_Message="""
        You are a RAG agent that must call the RAG_Tool function with a query to provide context to the VL reviewer.
        The query must be based on the summary provided by the VL reviewer agent."""

        if self.Analysis_Type=="Usability":
            self.RAG_User_Agent_Updated_System_Message=f"""
            **ROLE**
            Your role is to query a database of factual information about High Performance Liquid Chromatography (HPLC) and Mass Spectrometry (MS) data.
            You MUST focus on asking questions that would allow you to understand how to determine if the data quality is acceptable or usable
            for biopharmaceutical manufacturing and quality control. You will be provided with a summary of the data analysis that has been performed
            and you MUST attempt to learn more about the topic area to inform any judgements.

            **Task**
            - You MUST ALWAYS come up with your own queries based on the summary that has been provided to you about the data
            analysis that has already been carried out.
            - The function you have access to is called RAG. This will allow you to query a database of factual information.
            - You have been provided with some example query styles. However, you MUST use your own judgement to decide what
            queries are relevant to help you understand the data analysis that has been performed and how to interpret it. Avoid
            duplicate queries or any queres that are too similar in nature to previous queries. This will ensure a broader overview of the topic
            area is covered.

            Example Query Styles (Illustrative not Exhaustive):
            - What is the purpose of HPLC and MS data in biopharmaceutical manufacturing?
            - Is it appropriate or standard practice to plot X vs Y in Z analysis?
            - Is it acceptable to use a linear regression model to predict Y from X in Z analysis?
            - Is the observed trend between X and Y expected in Z topic?
            - What is the expected outcome of X analysis?
            - Should X analysis be performed on the raw data or the processed data?
            - What determines the quality of HPLC or MS data in biopharmaceutical manufacturing based on X analysis technique?

            **Context**
            The current summary of the images is: {{{self.Analysis_Type}_Image_Summary{self.Current_Image_Index}}}
            The current score is: {{{self.Analysis_Type}_Score{self.Current_Image_Index}}}
            The current RAG queries and results are: {{Reviewer_RAG_Results{self.Current_Image_Index}}}
            """
        elif self.Analysis_Type=="HPLC":
            self.RAG_User_Agent_Updated_System_Message=f"""
            **ROLE**
            Your role is to query a database of factual information about High Performance Liquid Chromatography (HPLC) data.
            You MUST focus on asking questions that would allow you to understand how to determine if the HPLC data analysis is acceptable to 
            provide conclusions about the bopharmaceutical manufacturing process and its quality. You will be provided with a summary of the data analysis that has been performed
            and you MUST attempt to learn more about the topic area to inform any judgements.

            **Task**
            - You MUST ALWAYS come up with your own queries based on the summary that has been provided to you about the data
            analysis that has already been carried out.
            - The function you have access to is called RAG. This will allow you to query a database of factual information.
            - You have been provided with some example query styles. However, you MUST use your own judgement to decide what
            queries are relevant to help you understand the data analysis that has been performed and how to interpret it. Avoid
            duplicate queries or any queres that are too similar in nature to previous queries. This will ensure a broader overview of the topic
            area is covered.

            Example Query Styles (Illustrative not Exhaustive):
            - What is the purpose of HPLC data and analysis in biopharmaceutical manufacturing?
            - Is it appropriate or standard practice to plot X vs Y in Z analysis?
            - Is it acceptable to use a linear regression model to predict Y from X in Z analysis?
            - Is the observed trend between X and Y expected in Z topic?
            - What is the expected outcome of X analysis?
            - Should X analysis be performed on the raw data or the processed data?
            -How dou you interpet HPLC data in biopharmaceutical manufacturing based on X analysis technique?
            
            **Context**
            The current summary of the images is: {{{self.Analysis_Type}_Image_Summary{self.Current_Image_Index}}}
            The current score is: {{{self.Analysis_Type}_Score{self.Current_Image_Index}}}
            The current RAG queries and results are: {{Reviewer_RAG_Results{self.Current_Image_Index}}}
            """
        elif self.Analysis_Type=="MS":
            self.RAG_User_Agent_Updated_System_Message=f"""
            **ROLE**
            Your role is to query a database of factual information about Mass Spectrometry (MS) data.
            You MUST focus on asking questions that would allow you to understand how to determine if the MS data analysis is acceptable to 
            provide conclusions about the bopharmaceutical manufacturing process and its quality. You will be provided with a summary of the data analysis that has been performed
            and you MUST attempt to learn more about the topic area to inform any judgements.

            **Task**
            - You MUST ALWAYS come up with your own queries based on the summary that has been provided to you about the data
            analysis that has already been carried out.
            - The function you have access to is called RAG. This will allow you to query a database of factual information.
            - You have been provided with some example query styles. However, you MUST use your own judgement to decide what
            queries are relevant to help you understand the data analysis that has been performed and how to interpret it. Avoid
            duplicate queries or any queres that are too similar in nature to previous queries. This will ensure a broader overview of the topic
            area is covered.

            Example Query Styles (Illustrative not Exhaustive):
            - What is the purpose of MS data and analysis in biopharmaceutical manufacturing?
            - Is it appropriate or standard practice to plot X vs Y in Z analysis?
            - Is it acceptable to use a linear regression model to predict Y from X in Z analysis?
            - Is the observed trend between X and Y expected in Z topic?
            - What is the expected outcome of X analysis?
            - Should X analysis be performed on the raw data or the processed data?
            -How dou you interpet MS data in biopharmaceutical manufacturing based on X analysis technique?
            
            **Context**
            The current summary of the images is: {{{self.Analysis_Type}_Image_Summary{self.Current_Image_Index}}}
            The current score is: {{{self.Analysis_Type}_Score{self.Current_Image_Index}}}
            The current RAG queries and results are: {{Reviewer_RAG_Results{self.Current_Image_Index}}}
            """
        else:
            raise ValueError("Invalid Analysis_Type provided. Must be 'Usability', 'HPLC', or 'MS'.")
        VL_Name="VL_Reviewer_Agent"
        RAG_User_Name="RAG_User_Agent"
        VL_Reviewer_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="VL").build_config()
        RAG_User_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.4,enable_thinking=False,LLM_Type="VL").build_config()

        self.VL_Reviewer=VL_Reviewer_Agent(VL_Name,VL_Reviewer_LLM_Config,self.VL_Reviewer_Agent_System_Message,self.VL_Reviewer_Agent_Updated_System_Message,self.context_variables)
        self.RAG_User_Agent=RAG_User_Agent(RAG_User_Name,RAG_User_LLM_Config,self.RAG_User_Agent_System_Message,self.RAG_User_Agent_Updated_System_Message,self.context_variables)
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="VL").build_config()

        # Handoffs ---------------------------------
        self.VL_Reviewer.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.RAG_User_Agent.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Score_Available} == True")
                )
            )
        )

        self.VL_Reviewer.agent.handoffs.set_after_work(AgentTarget(self.VL_Reviewer.agent))

        self.RAG_User_Agent.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.VL_Reviewer.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Reviewer_RAG_Used} == True")
                )
            )
        )

        self.RAG_User_Agent.agent.handoffs.set_after_work(AgentTarget(self.RAG_User_Agent.agent))

    def run_Conversation(self):
        # Transforming Message History (To Limit)----
        context_handling = transform_messages.TransformMessages(
            transforms=[transforms.MessageHistoryLimiter(max_messages=3)])
        context_handling.add_to_agent(self.VL_Reviewer.agent)
        context_handling.add_to_agent(self.RAG_User_Agent.agent)
        #--------------------------------------------
        # Resetting Shared Context Variables:
        self.context_variables["RAG_Number"]=0
        self.context_variables["RAG_Approval"]=False
        self.context_variables["Score_Approval"]=False
        self.context_variables["Reviewer_RAG_Used"]=False
        self.context_variables["Score_Available"]=False

        agents=[self.VL_Reviewer.agent,self.RAG_User_Agent.agent]
        pattern=DefaultPattern(
        initial_agent=self.VL_Reviewer.agent,
        agents=agents,
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables)

        register_function(
            RAG,
            caller=self.RAG_User_Agent.agent,
            executor=self.RAG_User_Agent.agent,
            name="RAG",
            description="Query the RAG tool for context to the VL reviewer."
        )

        register_function(
            Score_Reviewer,
            caller=self.VL_Reviewer.agent,
            executor=self.VL_Reviewer.agent,
            name="Score_Reviewer",
            description="Review the score and provide a score out of 10 based on the quality of the data analysis."
        )

        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages=f"""Analyse the images and provide a description of what is in the images as well as a score between 1 and 10 based on the quality of the data analysis.
            <img {self.context_variables[f"{self.Analysis_Types}_image{self.Current_Image_Index}"]}>
            """, 
            max_rounds=self.Max_Rounds,
        )
        
        return result, ctx

def RAG(first_query: Annotated[str, "The query to the RAG"],second_query: Annotated[str, "The query to the RAG"],third_query: Annotated[str, "The query to the RAG"], context_variables: ContextVariables) -> ReplyResult:
    results1 = RAG_Tool(first_query)
    results2 = RAG_Tool(second_query)
    results3 = RAG_Tool(third_query)

    # Limit to most recent 3-5 queries to prevent context overflow
    MAX_RAG_HISTORY = 3
    context_variables[f"Reviewer_RAG_Results{VL_Reviewer_Chat.Current_Image_Idx}"].append({"Query": first_query, "Results": results1})
    context_variables[f"Reviewer_RAG_Results{VL_Reviewer_Chat.Current_Image_Idx}"].append({"Query": second_query, "Results": results2})
    context_variables[f"Reviewer_RAG_Results{VL_Reviewer_Chat.Current_Image_Idx}"].append({"Query": third_query, "Results": results3})
    
    # Keep only the most recent queries
    if len(context_variables[f"Reviewer_RAG_Results{VL_Reviewer_Chat.Current_Image_Idx}"]) > MAX_RAG_HISTORY:
        context_variables[f"Reviewer_RAG_Results{VL_Reviewer_Chat.Current_Image_Idx}"] = context_variables[f"Reviewer_RAG_Results{VL_Reviewer_Chat.Current_Image_Idx}"][-MAX_RAG_HISTORY:]
    
    context_variables["RAG_Number"] += 1
    context_variables["Reviewer_RAG_Used"] = True
    
    reminder= f""" The image to be analysed is: <img {context_variables[f"{VL_Reviewer_Chat.Analysis_Types}_image{VL_Reviewer_Chat.Current_Image_Idx}"]}>"""

    #Reset flags for VL Reviewer agent.
    context_variables["Score_Available"] = False

    return ReplyResult(
        message=f"RAG results saved successfully, ready for score review. {reminder}",
        context_variables=context_variables,
    )

def Score_Reviewer(Image_Summary: Annotated[str, "The summary of the data analysis image with any critical comments or suggestions for improvement. You must always justify the score you provide based on the context provided by the RAG agent."], Score: Annotated[float, "The score out of 10 based on the quality of the data analysis. 10 is the best score, 1 is the worst score. Scores can be non-integer values."],context_variables: ContextVariables) -> ReplyResult:

    context_variables[f"{VL_Reviewer_Chat.Analysis_Types}_Score{VL_Reviewer_Chat.Current_Image_Idx}"] = Score
    context_variables[f"{VL_Reviewer_Chat.Analysis_Types}_Image_Summary{VL_Reviewer_Chat.Current_Image_Idx}"] = Image_Summary
    context_variables["Score_Available"] = True

    # Reset flags for RAG agent.
    context_variables["Reviewer_RAG_Used"] = False

    # Reset Flags for Reasoning Reviewer In case Supervisor-Reviewer used again:
    context_variables["Final_Reviews_Submitted"]=False

    return ReplyResult(
        message=f"Score reviewed successfully.",
        context_variables=context_variables,
    )

class Reasoning_Reviewer_Chat:
    Type_of_Analysis=""
    def __init__(self,context_variables: ContextVariables, Analysis_Type: str):
        self.Analysis_Type=Analysis_Type
        Reasoning_Reviewer_Chat.Type_of_Analysis=Analysis_Type
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables=context_variables
        for ID in range(1,VL_Reviewer_Chat.Current_Image_Idx+1): # Current Image Idx will be the maximum number of images available. +1 due to how the range() function works.
            summary_key=f"{self.Analysis_Type}_Image_Summary{ID}"
            if summary_key in self.context_variables:
                self.context_variables["Combined_Summary"]=f"""{self.context_variables["Combined_Summary"]} {self.context_variables[f"{self.Analysis_Type}_Image_Summary{ID}"]}""" # Combining all summaries into 1 large report.
            else:
                self.context_variables["Combined_Summary"]="No summaries available."
        


        self.Reasoning_Reviewer_Agent_System_Message="""
        You are a Reviewer Agent.
        """
        if self.Analysis_Type=="Usability":
            self.Reasoning_Reviewer_Agent_Updated_System_Message=f"""
            **ROLE**
            Your role is to review how effective a data analysis plan is. You must do this in light of the results from
            data analysis to determine if the plan led to meanignful results. The topic area to focus on is assessing
            the usability and data quality of High Performance Liquid Chromatography (HPLC) and Mass Spectrometry (MS) data.
            This is crucial in the manufacturing and quality control of biopharmaceuticals. You MUST keep this context in mind
            when reviewing the plan and data analysis summaries. When you assign scores, they must earn the score you give, otherwise the score must be
            set low. High scores must be justified. When you review steps or suggest a new step, you must always determine if this is appropriate for the data that is
            provided. To understand this, you have access to a summary of the data that is present.

            **Task**
            - You have been provdied with both the plan and the combined summary of all of the data analysis results. You must review
            both in order to assess what has worked well and what needs to be improved.
            - If the data analysis has worked well, you must simply provide a review of this, and try to summarise if anything
            would make this even better.
            - If the data analysis has not worked well, you must be critical and provide detailed feedback on what needs to be improved.
            - You MUST provide a score out of 10 based on how good the plan is and how well it was followed to produce meaningful data analysis results.
            You have been provided with a scoring rubric that you must use to aid your scoring decision. The score must be conservative to prevent giving a false
            impression of good data analysis, unless a high score is deserved.
            - In your reviews/ criticism, you must also suggest any potential steps to add to the plan that would improve the data analysis.
            - To submit your review, you MUST use the Submit_Reviews function.

            ** Crucial Output Format**
            - The plan must instruct the output to be 1 markdown file called "Usability.md" that contains the summary of data analysis performed. Images must be
            saved as images and references to the images must be made in the markdown files. csv files are allowed for intermediate working only.

            The structure of your review is as follows:
            - Plan_Score: A score out of 10
            - Step_Reviews: A list of Review_Step objects for each plan step that needs improvement. Each Review_Step must specify:
            * Step_Number: The step number within that section
            * What_to_Improve: Detailed description of what needs improvement, referencing image summaries
            * Critical_Comments: Whether the step should be removed, modified, or what's missing
            - Steps_To_Add: A list of Add_Step objects for any missing steps that should be added to the plan. Each Add_Step must specify:
            * Step_Number: The step number (can be inserted between existing steps or at the end)
            * Description: Detailed description of what the step should do

            Scoring Rubric:
            Score = 10: The data analysis is perfect and is appropriate. It allows meaningful conclusions to be drawn from the data.
            Score = 7: The data analysis is reasonable and mostly appropriate. Some meaningful conclusions can be drawn, but there may be some minor improvements that could be made.
            Score = 4: The data analysis is ok but far from ideal. Some meaningful conclusions can be drawn, but there are some major issues that need to be addressed either with the data analysis approach or the visualisation of results.
            Score = 1: The data analysis is completely invalid and inappropriate. It does not allow meaningful conclusions to be drawn from the data. The data analysis is superficial and no meaningful conclusions can be drawn from the data.

            **Context**
            The Plan is provided by: {{Usability_Plan}}
            The Combined Summary of the data analysis is provided by: {{Combined_Summary}}
            The data that is present is provided by: {{Data_Discovery_Summary}}
            """
        elif self.Analysis_Type=="HPLC":
            self.Reasoning_Reviewer_Agent_Updated_System_Message=f"""
            **ROLE**
            Your role is to review how effective a data analysis plan is. You must do this in light of the results from
            data analysis to determine if the plan led to meanignful results. The topic area to focus on is assessing
            High Performance Liquid Chromatography (HPLC) data anlysis and the conclusion about the manufacturing process that can be drawn from it.
            This is crucial in the manufacturing and quality control of biopharmaceuticals. You MUST keep this context in mind
            when reviewing the plan and data analysis summaries. When you assign scores, they must earn the score you give, otherwise the score must be
            set low. High scores must be justified. When you review steps or suggest a new step, you must always determine if this is appropriate for the data that is
            provided. To understand this, you have access to a summary of the data that is present. If the dataset is missing the data required to perform data analysis, then the plan must reflect this and not attempt to do any analysis on other
            data.

            **Task**
            - You have been provdied with both the plan and the combined summary of all of the data analysis results. You must review
            both in order to assess what has worked well and what needs to be improved.
            - If the data analysis has worked well, you must simply provide a review of this, and try to summarise if anything
            would make this even better.
            - If the data analysis has not worked well, you must be critical and provide detailed feedback on what needs to be improved.
            - You MUST provide a score out of 10 based on how good the plan is and how well it was followed to produce meaningful data analysis results.
            You have been provided with a scoring rubric that you must use to aid your scoring decision. The score must be conservative to prevent giving a false
            impression of good data analysis, unless a high score is deserved.
            - In your reviews/ criticism, you must also suggest any potential steps to add to the plan that would improve the data analysis.
            - To submit your review, you MUST use the Submit_Reviews function.
            - If the dataset is missing the data required to perform data analysis, then the plan must reflect this and not attempt to do any analysis on other
            data.

            ** Crucial Output Format**
            - The plan must instruct the output to be 1 markdown file called "HPLC.md" that contains the summary of data analysis performed. Images must be
            saved as images and references to the images must be made in the markdown files. csv files are allowed for intermediate working only.

            The structure of your review is as follows:
            - Plan_Score: A score out of 10
            - Step_Reviews: A list of Review_Step objects for each plan step that needs improvement. Each Review_Step must specify:
            * Step_Number: The step number within that section
            * What_to_Improve: Detailed description of what needs improvement, referencing image summaries
            * Critical_Comments: Whether the step should be removed, modified, or what's missing
            - Steps_To_Add: A list of Add_Step objects for any missing steps that should be added to the plan. Each Add_Step must specify:
            * Step_Number: The step number (can be inserted between existing steps or at the end)
            * Description: Detailed description of what the step should do

            Scoring Rubric:
            Score = 10: The data analysis is perfect and is appropriate. It allows meaningful conclusions to be drawn from the data.
            Score = 7: The data analysis is reasonable and mostly appropriate. Some meaningful conclusions can be drawn, but there may be some minor improvements that could be made.
            Score = 4: The data analysis is ok but far from ideal. Some meaningful conclusions can be drawn, but there are some major issues that need to be addressed either with the data analysis approach or the visualisation of results.
            Score = 1: The data analysis is completely invalid and inappropriate. It does not allow meaningful conclusions to be drawn from the data. The data analysis is superficial and no meaningful conclusions can be drawn from the data.

            **Context**
            The Plan is provided by: {{HPLC_Analysis_Plan}}
            The Combined Summary of the data analysis is provided by: {{Combined_Summary}}
            The data that is present is provided by: {{Data_Discovery_Summary}}
            """
        elif self.Analysis_Type=="MS":
            self.Reasoning_Reviewer_Agent_Updated_System_Message=f"""
            **ROLE**
            Your role is to review how effective a data analysis plan is. You must do this in light of the results from
            data analysis to determine if the plan led to meanignful results. The topic area to focus on is assessing
            Mass Spectrometry (MS) data anlysis and the conclusion about the manufacturing process that can be drawn from it.
            This is crucial in the manufacturing and quality control of biopharmaceuticals. You MUST keep this context in mind
            when reviewing the plan and data analysis summaries. When you assign scores, they must earn the score you give, otherwise the score must be
            set low.  High scores must be justified. When you review steps or suggest a new step, you must always determine if this is appropriate for the data that is
            provided. To understand this, you have access to a summary of the data that is present. If the dataset is missing the data required to perform data analysis, then the plan must reflect this and not attempt to do any analysis on other
            data.

            **Task**
            - You have been provdied with both the plan and the combined summary of all of the data analysis results. You must review
            both in order to assess what has worked well and what needs to be improved.
            - If the data analysis has worked well, you must simply provide a review of this, and try to summarise if anything
            would make this even better.
            - If the data analysis has not worked well, you must be critical and provide detailed feedback on what needs to be improved.
            - You MUST provide a score out of 10 based on how good the plan is and how well it was followed to produce meaningful data analysis results.
            You have been provided with a scoring rubric that you must use to aid your scoring decision. The score must be conservative to prevent giving a false
            impression of good data analysis, unless a high score is deserved.
            - In your reviews/ criticism, you must also suggest any potential steps to add to the plan that would improve the data analysis.
            - To submit your review, you MUST use the Submit_Reviews function.
            - If the dataset is missing the data required to perform data analysis, then the plan must reflect this and not attempt to do any analysis on other
            data.

            ** Crucial Output Format**
            - The plan must instruct the output to be 1 markdown file called "MS.md" that contains the summary of data analysis performed. Images must be
            saved as images and references to the images must be made in the markdown files. csv files are allowed for intermediate working only.

            The structure of your review is as follows:
            - Plan_Score: A score out of 10
            - Step_Reviews: A list of Review_Step objects for each plan step that needs improvement. Each Review_Step must specify:
            * Step_Number: The step number within that section
            * What_to_Improve: Detailed description of what needs improvement, referencing image summaries
            * Critical_Comments: Whether the step should be removed, modified, or what's missing
            - Steps_To_Add: A list of Add_Step objects for any missing steps that should be added to the plan. Each Add_Step must specify:
            * Step_Number: The step number (can be inserted between existing steps or at the end)
            * Description: Detailed description of what the step should do

            Scoring Rubric:
            Score = 10: The data analysis is perfect and is appropriate. It allows meaningful conclusions to be drawn from the data.
            Score = 7: The data analysis is reasonable and mostly appropriate. Some meaningful conclusions can be drawn, but there may be some minor improvements that could be made.
            Score = 4: The data analysis is ok but far from ideal. Some meaningful conclusions can be drawn, but there are some major issues that need to be addressed either with the data analysis approach or the visualisation of results.
            Score = 1: The data analysis is completely invalid and inappropriate. It does not allow meaningful conclusions to be drawn from the data. The data analysis is superficial and no meaningful conclusions can be drawn from the data.

            **Context**
            The Plan is provided by: {{Mass_Spectrometry_Plan}}
            The Combined Summary of the data analysis is provided by: {{Combined_Summary}}
            The data that is present is provided by: {{Data_Discovery_Summary}}
            """
        Reasoning_Reviewer_Name="Reasoning_Reviewer_Agent"
        Reasoning_Reviewer_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.4,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Reasoning_Reviewer_Agent=Reasoning_Reviewer_Agent(Reasoning_Reviewer_Name,Reasoning_Reviewer_LLM_Config,self.Reasoning_Reviewer_Agent_System_Message,self.Reasoning_Reviewer_Agent_Updated_System_Message,self.context_variables)
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="VL").build_config()

        # Handoffs ---------------------------------
        self.Reasoning_Reviewer_Agent.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Final_Reviews_Submitted} == True")
                )
            )
        )

        self.Reasoning_Reviewer_Agent.agent.handoffs.set_after_work(AgentTarget(self.Reasoning_Reviewer_Agent.agent))

    def run_Conversation(self):
        agents=[self.Reasoning_Reviewer_Agent.agent]
        pattern=DefaultPattern(
            initial_agent=self.Reasoning_Reviewer_Agent.agent,
            agents=agents,
            group_manager_args={"llm_config": self.Chat_Config},
            context_variables=self.context_variables)
        
        register_function(
            Submit_Reviews,
            caller=self.Reasoning_Reviewer_Agent.agent,
            executor=self.Reasoning_Reviewer_Agent.agent,
            name="Submit_Reviews",
            description="Submit the structured reviews (Reasoning_Review) and score for the plan. Must provide Reasoning_Review with Plan_Score, Overall_Summary, Step_Reviews for problematic steps, and Steps_To_Add for missing steps."
        )

        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages=f"""Review the information provided and provide a review and a score.""",
            max_rounds=5,
        )
        return result, ctx

    
def Submit_Reviews(
    Review: Annotated[Reasoning_Review, "The structured review of the plan and data analysis. Must include Plan_Score, Overall_Summary, Step_Reviews for steps needing improvement, and Steps_To_Add for missing steps."], 
    context_variables: ContextVariables) -> ReplyResult:
    # Store the structured review as JSON
    if Reasoning_Reviewer_Chat.Type_of_Analysis=="Usability":
        context_variables["Usability_Plan_Review"] = Review.Step_Reviews
        context_variables["Usability_Steps_To_Add"] = Review.Steps_To_Add
        context_variables["Usability_Plan_Score"] = Review.Plan_Score
    elif Reasoning_Reviewer_Chat.Type_of_Analysis=="HPLC":
        context_variables["HPLC_Analysis_Plan_Review"] = Review.Step_Reviews
        context_variables["HPLC_Analysis_Steps_To_Add"] = Review.Steps_To_Add
        context_variables["HPLC_Plan_Score"] = Review.Plan_Score
    elif Reasoning_Reviewer_Chat.Type_of_Analysis=="MS":
        context_variables["Mass_Spectrometry_Plan_Review"] = Review.Step_Reviews
        context_variables["Mass_Spectrometry_Steps_To_Add"] = Review.Steps_To_Add
        context_variables["MS_Plan_Score"] = Review.Plan_Score
    else:
        raise ValueError("Invalid Analysis_Type in Reasoning_Reviewer_Chat. Must be 'Usability', 'HPLC', or 'MS'.")

    context_variables["Final_Reviews_Submitted"] = True

    context_variables["last_speaker"] = "Reasoning_Reviewer"

    # Reset flags for next instnaitation of Supervisor-Reviewer Chat:
    context_variables["Score_Available"] = False
    context_variables["Reviewer_RAG_Used"] = False
    
    return ReplyResult(
        message=f"Reviews submitted successfully. Plan Score: {Review.Plan_Score}. {len(Review.Step_Reviews)} step(s) to review, {len(Review.Steps_To_Add)} step(s) to add. Proceed to the next iteration.",
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
    
    correct_format="""{"Plan_Score": 0.0,"Step_Reviews": [{"Step_Number": 1,"What_to_Improve": "[Describe clearly what aspect of this step needs improvement. Reference specific issues observed, such as incorrect assumptions, missing context, poor data presentation, or inappropriate analytical approach.]","Critical_Comments": "[Provide critical evaluation of this step. State whether it should be modified, expanded, or partially removed. Include specific, actionable recommendations for improvement.]"},{"Step_Number": 2,"What_to_Improve": "[Explain what is insufficient, unclear, or misleading in this step. Reference any methodological, interpretative, or compliance-related gaps.]","Critical_Comments": "[Detail why this step is problematic in its current form. Indicate required changes and explain the impact of not addressing them.]"}],"Steps_To_Add": [{"Step_Number": 3,"Description": "[Describe the missing step that should be added to the plan. Explain what should be done, why it is necessary, and how it improves completeness, data quality, or compliance.]"},{"Step_Number": 4,"Description": "[Describe an additional step required for robustness, validation, documentation, or review. Be specific about expected outputs or acceptance criteria.]"}]}"""

    if sender.context_variables["last_speaker"] == "Reasoning_Reviewer":
        if isinstance(message, dict):
            message["content"]=f"""{content} \n\n Reminder of the correct response format: {correct_format}"""
            return message
        else:
            return f"""{content} \n\n Reminder of the correct response format: {correct_format}"""
    else:
        pass
    
    return message



class Supervisor_Reviewer_Chat:
    def __init__(self,context_variables: ContextVariables, Analysis_Type: str, Max_Rounds: int):
        self.context_variables=context_variables
        self.Analysis_Type=Analysis_Type
        self.Max_Rounds=Max_Rounds
    def run_Conversation(self):
        if self.Analysis_Type=="Usability":
            results_dir=Path("./Data_Results/Usability_Results")
        elif self.Analysis_Type=="HPLC":
            results_dir=Path("./Data_Results/HPLC_Results")
        elif self.Analysis_Type=="MS":
            results_dir=Path("./Data_Results/MS_Results")
        else:
            raise ValueError(f"Invalid Analysis_Type: {self.Analysis_Type}. Must be 'Usability', 'HPLC', or 'MS'.")
        # Define image extensions to look for
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif']
        i=0
        for idx, file in enumerate(results_dir.iterdir(),start=1):
            # Check if file is an image (has image extension)
            if file.is_file() and file.suffix.lower() in image_extensions:
                i+=1
                # Get the full path to the image
                image_path = f"./{results_dir}/{file.name}"
                self.context_variables[f"{self.Analysis_Type}_Image_Summary{i}"]=""
                self.context_variables[f"{self.Analysis_Type}_Score{i}"]=0
                self.context_variables[f"Reviewer_RAG_Results{i}"]=[]
                self.context_variables[f"{self.Analysis_Type}_image{i}"]=image_path
                vl_reviewer_chat=VL_Reviewer_Chat(self.context_variables, i, self.Analysis_Type, self.Max_Rounds)
                vl_reviewer_chat.run_Conversation()
        reasoning_reviewer_chat=Reasoning_Reviewer_Chat(self.context_variables, self.Analysis_Type)
        reasoning_reviewer_chat.run_Conversation()


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
    })
    with open("Debug.json", "r") as f:
        CV=json.load(f)
        # The data is nested under "data" key
        data_section = CV.get("data", {})
        metadata_value = data_section.get("metadata", {})
        usability_plan_value = data_section.get("Usability_Plan", {})
        hplc_plan_value = data_section.get("HPLC_Analysis_Plan", {})
        ms_plan_value = data_section.get("Mass_Spectrometry_Plan", {})
        context_variables["metadata"]=metadata_value
        context_variables["Usability_Plan"]=usability_plan_value
        context_variables["HPLC_Analysis_Plan"]=hplc_plan_value
        context_variables["Mass_Spectrometry_Plan"]=ms_plan_value
        
        # Validate that we have actual data
        if not metadata_value:
            raise ValueError("metadata is empty or None in Planning_Conversation.json")
        #if not plan_value:
            #raise ValueError("Plan is empty or None in Planning_Conversation.json")


    Supervisor_Reviews1=Supervisor_Reviewer_Chat(context_variables=context_variables, Analysis_Type="Usability", Max_Rounds=10)
    Supervisor_Reviews1.run_Conversation()
    Supervisor_Reviews2=Supervisor_Reviewer_Chat(context_variables=context_variables, Analysis_Type="HPLC", Max_Rounds=10)
    Supervisor_Reviews2.run_Conversation()
    Supervisor_Reviews3=Supervisor_Reviewer_Chat(context_variables=context_variables, Analysis_Type="MS", Max_Rounds=10)
    Supervisor_Reviews3.run_Conversation()
    
    with open("Supervisor_Reviewer_Conversation.json", "w") as f:
        json.dump(context_variables.model_dump(), f, indent=2)

if __name__ == "__main__":
    main()