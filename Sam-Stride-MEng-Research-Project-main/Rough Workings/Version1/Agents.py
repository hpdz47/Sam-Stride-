# ---------------------Imports -----------------------------
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function
from autogen import GroupChat, GroupChatManager
from autogen.agentchat.group.patterns import AutoPattern
import os
from autogen.agentchat import initiate_group_chat
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from pydantic import BaseModel, Field
from autogen.agentchat.group.guardrails import Guardrail, RegexGuardrail, GuardrailResult
from autogen.agentchat.group import AgentTarget
import json
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import AgentTarget, RevertToUserTarget, OnCondition, StringLLMCondition

from Guardrails import SafeLLMGuardrail
from LLM_Configurations import llm_config, llm_config_guardrail_agent
from Structured_Responses import GuardrailResponse, PolicyViolation
from Functions import *
#---------------------------------------------------------------------------------------------------------

def Create_Planner_Manager(context):
    # Create all Agents to be used in Nested Chat
    
    # Profiler Agent (For Data Usability)

    Profiler_Agent_Message= """
    You are a Data Profiler. Your job is to extract the correct metadata from the provided data using
    the tools you have access to. The tools you have access to are:

    - list_csv_files: List all CSV files in the current directory.
    - Processing: Extract metadata from a specified CSV file.

    Example Output:
    - This data contains 20 rows and 100 columns.
    - The columns are: "Time", "Intensity", "m/z", "Retention Time",
    - There are 5 missing values in the entire dataset which is 5 percent of the data.
    - The numerical metadata is: "Time", "Intensity", "m/z".
    - The categorical metadata is: "Sample ID".
    
    The Processing function automatically stores metadata in context, so once you call it successfully,
    report back the metadata summary.

    You must NEVER attempt to analyse the data yourself. Your only job is to provide
    informative metadata. When you have finished your response, you must end with:
    """

    Profiler_Agent= ConversableAgent(
        name="Profiler_Agent",
        llm_config=llm_config,
        system_message= Profiler_Agent_Message,
        human_input_mode="NEVER",
    )
    
    # Register functions using register_function with caller and executor
    register_function(
        list_csv_files,
        caller=Profiler_Agent,
        executor=Profiler_Agent,
        name="list_csv_files",
        description="List all CSV files in the current directory.",
    )
    
    register_function(
        Processing,
        caller=Profiler_Agent,
        executor=Profiler_Agent,
        name="Processing",
        description="Extract metadata from a specified CSV file and automatically store it in context variables.",
    )

    Planner_Message="""
    You are an expert in Chemical Processes and Proteomics. Your task
    is to supervise specialists in analyzing experimental data from High
    Performance Liquid Chromatography (HPLC) experiments and Mass Spectrometry
    (MS) experiments, which are used to analyse the purity for biopharmaceuticals.

    You have access to the following tools:
    - Get_Dataset_Metadata: Retrieves the stored dataset metadata from the Profiler Agent.
    You must use this as it provides context about what the data contains.

    You must do the follwing:
    - Plan the usability of the data without making assumptions that the experiment
    was correctly performed.
    - Come up with a plan to analyze the HPLC data for the HPLC Specialist Agent.
    - Come up with a plan to analyze the MS data for the MS Specialist Agent.

    Example Output:
    Usability Plan:
    1. Check for missing values.
    2. Perform [statistical test] to assess data quality.

    HPLC Analysis Plan:
    1. Process the HPLC data to identify ????
    2. Analyze ???? by writing code to do [analysis steps].

    MS Analysis Plan:
    1. Process the MS data to identify ????
    2. Analyze ???? by writing code to do [analysis steps].

    ???? and [] are placeholders. You must come up with more steps for a more detailed plan.
    
    """

    Planner_Agent= ConversableAgent(
        name="Planner_Agent",
        llm_config=llm_config,
        system_message=Planner_Message,
        human_input_mode="NEVER",
    )
    
    # Register functions using register_function with caller and executor
    register_function(
        Get_Dataset_Metadata,
        caller=Planner_Agent,
        executor=Planner_Agent,
        name="Get_Dataset_Metadata",
        description="Retrieves the stored dataset metadata from context variables.",
    )

    Plan_Reviewer_Message="""
    You are the Plan Reviewer. Your job is to use the following tool:
    - Get_Dataset_Metadata: Retrieves the stored dataset metadata from the Profiler Agent.

    You must use this output to review the analysis plans created by the Planner Agent. 
    
    You must ensure that the plans are feasible given the dataset metadata.
    If there are any issues with the plans, you must instruct the Planner_Agent to 
    re-write the plan. 

    Once all plans are appropriate, you must approve them and add the plans to the
    plan storage using the following tool:
    - Store_Plan: Stores the finalized analysis plan for future reference.
    """
    Plan_Reviewer_Agent= ConversableAgent(
        name="Plan_Reviewer_Agent",
        llm_config=llm_config,
        system_message=Plan_Reviewer_Message,
        human_input_mode="NEVER",
    )
    
    # Register functions using register_function with caller and executor
    register_function(
        Get_Dataset_Metadata,
        caller=Plan_Reviewer_Agent,
        executor=Plan_Reviewer_Agent,
        name="Get_Dataset_Metadata",
        description="Retrieves the stored dataset metadata from context variables.",
    )
    
    register_function(
        Store_Plan,
        caller=Plan_Reviewer_Agent,
        executor=Plan_Reviewer_Agent,
        name="Store_Plan",
        description="Stores the finalized analysis plan in context variables.",
    )

    # Create Nested Chat Settings
    Planning_Manager_Message="""
    You are the Planning Manager. You oversee the planning group. You receive instructions from
    the Data Supervisor and you must report all results back to the Data Supervisor Agent.
    """

    Planner_Manager= ConversableAgent(
        name="Planner_Manager",
        llm_config=llm_config,
        system_message=Planning_Manager_Message,
        human_input_mode="NEVER",
    )

    planning_groupchat = GroupChat(
        agents=[Planner_Agent,Plan_Reviewer_Agent,Profiler_Agent],
        max_round=20,
        speaker_selection_method="auto",
        send_introductions=False
    )

    planning_manager = GroupChatManager(
        groupchat=planning_groupchat,
        llm_config=llm_config,
    )

    planning_nested_chat=[{
        "recipient": Profiler_Agent,
        "message": "Profile the dataset and extract comprehensive metadata.",
        "max_turns": 5,
        "context_variables": context  # Pass context variables to nested chat
    },
        {"recipient": planning_manager,
        "message": "Create and review the analysis plan based on the dataset metadata provided by the Profiler Agent.",
        "max_turns": 20,
        "context_variables": context  # Pass context variables to nested chat
        }]

    Planner_Manager.register_nested_chats(
        chat_queue=planning_nested_chat,
        trigger=lambda sender: sender not in [Profiler_Agent, Planner_Agent,Plan_Reviewer_Agent]
    )
    return Planner_Manager,Profiler_Agent,Planner_Agent,Plan_Reviewer_Agent