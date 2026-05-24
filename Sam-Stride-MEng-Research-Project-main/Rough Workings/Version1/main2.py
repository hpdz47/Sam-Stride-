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
#

from LLM_Configurations import llm_config

from Functions import *
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

context_variables= ContextVariables(data={
    "Metadata": None,
    "Plan": []

})


# ======================== AGENTS =====================================

Data_Supervisor_Message="""
You are a Supervisor. You must ask the Plan_Manager Agent to create Analysis Plans.
"""

Data_Supervisor= ConversableAgent(
    name="Data_Supervisor",
    llm_config=llm_config,
    system_message=Data_Supervisor_Message,
    human_input_mode="NEVER",
    context_variables=context_variables
)

Planner_Message="""
You are an expert in Chemical Processes and Proteomics. Your task
is to supervise specialists in analyzing experimental data from High
Performance Liquid Chromatography (HPLC) experiments and Mass Spectrometry
(MS) experiments, which are used to analyse the purity for biopharmaceuticals.

You must do the following:
- First, call the Get_Dataset_Metadata tool to retrieve the dataset information (DO NOT pass any arguments - it retrieves automatically from context)
- Plan the usability of the data without making assumptions that the experiment was correctly performed
- Come up with a plan to analyze the HPLC data for the HPLC Specialist Agent
- Come up with a plan to analyze the MS data for the MS Specialist Agent

You must use the following tools to get the correct information for your plan:
- Get_Dataset_Metadata: Retrieves the stored dataset metadata from the Profiler Agent.
"""

Planner_Agent= ConversableAgent(
    name="Planner_Agent",
    llm_config=llm_config,  # Start with standard config
    system_message=Planner_Message,
    human_input_mode="NEVER",  # Never ask for human input
    context_variables=context_variables
)



# Profiler Agent (For Data Usability)

Profiler_Agent_Message= """
You are a Data Profiler. Your job is to extract the correct metadata from the provided data using
the tools you have access to. The tools you have access to are:

- Profile_Check: Extract metadata from a specified CSV file and store for other agents.

Example Output:
- This data contains 20 rows and 100 columns.
- The columns are: "Time", "Intensity", "m/z", "Retention Time",
- There are 5 missing values in the entire dataset which is 5 percent of the data.
- The numerical metadata is: "Time", "Intensity", "m/z".
- The categorical metadata is: "Sample ID".
"""

Profiler_Agent= ConversableAgent(
    name="Profiler_Agent",
    llm_config=llm_config,
    system_message= Profiler_Agent_Message,
    human_input_mode="NEVER",  # Never ask for human input
    context_variables=context_variables
)


Plan_Reviewer_Message="""
    You are the Plan Reviewer. Your job is to use the following tool:
    - Get_Dataset_Metadata: Retrieves the stored dataset metadata from the Profiler Agent.

    You must use this output to review the analysis plans created by the Planner Agent. 
    
    You must ensure that the plans are feasible given the dataset metadata.
    If there are any issues with the plans, you must instruct the Planner_Agent to 
    re-write the plan. 
    """
Plan_Reviewer_Agent= ConversableAgent(
    name="Plan_Reviewer_Agent",
    llm_config=llm_config,
    system_message=Plan_Reviewer_Message,
    human_input_mode="NEVER",
    context_variables=context_variables
)
    


# ========= Planner Group Nested Chat (Takes Place Inside Planning Manager Agent) =============
# The planning manager is the only Agent that the Data Supervisor sees. The groupchat needs its own
# chat manager as well, but it is all hosted by the Planning Manager Agent.
Plan_Manager_Message="""
You are the Plan Manager. You oversee the planning group. You receive instructions from
the Data Supervisor and you must report all results back to the Data Supervisor Agent.
"""

Plan_Manager= ConversableAgent(
    name="Plan_Manager",
    llm_config=llm_config,
    system_message=Plan_Manager_Message,
    human_input_mode="NEVER",
    context_variables=context_variables
)

register_function(
    Profile_Check,
    caller=Profiler_Agent,
    executor=Profiler_Agent,
    name="Profile_Check",
    description="Extract metadata from a specified CSV file and store for other agents."
)
register_function(
    Get_Dataset_Metadata,
    caller=Planner_Agent,
    executor=Planner_Agent,
    name="Get_Dataset_Metadata",
    description="Retrieve stored dataset metadata. Call with no arguments: {}. DO NOT pass data or context_variables."
)
Plan_Chat=GroupChat(
    agents=[Profiler_Agent, Planner_Agent, Plan_Reviewer_Agent],
)
Manager= GroupChatManager(
    groupchat=Plan_Chat,
    llm_config=llm_config,
    context_variables=context_variables)

planning_nested_chat=[{
    "recipient": Manager,  # Use the GroupChatManager, not the GroupChat object
    "message": "Profile the dataset and create a comprehensive analysis plan.",
    "max_turns": 5}]

Plan_Manager.register_nested_chats(
    chat_queue=planning_nested_chat,
    trigger=lambda sender: sender not in [Profiler_Agent, Planner_Agent, Plan_Reviewer_Agent],
)


#============ Start the Overall Conversations (Outer Group Chat With AutoPattern) ============================

initial_message="""
Analyse the data and provide a plan for analysis.
"""
pattern=AutoPattern(
    initial_agent=Plan_Manager,
    agents=[Plan_Manager], 
    group_manager_args={"llm_config": llm_config},
    context_variables=context_variables,
)
result, context, _ = initiate_group_chat(
    pattern=pattern,
    messages=initial_message,
    max_rounds=20,
)
