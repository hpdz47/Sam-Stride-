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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------ Context Variables and Functions ---------------------
context = ContextVariables(data={
    "Agent_Conversation_Path": [],
    "dataset_metadata": None,  # Store profiler results here
})

def History_Context():
    """
    Returns a list of agents that have participated in the conversation in the 
    order they were accessed.
    Example:
    [Data_Supervisor, Profiler_Agent, Data Supervisor, Planner_Agent]
    The data supervisor obtained the information from the profiler agent, and
    then instructed the planner agent to come up with a plan based on that information.
    """
    Previous_Agents=context.get("Agent_Conversation_Path")
    return Previous_Agents

def Add_to_History(agent_name:str):
    """
    Adds the agent name to the conversation history.
    """
    Previous_Agents=context.get("Agent_Conversation_Path")
    Previous_Agents.append(agent_name)
    context.set("Agent_Conversation_Path",Previous_Agents)

def Store_Dataset_Metadata(metadata: dict):
    """
    Stores the dataset metadata from the Profiler Agent for use by other agents.
    
    Args:
        metadata (dict): The metadata dictionary returned by the Processing function
    """
    context.set("dataset_metadata", metadata)
    return "Dataset metadata stored successfully"

def Get_Dataset_Metadata():
    """
    Retrieves the stored dataset metadata.
    
    Returns:
        dict: The dataset metadata, or None if not yet stored
    """
    metadata = context.get("dataset_metadata")
    if metadata is None:
        return "No dataset metadata available yet. Please run the Planner_Manager Agent first."
    return metadata


# ------------------- Planning Phase Functions ------------------------
# Initial Data Processing Function ==> Extract Metadata
def Processing(csv_file_path: str, sample_rows: int = 5) -> Dict[str, Any]:
    """
    Extract metadata and structure information from a CSV file for LLM agent processing.
    
    This function reads a CSV file and extracts key metadata including column names,
    data types, sample data, and basic statistics.
    
    Args:
        csv_file_path (str): Path to the CSV file to process. Can be:
            - Just a filename (e.g., "Test_Data.csv") - will look in script directory
            - Relative path from script directory
            - Absolute path
        sample_rows (int): Number of sample rows to include in metadata (default: 5)
    
    Returns:
        Dict[str, Any]: A structured dictionary containing:
            - file_info: Basic file information (name, size, path)
            - columns: List of column names
            - column_count: Number of columns
            - row_count: Total number of rows (excluding header)
            - data_types: Inferred data types for each column
            - numeric_columns: List of numeric column names
            - categorical_columns: List of non-numeric column names
            - sample_data: First N rows as list of dictionaries
            - missing_data: Count of missing values per column
            - column_stats: Basic statistics for each column
            - summary: Human-readable summary string for LLM context
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file is empty or malformed
    """
    import os
    
    # Handle relative paths - if not absolute, check in script directory
    if not os.path.isabs(csv_file_path):
        # Try in script directory first
        script_dir_path = os.path.join(SCRIPT_DIR, csv_file_path)
        if os.path.exists(script_dir_path):
            csv_file_path = script_dir_path
        elif not os.path.exists(csv_file_path):
            # If not found, provide helpful error with available files
            available_files = list_csv_files()
            error_msg = f"CSV file not found: {csv_file_path}\n"
            if available_files:
                error_msg += f"Available CSV files in {SCRIPT_DIR}: {', '.join(available_files)}"
            else:
                error_msg += f"No CSV files found in {SCRIPT_DIR}"
            raise FileNotFoundError(error_msg)
    
    # Validate file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    # Initialize metadata structure
    metadata = {
        "file_info": {},
        "columns": [],
        "column_count": 0,
        "row_count": 0,
        "data_types": {},
        "numeric_columns": [],
        "categorical_columns": [],
        "sample_data": [],
        "missing_data": {},
        "column_stats": {},
        "summary": ""
    }
    
    try:
        # Read CSV file with pandas for robust parsing
        df = pd.read_csv(csv_file_path)
        
        # Check if file is empty
        if df.empty:
            raise ValueError(f"CSV file is empty: {csv_file_path}")
        
        # Extract file information
        file_size = os.path.getsize(csv_file_path)
        metadata["file_info"] = {
            "filename": os.path.basename(csv_file_path),
            "filepath": os.path.abspath(csv_file_path),
            "size_bytes": file_size,
            "size_readable": f"{file_size / 1024:.2f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
        }
        
        # Extract column information
        metadata["columns"] = df.columns.tolist()
        metadata["column_count"] = len(df.columns)
        metadata["row_count"] = len(df)
        
        # Infer data types
        for col in df.columns:
            dtype = str(df[col].dtype)
            metadata["data_types"][col] = dtype
            
            # Categorize columns
            if pd.api.types.is_numeric_dtype(df[col]):
                metadata["numeric_columns"].append(col)
            else:
                metadata["categorical_columns"].append(col)
        
        # Extract sample data (first N rows)
        sample_size = min(sample_rows, len(df))
        metadata["sample_data"] = df.head(sample_size).to_dict('records')
        
        # Count missing values
        missing_counts = df.isnull().sum()
        metadata["missing_data"] = {col: int(count) for col, count in missing_counts.items()}
        
        # Generate column statistics
        for col in df.columns:
            col_stats = {}
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric column statistics
                col_stats["type"] = "numeric"
                col_stats["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                col_stats["max"] = float(df[col].max()) if not df[col].isnull().all() else None
                col_stats["mean"] = float(df[col].mean()) if not df[col].isnull().all() else None
                col_stats["median"] = float(df[col].median()) if not df[col].isnull().all() else None
                col_stats["std"] = float(df[col].std()) if not df[col].isnull().all() else None
            else:
                # Categorical column statistics
                col_stats["type"] = "categorical"
                col_stats["unique_count"] = int(df[col].nunique())
                col_stats["most_common"] = df[col].mode()[0] if not df[col].mode().empty else None
                
                # Include unique values if reasonable number
                if col_stats["unique_count"] <= 20:
                    col_stats["unique_values"] = df[col].dropna().unique().tolist()
            
            col_stats["missing_count"] = int(metadata["missing_data"][col])
            col_stats["missing_percentage"] = round((metadata["missing_data"][col] / len(df)) * 100, 2)
            
            metadata["column_stats"][col] = col_stats
        
        # Generate human-readable summary for LLM
        summary_parts = [
            f"CSV File: {metadata['file_info']['filename']}",
            f"Dimensions: {metadata['row_count']} rows × {metadata['column_count']} columns",
            f"\nColumns:",
        ]
        
        for col in metadata["columns"]:
            col_info = metadata["column_stats"][col]
            if col_info["type"] == "numeric":
                summary_parts.append(
                    f"  - {col} ({metadata['data_types'][col]}): "
                    f"Range [{col_info.get('min', 'N/A')} to {col_info.get('max', 'N/A')}], "
                    f"Mean: {col_info.get('mean', 'N/A')}"
                )
            else:
                summary_parts.append(
                    f"  - {col} ({metadata['data_types'][col]}): "
                    f"{col_info['unique_count']} unique values"
                )
            
            if col_info["missing_count"] > 0:
                summary_parts.append(f"    ⚠ {col_info['missing_count']} missing values ({col_info['missing_percentage']}%)")
        
        metadata["summary"] = "\n".join(summary_parts)
        
        return metadata
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty or malformed: {csv_file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error processing CSV file: {str(e)}")

def list_csv_files(directory: Optional[str] = None) -> str:
    """
    List all CSV files in the specified directory or script directory.
    
    Args:
        directory (str, optional): Directory path to search. If None or empty, uses script directory.
    
    Returns:
        str: Formatted string with list of CSV filenames found in the directory
    """
    if directory is None or directory == "":
        directory = SCRIPT_DIR
    
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        if files:
            return f"Found {len(files)} CSV file(s) in {directory}:\n" + "\n".join(f"  - {f}" for f in files)
        else:
            return f"No CSV files found in {directory}"
    except Exception as e:
        return f"Error listing files in {directory}: {str(e)}"

#------------------ Termination Condition ---------------------------------
def is_termination_msg(msg: dict[str, Any]) -> bool:
    content = msg.get("content", "")
    return (content is not None) and "==== Terminate ====" in content



# ======================== AGENTS =====================================

Data_Guardrail_Message="""
You are a Policy Enforcer. Your job is to monitor the outputs from other Agents that have
triggered the guardrails for policy violations. You must evaluate their outputs and 
explain:
- Why the output has violated a policy.
- What policy has been violated.
- Instruct the Agent to re-produce their output without violating the policy.
"""

Data_Guardrail_Agent= ConversableAgent(
    name="Data_Guardrail_Agent",
    llm_config=llm_config_guardrail_agent,  # Use structured config for detailed feedback
    system_message=Data_Guardrail_Message,
    human_input_mode="NEVER"
)




Data_Supervisor_Message="""
You are a Data Analysis Supervisor. Your job is to delegate tasks
to specialist Agents for analyzing High Performance Liquid Chromatography (HPLC)
and Mass Spectrometry (MS) experimental data. You can talk to the following Agents:

- Planning_Manager: Specialists in planning data analysis, with access to the dataset.


You must NEVER attempt to analyse the data yourself. Your only job is to delegate tasks.

You must keep track of which agents have been invloved in the conversation and provide relevant summaries. You
can use the following tools:

- History_Context: Returns a list of agents that have participated in the conversation in the
    order they were accessed.
- Add_to_History: Adds the agent name to the conversation history.
- Get_Dataset_Metadata: Retrieves the stored dataset metadata from the Planning Group.
"""

Data_Supervisor= ConversableAgent(
    name="Data_Supervisor",
    llm_config=llm_config,
    system_message=Data_Supervisor_Message,
    human_input_mode="NEVER",
    functions=[History_Context, Add_to_History,Get_Dataset_Metadata]
)



# ----------------Guardrails--------------------
Planning_Guardrail= SafeLLMGuardrail(
    name="Planning_Guardrail",
    llm_config=llm_config,
    condition= "Does the Agent try to produce its own plan or analysis instead of delegating to other Agents",
    target=AgentTarget(Data_Guardrail_Agent),
    activation_message="Planning Policy Violation Detected"
)
SH_GR_Condition="""
Does this output contain fabricated results or analysis. This agent can't analyse or know
anything about the data until it consults the Planning Manager.
"""
Supervisor_Hallucination_Guardrail= SafeLLMGuardrail(
    name="Supervisor_Hallucination_Guardrail",
    llm_config=llm_config,
    condition= SH_GR_Condition,
    target=AgentTarget(Data_Guardrail_Agent),
    activation_message="Data Analysis Hallucination Detected"
)

Data_Supervisor.register_output_guardrail(Planning_Guardrail)
Data_Supervisor.register_output_guardrail(Supervisor_Hallucination_Guardrail)
#----------------------------------------------------------------------------------------

Planner_Message="""
You are an expert in Chemical Processes and Proteomics. Your task
is to supervise specialists in analyzing experimental data from High
Performance Liquid Chromatography (HPLC) experiments and Mass Spectrometry
(MS) experiments, which are used to analyse the purity for biopharmaceuticals.

You have access to the following tools:
- Get_Dataset_Metadata: Retrieves the stored dataset metadata from the Profiler Agent.
You must use this as it provides context about what the data contains.

- Store_Dataset_Metadata: Store the extracted metadata.
You must use this tool to store a concise plan to document for other Agents.

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
    llm_config=llm_config,  # Start with standard config
    system_message=Planner_Message,
    human_input_mode="NEVER",  # Never ask for human input
)

# Register functions for Planner_Agent using register_function
register_function(
    Get_Dataset_Metadata,
    caller=Planner_Agent,
    executor=Planner_Agent,
    name="Get_Dataset_Metadata",
    description="Retrieve the stored dataset metadata from the Profiler Agent's analysis.",
)

register_function(
    Store_Dataset_Metadata,
    caller=Planner_Agent,
    executor=Planner_Agent,
    name="Store_Dataset_Metadata",
    description="Store the dataset metadata for use by other agents.",
)


# Profiler Agent (For Data Usability)

Profiler_Agent_Message= """
You are a Data Profiler. Your job is to extract the correct metadata from the provided data using
the tools you have access to. The tools you have access to are:

- list_csv_files: List all CSV files in the current directory.
- Processing: Extract metadata from a specified CSV file.
- Store_Dataset_Metadata: Store the extracted metadata.

Example Output:
- This data contains 20 rows and 100 columns.
- The columns are: "Time", "Intensity", "m/z", "Retention Time",
- There are 5 missing values in the entire dataset which is 5 percent of the data.
- The numerical metadata is: "Time", "Intensity", "m/z".
- The categorical metadata is: "Sample ID".
Then store the metadata.


You must NEVER attempt to analyse the data yourself. Your only job is to provide
informative metadata. When you have finished your response, you must end with:

== PROFILING COMPLETE ==
"""

Profiler_Agent= ConversableAgent(
    name="Profiler_Agent",
    llm_config=llm_config,
    system_message= Profiler_Agent_Message,
    human_input_mode="NEVER",  # Never ask for human input
)

# Register functions for Profiler_Agent using register_function
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
    description="Extract metadata and structure information from a CSV file.",
)

register_function(
    Store_Dataset_Metadata,
    caller=Profiler_Agent,
    executor=Profiler_Agent,
    name="Store_Dataset_Metadata",
    description="Store the dataset metadata for use by other agents.",
)

#--------------Guardrails-------------------------------
#Data_Hallucination_Guardrail= SafeLLMGuardrail(
   # name="Data_Hallucination_Guardrail",
   # llm_config=llm_config,
   # condition= "Does this output contain fabricated data analysis that could not have been performed by this agent",
    #target=AgentTarget(Data_Guardrail_Agent),
   # activation_message="Data Analysis Hallucination Detected"
#)
#Profiler_Agent.register_output_guardrail(Data_Hallucination_Guardrail)
#-------------------------------------------------------


# ========= Planner Group Nested Chat (Takes Place Inside Planning Manager Agent) =============
# The planning manager is the only Agent that the Data Supervisor sees. The groupchat needs its own
# chat manager as well, but it is all hosted by the Planning Manager Agent.
Planning_Manager_Message="""
You are the Planning Manager. You oversee the planning group. You receive instructions from
the Data Supervisor and you must report all results back to the Data Supervisor Agent.
"""

Planning_Manager= ConversableAgent(
    name="Planning_Manager",
    llm_config=llm_config,
    system_message=Planning_Manager_Message,
    human_input_mode="NEVER",
)

# Use GroupChat with "auto" speaker selection (LLM-based, like AutoPattern)
planning_groupchat = GroupChat(
    agents=[Profiler_Agent, Planner_Agent],
    max_round=20,
    speaker_selection_method="auto",  # LLM decides next speaker
    send_introductions=False
)

planning_manager = GroupChatManager(
    groupchat=planning_groupchat,
    llm_config=llm_config,
)

planning_nested_chat=[{
    "recipient": planning_manager,
    "message": lambda recipient, messages, sender, config: "Profile the dataset and create a comprehensive analysis plan.",
    "max_turns": 10  # Allow more turns for tool calls and back-and-forth
}]

Planning_Manager.register_nested_chats(
    chat_queue=planning_nested_chat,
    trigger=lambda sender: sender not in [Profiler_Agent, Planner_Agent]
)


#============ Start the Overall Conversations (Outer Group Chat With AutoPattern) ============================

initial_message="""
Analyse the data and provide a plan for analysis.
"""
pattern=AutoPattern(
    initial_agent=Data_Supervisor,
    agents=[Data_Supervisor, Planning_Manager, Data_Guardrail_Agent], 
    group_manager_args={"llm_config": llm_config}
)
result, context, _ = initiate_group_chat(
    pattern=pattern,
    messages=initial_message,
    max_rounds=40
)









