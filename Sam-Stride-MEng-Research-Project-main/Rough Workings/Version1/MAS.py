from autogen import UserProxyAgent, ConversableAgent, LLMConfig
import os
from autogen.agentchat import initiate_group_chat
# from autogen.agentchat.group.patterns import AutoPatternm -- Testing Default Pattern  --
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

# --------------------- Context Variables ---------------------
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
        return "No dataset metadata available yet. Please run the Profiler Agent first."
    return metadata
#--------------------------------------------------------------

# Guardrail Agent Structured Output Models

class PolicyViolation(BaseModel):
    """Details of a specific policy violation detected."""
    violation_type: str = Field(description="Type of violation (e.g., 'hallucination', 'unauthorized_action', 'incorrect_format', 'out_of_scope')")
    severity: str = Field(description="Severity level: 'low', 'medium', 'high', or 'critical'")
    reason: str = Field(description="Detailed explanation of why this is a policy violation")
    violated_rule: str = Field(description="The specific rule or policy that was violated")

class GuardrailResponse(BaseModel):
    """Structured response from guardrail agent when evaluating agent behavior."""
    compliant: bool = Field(description="True if the response passes all guardrails, False if there are violations")
    violations: List[PolicyViolation] = Field(
        default=[],
        description="List of policy violations detected (empty if compliant=True)"
    )
    action: str = Field(description="Required action: 'approve' (if compliant), 'regenerate' (minor violations), 'terminate' (critical violations), or 'redirect' (redirect to correct agent)")
    feedback_to_agent: str = Field(description="Specific feedback to help the agent correct its behavior if action is 'regenerate' or 'redirect'")
    corrective_prompt: Optional[str] = Field(
        default=None,
        description="A corrective prompt to append to the agent's next attempt if action='regenerate'"
    )

# LLM Configuration
llm_config = LLMConfig(config_list=[{
    "api_type": "openai",
    "model": os.environ["LLM_Model"],
    "api_key": os.environ["API_KEY"],
    "base_url": "http://127.0.0.1:8000/v1",
    "cache_seed": None,
    "temperature": 0.2,
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": False}  # Disable Qwen3 thinking mode
    }
}])

# LLM Configuration with structured output for Planner Agent
llm_config_structured = LLMConfig(config_list=[{
    "api_type": "openai",
    "model": os.environ["LLM_Model"],
    "api_key": os.environ["API_KEY"],
    "base_url": "http://127.0.0.1:8000/v1",
    "cache_seed": None,
    "temperature": 0.2,
    "response_format": DataAnalysisPlan,
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": False}  # Disable Qwen3 thinking mode
    }
}])

# LLM Configuration for Data Guardrail Agent (if triggered)
llm_config_guardrail_agent = LLMConfig(config_list=[{
    "api_type": "openai",
    "model": os.environ["LLM_Model"],
    "api_key": os.environ["API_KEY"],
    "base_url": "http://127.0.0.1:8000/v1",
    "cache_seed": None,
    "temperature": 0.2,
    "response_format": GuardrailResponse,
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": False}  # Disable Qwen3 thinking mode
    }
}])


# Adapter guardrail to accept dict replies and extract content safely
class SafeLLMGuardrail(Guardrail):
    """Wrapper around Guardrail that tolerates dict replies from tools/agents.

    This guardrail handles tool calls and function responses gracefully by:
    1. Skipping validation for tool calls (they're system-level, not agent output)
    2. Extracting text content from various message formats
    3. Converting complex objects to strings for validation
    
    Only validates actual agent text responses, not tool invocations.
    """

    def __init__(self, name: str, condition: str, target: AgentTarget, activation_message: str, llm_config: LLMConfig):
        """Initialize the guardrail with LLM-based validation.
        
        Args:
            name: Name of the guardrail
            condition: The condition to check (as a natural language question)
            target: The target agent to notify if guardrail is triggered
            activation_message: Message to display when guardrail activates
            llm_config: LLM configuration for validation
        """
        super().__init__(name=name, condition=condition, target=target, activation_message=activation_message)
        self.llm_config = llm_config
        self._client = None

    def _get_llm_client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            from openai import OpenAI
            config = self.llm_config.config_list[0]
            # Convert base_url to string if it's a Pydantic HttpUrl object
            base_url = str(config["base_url"]) if config["base_url"] else "http://127.0.0.1:8000/v1"
            self._client = OpenAI(
                api_key=config["api_key"],
                base_url=base_url
            )
        return self._client

    def _to_text(self, val) -> str:
        """Convert any value to a string representation."""
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        # Pydantic models / objects
        try:
            if hasattr(val, "model_dump_json"):
                return val.model_dump_json()
            if hasattr(val, "json"):
                return val.json()
        except Exception:
            pass
        # Fallback to JSON if possible, else str
        try:
            return json.dumps(val, ensure_ascii=False, default=str)
        except Exception:
            return str(val)

    def _is_tool_call_message(self, context) -> bool:
        """Check if this message is a tool call or tool response."""
        if not isinstance(context, dict):
            return False
        
        # Check for tool call indicators
        if "tool_calls" in context and context["tool_calls"]:
            return True
        if "function_call" in context and context["function_call"]:
            return True
        if context.get("role") == "tool":
            return True
        if context.get("name") and not context.get("content"):
            # Function call without content
            return True
        
        # Check if content is None or empty (often indicates tool call)
        content = context.get("content")
        if content is None or (isinstance(content, str) and not content.strip()):
            # But only if there are tool_calls present
            if "tool_calls" in context or "function_call" in context:
                return True
        
        return False

    def check(self, context):  # type: ignore[override]
        """
        Check the context against guardrail conditions.
        Skips validation for tool calls and system messages.
        """
        # Handle dict messages (most common case)
        if isinstance(context, dict):
            # Skip tool calls entirely - they're not agent output to validate
            if self._is_tool_call_message(context):
                # Return a non-activated result (no violation)
                return GuardrailResult(activated=False, guardrail=self)
            
            # Extract content from message dict
            content = context.get("content", None)
            if content is None or (isinstance(content, str) and not content.strip()):
                # Empty content, nothing to validate
                return GuardrailResult(activated=False, guardrail=self)
            
            context_norm = self._to_text(content)
        
        # Handle list of messages
        elif isinstance(context, list):
            # Convert a list of messages into a readable transcript string
            # But skip tool calls
            parts: List[str] = []
            for item in context:
                if isinstance(item, dict):
                    if self._is_tool_call_message(item):
                        continue  # Skip tool calls
                    
                    role = item.get("role", "assistant")
                    content = item.get("content", item)
                    if content and (not isinstance(content, str) or content.strip()):
                        c = self._to_text(content)
                        parts.append(f"{role}: {c}")
                else:
                    parts.append(self._to_text(item))
            
            if not parts:
                # No content to validate
                return GuardrailResult(activated=False, guardrail=self)
            
            context_norm = "\n".join(parts)
        
        # Handle plain strings
        elif isinstance(context, str):
            if not context.strip():
                return GuardrailResult(activated=False, guardrail=self)  # Empty string, nothing to validate
            context_norm = context
        
        # Handle other types
        else:
            context_norm = self._to_text(context)
            if not context_norm.strip():
                return GuardrailResult(activated=False, guardrail=self)

        # Now validate using LLM
        return self._llm_check(context_norm)

    def _llm_check(self, content: str) -> GuardrailResult:
        """Use LLM to evaluate if content violates the guardrail condition."""
        try:
            client = self._get_llm_client()
            config = self.llm_config.config_list[0]
            
            # Create a prompt that asks the LLM to evaluate the condition
            prompt = (
                "You are evaluating agent output for policy compliance.\n\n"
                f"Condition to check: {self.condition}\n\n"
                "Agent output to evaluate:\n"
                f"{content}\n\n"
                "Answer with ONLY 'yes' if the condition is true (violation detected), "
                "or 'no' if the condition is false (no violation).\n"
                "Your answer:"
            )

            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get("temperature", 0.0),
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().lower()
            
            # If LLM says "yes", the condition is met (violation detected)
            activated = "yes" in answer
            
            # Debug: Print guardrail check results and save to file
            if activated:
                print(f"⚠️ GUARDRAIL VIOLATION DETECTED by {self.name}")
                print(f"   Condition: {self.condition}")
                print(f"   LLM Response: {answer}")
                
                # Save violation details to file for debugging
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"guardrail_violation_{timestamp}.txt"
                try:
                    with open(log_file, "w") as f:
                        f.write(f"=== GUARDRAIL VIOLATION LOG ===\n")
                        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                        f.write(f"Guardrail: {self.name}\n")
                        f.write(f"Condition: {self.condition}\n")
                        f.write(f"LLM Response: {answer}\n")
                        f.write(f"\n=== AGENT OUTPUT THAT TRIGGERED VIOLATION ===\n")
                        f.write(content)
                        f.write(f"\n\n=== END OF LOG ===\n")
                    print(f"   Violation details saved to: {log_file}")
                except Exception as e:
                    print(f"   Failed to save violation log: {e}")
            else:
                print(f"✓ Guardrail passed: {self.name}")
            
            return GuardrailResult(activated=activated, guardrail=self)
            
        except Exception as e:
            # If LLM call fails, don't block the agent - return non-activated
            print(f"⚠️ Warning: Guardrail '{self.name}' LLM check failed: {e}")
            import traceback
            traceback.print_exc()
            return GuardrailResult(activated=False, guardrail=self)


# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Helper function to list CSV files in the directory
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

# Termination Condition
def is_termination_msg(msg: dict[str, Any]) -> bool:
    content = msg.get("content", "")
    return (content is not None) and "==== Terminate ====" in content


# Data Analysis Branch ====================================================================================

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

- Profiler_Agent: Specialist in providing metadata about the data.
This agent will finish conversations with == PROFILING COMPLETE ==

- Planner_Agent: Specialist in planning data analysis. The planner Agent does not have access to the data
and must rely on your relaying the summary from the Profiler Agent.

When you plan to relay a message to an Agent, you must finish with Delegating to [Agent Name].


You must NEVER attempt to analyse the data yourself. Your only job is to delegate tasks.

You must keep track of which agents have been invloved in the conversation and provide relevant summaries. You
can use the following tools:
- History_Context: Returns a list of agents that have participated in the conversation in the
    order they were accessed.
- Add_to_History: Adds the agent name to the conversation history.
- Get_Dataset_Metadata: Retrieves the stored dataset metadata from the Profiler Agent.
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
anything about the data until it instructs the Profiler Agent.
"""
Supervisor_Hallucination_Guardrail= SafeLLMGuardrail(
    name="Supervisor_Hallucination_Guardrail",
    llm_config=llm_config,
    condition= SH_GR_Condition,
    target=AgentTarget(Data_Guardrail_Agent),
    activation_message="Data Analysis Hallucination Detected"
)
Context_Guardrail= SafeLLMGuardrail(
    name="Context_Guardrail",
    llm_config=llm_config,
    condition= "Does the Agent fail to summarise the profiler Agent's response before sending instruction to the Planner Agent?",
    target=AgentTarget(Data_Guardrail_Agent),
    activation_message="Context Retrieval Policy Violation Detected"
)
Data_Supervisor.register_output_guardrail(Planning_Guardrail)
Data_Supervisor.register_output_guardrail(Supervisor_Hallucination_Guardrail)
Data_Supervisor.register_output_guardrail(Context_Guardrail)
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
    functions=[Get_Dataset_Metadata, Store_Dataset_Metadata]
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
    functions=[Processing, list_csv_files, Store_Dataset_Metadata],
    human_input_mode="NEVER",  # Never ask for human input
)

#--------------Guardrails-------------------------------
Data_Hallucination_Guardrail= SafeLLMGuardrail(
    name="Data_Hallucination_Guardrail",
    llm_config=llm_config,
    condition= "Does this output contain fabricated data analysis that could not have been performed by this agent",
    target=AgentTarget(Data_Guardrail_Agent),
    activation_message="Data Analysis Hallucination Detected"
)
Profiler_Agent.register_output_guardrail(Data_Hallucination_Guardrail)
#-------------------------------------------------------


# -------------Handoffs---------------------------
Data_Guardrail_Agent.handoffs.add_llm_conditions([
        OnCondition(
            target=AgentTarget(Data_Supervisor),
            condition=StringLLMCondition(prompt="When Data Supervisor Agent triggers the guardrail"),
        ),
        OnCondition(
            target=AgentTarget(Profiler_Agent),
            condition=StringLLMCondition(prompt="When Profiler Agent triggers the guardrail"),
        )
])

#----------- Handoffs ---------------------------
Data_Supervisor.handoffs.add_llm_conditions([
        OnCondition(
            target=AgentTarget(Profiler_Agent),
            condition=StringLLMCondition(prompt="Before developing a plan, dataset metadata is needed"),
        ),
        OnCondition(
            target=AgentTarget(Planner_Agent),
            condition=StringLLMCondition(prompt="Only when dataset metadata has been provided by the Profiler Agent, a plan needs to be developed"),
        )
])
Planner_Agent.handoffs.add_llm_conditions([
        OnCondition(
            target=AgentTarget(Data_Supervisor),
            condition=StringLLMCondition(prompt="When the plan for data analysis has been completed"),
        )
])
Profiler_Agent.handoffs.add_llm_conditions([
        OnCondition(
            target=AgentTarget(Data_Supervisor),
            condition=StringLLMCondition(prompt="When the output contains '== PROFILING COMPLETE ==' or profiling is finished"),
        )
])

Profiler_Agent.handoffs.set_after_work(AgentTarget(Data_Supervisor))
Planner_Agent.handoffs.set_after_work(AgentTarget(Data_Supervisor))

#------------------------------------------------

pattern = DefaultPattern(
    initial_agent=Data_Supervisor,
    agents=[Data_Supervisor, Profiler_Agent, Planner_Agent, Data_Guardrail_Agent],
    group_manager_args={"llm_config": llm_config}
)

initial_message=("Look at the data and come up with a plan for analysis")

result, context, _ = initiate_group_chat(
    pattern=pattern,
    messages=initial_message,
    max_rounds=20
)