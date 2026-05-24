# These functions are for the agents in the Discovery Phase to reply so that they can update context variables to enable
# correct handoffs and to update the shared memory.

#Imports:
from autogen.agentchat.group import ContextVariables, ReplyResult
from autogen import register_function
from typing import Any, Dict, List, Optional, Annotated, Tuple, Union
from autogen import Agent
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function
from concurrent.futures import ThreadPoolExecutor
import json
import csv
import numpy as np
import pandas as pd
import os
from pathlib import Path

def EDA_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    """Parse EDA interpretation from agent message."""
    
    content = message.get("content", "") if isinstance(message, dict) else str(message)

    # ===== HANDOFF GUARD - Add this to ALL hooks =====
    if isinstance(content, str):
        content_stripped = content.strip()
        # Check for handoff patterns
        if (content_stripped.startswith("[Handing off to") or 
            content_stripped.startswith("Handing off to") or
            content_stripped.startswith("***** ") or  # AutoGen handoff markers
            "handoff" in content_stripped.lower()[:50]):  # Check first 50 chars
            return message  # Skip processing, return unchanged
    # ===== END HANDOFF GUARD =====
    
    # Try JSON first, fallback to plain text
    try:
        data = json.loads(content)
        interpretation = data.get("interpretation", data.get("Interpretation", content))
    except:
        interpretation = content  # Use entire message as interpretation
    
    sender.context_variables["EDA_Interpretation"] = interpretation
    sender.context_variables["Interpretation_Available"] = True
    
    return message

def Profile_Check(context_variables: ContextVariables, Input_Dir: str) -> ReplyResult:
    inputs_dir = Input_Dir

    def get_file_info(csv_file):
        file_size = csv_file.stat().st_size  # File size in bytes

        with csv_file.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)

            headers = next(reader, None)  # First row (columns)
            column_count = len(headers) if headers else 0

            row_count = 0
            for i in reader:
                row_count += 1

        return {
            "name": csv_file.name,
            "headers": headers,
            "columns": column_count,
            "rows": row_count,
            "file_size": file_size
        }

    csv_files = []
    for file in inputs_dir.iterdir():
        if file.suffix.lower() == ".csv":
            csv_files.append(file)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for info in executor.map(get_file_info, csv_files):

            context_variables["metadata"]= {
                "headers": info["headers"],
                "columns": info["columns"],
                "rows": info["rows"],
                "file_size_bytes": info["file_size"]
            }

    print(context_variables["metadata"])

    return ReplyResult(
        message="Profile_Check Successful",
        context_variables=context_variables
    )

def Deterministic_EDA(context_variables: ContextVariables, Input_Dir: str) -> ReplyResult:
    inputs_dir = Input_Dir
    file_results = {}
    results=[]
    for file in Input_Dir.iterdir():
        if file.suffix.lower() == ".csv":
            # 1 - Load the CSV headers in a pandas dataframe:
            headers = pd.read_csv(file, nrows=0).columns.tolist()
            for col in headers:
                # For each variable, load the entire column before doing necessary statistics:
                chunks = pd.read_csv(
                file,
                usecols=[col])
                var = chunks[col]

                if pd.api.types.is_numeric_dtype(var):
                    x = var.describe()

                    file_results = {
                        "File": file.name,
                        "Variable": col,
                        "type": "numeric",
                        "count": int(x["count"]),
                        "mean": float(x["mean"]),
                        "std": float(x["std"]),
                        "min": float(x["min"]),
                        "max": float(x["max"]),
                        "missing": int(var.isna().sum()),
                        "cardinality": int(var.nunique(dropna=True))
                    }

                else:

                    x = var.describe()

                    file_results = {
                        "File": file.name,
                        "Variable": col,
                        "type": "categorical",
                        "count": int(x["count"]),
                        "missing": int(var.isna().sum()),
                        "cardinality": int(x["unique"]),
                    }

                results.append(file_results) # Creates a list of dictionaries for a specific file.
            context_variables["EDA_Results"] = results
                
        else:
            continue # skip non-csv files (for now).
            
    return ReplyResult(message="Deterministic EDA completed and report stored.", context_variables=context_variables)
