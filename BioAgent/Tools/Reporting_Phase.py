# These functions are for the agents in the Implementation Phase to reply so that they can update context variables to enable
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
from pydantic import ValidationError
from Utils.Pydantic_Schema import ReportResponse, VLResponse

def VL_Hook(
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
    
    if not content or not content.strip().startswith("{"):
        return message
    
    try:
        parsed = VLResponse.model_validate_json(content)
    except ValidationError:
        return message
    except Exception:
        return message

    a = parsed.model_dump()
    
    a["Image_File"] = sender.context_variables["Current_Image_Name"]

    sender.context_variables["Image_Analysis"].append(a)

    return message

def Markdown_Hook(
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
    
    a = content
     # Try parsing JSON
    try:
        a = json.loads(a)
    except Exception:
        # Not valid JSON → ignore (likely control message)
        return message

    sender.context_variables["Markdown_Analysis"].append(a)
    
    return message

def Report_Hook(
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
    
    a = content
     # Try parsing JSON
    try:
        a = json.loads(a)
    except Exception:
        # Not valid JSON → ignore (likely control message)
        return message

    sender.context_variables["Final_Report"] = ReportResponse(**a)
    
    return message