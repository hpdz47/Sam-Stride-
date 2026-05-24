# These functions are for the agents in the Code Review Panel in the Implementation Phase to reply so that they can update context variables to enable
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
from Tools.RAG_Tools import RAG_Tool

def Code_Adapter_Hook(
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
    
    allowed_specialisms = ["ECL", "PEL", "OL"]
    if sender.context_variables["Specialism"] not in allowed_specialisms:
        raise ValueError(f"Invalid Specialism: {sender.context_variables['Specialism']}. Must be one of {allowed_specialisms}.")
    
    # Only valid JSON reaches here

    if sender.context_variables["Specialism"] == "ECL":
        sender.context_variables["ECL_Learnings"] = a
    elif sender.context_variables["Specialism"] == "PEL":
        sender.context_variables["PEL_Learnings"] = a
    elif sender.context_variables["Specialism"] == "OL":
        sender.context_variables["OL_Learnings"] = a
        
    sender.context_variables["Code_Reviews_Adapted"] = True

    return message

def Code_Score(Score: Annotated[float,"You must use the specialist feedback and the scoring criteria to provide a score out of 10."],
Justification: Annotated[str,"You must provide a concise justification for the score you have given based on the specialist feedback and the scoring criteria."],
context_variables: ContextVariables) -> ReplyResult:

    # NOTE: Justification is purely for debugging and is never passed to another agent or stored in context variables.

    Index = context_variables["Idx"]
    Code_Idx = context_variables["Code_Review_Count"]
    context_variables[f"Code_Score_{Code_Idx}"][Index] = Score

    context_variables["Scoring_Complete"]= True

    # Context Variables will be reset at the start of the next chatroom run so there is no need to reset them here.
    return ReplyResult(
        message=f"Judgement Provided",
        context_variables=context_variables,
    )

# =========== Hooks =============
def Error_Check_Hook(
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

    # Only valid JSON reaches here
    PN = {k: v for k, v in a.items() if k != "Improvements"}
    I = a.get("Improvements")
    #print(f"\n\n PN:Test \n{PN} \n\n") # For Debugging.
    sender.context_variables["Code_Errors"] = PN
    #print(f"\n\n I:Test \n{I} \n\n") # For Debugging.
    sender.context_variables["ECL_Improvements"] = I
    sender.context_variables["Errors_Checked"]=True

    return message

def Enforcement_Hook(
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

    # Only valid JSON reaches here
    PN = {k: v for k, v in a.items() if k != "Improvements"}
    I = a.get("Improvements")
    #print(f"\n\n PN:Test \n{PN} \n\n") # For Debugging.
    sender.context_variables["Plan_Enforcement"] = PN
    #print(f"\n\n I:Test \n{I} \n\n") # For Debugging.
    sender.context_variables["PEL_Improvements"] = I
    sender.context_variables["Enforcement_Complete"]=True

    return message

def Optimisation_Hook(
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

    # Only valid JSON reaches here
    PN = {k: v for k, v in a.items() if k != "Improvements"}
    I = a.get("Improvements")
    #print(f"\n\n PN:Test \n{PN} \n\n") # For Debugging.
    sender.context_variables["Optimisation_Goals"] = PN
    #print(f"\n\n I:Test \n{I} \n\n") # For Debugging.
    sender.context_variables["OL_Improvements"] = I
    sender.context_variables["Optimisation_Assessed"]=True

    return message
