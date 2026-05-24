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
from Utils.Pydantic_Schema import PlanResponse

from Chatrooms.Code_Review_Panel import Error_Check_Loop, Plan_Enforcement_Loop, Optimisation_Loop

def Coder_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    
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

    # Code will always be a string:
    
    Code = content.replace("`","#")
    
    # Store Code and Update Context Variables for Handoff
    sender.context_variables[f"Code"] = Code
    sender.context_variables["Code_Updated"] = True
    
    # Commit to git
    Manager = sender.context_variables["Code_Manager"]
    Manager.jog_memory()
    Manager.write_plan(Code)
    Manager.commit(approach="Code Updated", outcome=None, error_message=None)
    
    # Trigger review panel
    ECL = Error_Check_Loop(context_variables = sender.context_variables, Max_Rounds = 5)
    ECL.run_Conversation()
    PEL = Plan_Enforcement_Loop(context_variables = sender.context_variables, Max_Rounds = 5)
    PEL.run_Conversation()
    OL = Optimisation_Loop(context_variables = sender.context_variables, Max_Rounds = 5)
    OL.run_Conversation()
    
    # Increment counter AFTER reviews complete
    sender.context_variables["Code_Review_Count"] +=1
    
    return message

def Code_Fixer_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    
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

    # Code will always be a string:
    
    Code = content.replace("`","#")
    
    # Store Code and Update Context Variables for Handoff
    sender.context_variables["Code"] = Code
    sender.context_variables["Code_Fixed"] = True
    
    # Commit to git
    Manager = sender.context_variables["Code_Manager"]
    Manager.jog_memory()
    Manager.write_plan(Code)
    Manager.commit(approach="Code Updated", outcome=None, error_message=None)

    # Retrieve Diff:
    Diff = Manager.retrieve_diff()
    sender.context_variables["Code_Diffs"] = Diff

    # Reset Code Diff Agent context variables:
    sender.context_variables["Code_Diffs_Analysed"] = False
    
    return message

def Code_Fixer_Hook_No_Memory(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    
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

    # Code will always be a string:
    
    Code = content.replace("`","#")
    
    # Store Code and Update Context Variables for Handoff
    sender.context_variables["Code"] = Code
    sender.context_variables["Code_Fixed"] = True
    
    # Commit to git
    Manager = sender.context_variables["Code_Manager"]
    Manager.jog_memory()
    Manager.write_plan(Code)
    Manager.commit(approach="Code Updated", outcome=None, error_message=None)

    # Trigger Review Panel ==================
    ECL = Error_Check_Loop(context_variables = sender.context_variables, Max_Rounds=5)
    ECL.run_Conversation()
    PEL = Plan_Enforcement_Loop(context_variables = sender.context_variables, Max_Rounds=5)
    PEL.run_Conversation()
    OL = Optimisation_Loop(context_variables = sender.context_variables, Max_Rounds=5)
    OL.run_Conversation()
    #=========================================
    # Increment counter AFTER reviews complete.
    sender.context_variables["Code_Review_Count"] +=1
    return message


def Code_Diff_Hook(
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
    
    sender.context_variables["Code_Approach"] = a
    sender.context_variables["Code_Diffs_Analysed"] = True
    #print(f"\n \n Diff Received: {a} \n \n")

    # Reset context variables to ensure handoffs are triggered correctly.
    sender.context_variables["Code_Fixed"] = False # Reset the Code Fixed flag to ensure that the handoff to the Code Fixer is triggered when the Diff Agent updates the code based on the diff.

    #-------- Coding Review Panel Trigger:
    ECL = Error_Check_Loop(context_variables = sender.context_variables, Max_Rounds=5)
    ECL.run_Conversation()
    PEL = Plan_Enforcement_Loop(context_variables = sender.context_variables, Max_Rounds=5)
    PEL.run_Conversation()
    OL = Optimisation_Loop(context_variables = sender.context_variables, Max_Rounds=5)
    OL.run_Conversation()
    #-------------------------------------
    # Increment counter AFTER reviews complete.
    sender.context_variables["Code_Review_Count"] +=1
    return message

def Summary_Hook(
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
    O = a.get("Outcome")
    S = a.get("Summary")
    P = a.get("Pip")

    sender.context_variables["Success"] = O
    if O:
        sender.context_variables["Outcome"]= "Execution Successful"
    else:
        sender.context_variables["Outcome"]= "Execution Failed"

    sender.context_variables["Summary"] = S
    sender.context_variables["Error_Summarised"] = True
    sender.context_variables["Pip"] = P

    

    return message

def Git_Control_Hook(
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
    R = a.get("Rollback")
    S = a.get("Suggestions")

    sender.context_variables["Suggestions"] = S
    sender.context_variables["Rollback"] = R
    sender.context_variables["Controller_Finished"] = True

    return message

def Debug_Hook(
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
    C = a.get("Code")
    C = C.replace("`","#")
    AP = a.get("Approach")

    sender.context_variables["Code"] = C
    sender.context_variables["Approach"] = AP
    sender.context_variables["Debug_Finished"] = True

    return message

def Installations_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    
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

    # Code will always be a string:
    
    Shell = content.replace("`","#")

    sender.context_variables["Pip_Code"] = Shell
    sender.context_variables["Packages_Managed"] = True
    
    return message

def Analysis_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    
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

    
    Instructions = content.replace("`","#")

    sender.context_variables["Instructions"] = Instructions
    sender.context_variables["Failure_Analysed"] = True

   
    
    return message


