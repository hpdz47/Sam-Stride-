# These functions are for the agents in the Planning Phase to reply so that they can update context variables to enable
# correct handoffs and to update the shared memory.

#Imports:
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function
from autogen.agentchat.group import ContextVariables, ReplyResult
from autogen import register_function
from typing import Any, Dict, List, Optional, Annotated, Tuple, Union
from autogen import Agent
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function
from concurrent.futures import ThreadPoolExecutor
import json
import csv
from Utils.Pydantic_Schema import PlanResponse
from Chatrooms.Review_Panel import Variables_Chatroom, Focus_Area_Assessment, RAG_System, Output_Instruction, Compile_Review, Context_Review

def Planner_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    """Parse plan from Planner agent and trigger reviews."""
    
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
    
    # Parse JSON and validate with Pydantic
    plan = json.loads(content)
    
    # Store plan using CURRENT count as index (before incrementing)
    sender.context_variables["Plan"] = plan
    sender.context_variables["Plan_Updated"] = True
    
    # Commit to git
    Manager = sender.context_variables["Plan_Manager"]
    Manager.write_plan(plan)
    Manager.commit(approach="Plan Updated", outcome=None, error_message=None)
    
    # Trigger review panel
    from Chatrooms.Review_Panel import Variables_Chatroom, Focus_Area_Assessment, Output_Instruction, Context_Review, Compile_Review
    
    Var = Variables_Chatroom(context_variables=sender.context_variables, Max_Rounds=10)
    Var.run_Conversation()
    
    FA = Focus_Area_Assessment(context_variables=sender.context_variables, Max_Rounds=10)
    FA.run_Conversation()
    
    # RAG = RAG_System(context_variables=sender.context_variables, Max_Rounds=10)
    # RAG.run_Conversation()
    
    OP = Output_Instruction(context_variables=sender.context_variables, Max_Rounds=15)
    OP.run_Conversation()
    
    Ctx = Context_Review(context_variables=sender.context_variables, Max_Rounds=10)
    Ctx.run_Conversation()
    
    CR = Compile_Review(context_variables=sender.context_variables, Max_Rounds=10)
    CR.run_Conversation()
    
    # Increment counter AFTER reviews complete
    sender.context_variables["Plan_Review_Count"] += 1
    
    return message

def Plan_Hook(
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
    
    a = content
     # Try parsing JSON
    try:
        a = json.loads(a)
    except Exception:
        # Not valid JSON → ignore (likely control message)
        return message
 
    # Push New Plan to the Plan Git Repo:
    Manager = sender.context_variables["Plan_Manager"]
    Manager.jog_memory()
    Manager.write_plan(a)
    Manager.commit(approach="Plan Updated", outcome=None, error_message=None)

    # Retrieve Diff:
    Diff = Manager.retrieve_diff()
    sender.context_variables["Plan_Diffs"] = Diff
    
    sender.context_variables["Plan"] = a
    sender.context_variables["Plan_Fixed"] = True
    
    # Reset context variables to ensure handoffs are triggered correctly.
    sender.context_variables["Plan_Diff_Reviewed"] = False # Reset the flags to ensure handoffs are triggered correctly.
    return message

def Plan_Hook_No_Memory(
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
    
    a = content
     # Try parsing JSON
    try:
        a = json.loads(a)
    except Exception:
        # Not valid JSON → ignore (likely control message)
        return message

    # Push New Plan to the Plan Git Repo:
    Manager = sender.context_variables["Plan_Manager"]
    Manager.write_plan(a)
    Manager.commit(approach="Plan Updated", outcome=None, error_message=None)

    sender.context_variables["Plan"] = a
    sender.context_variables["Plan_Fixed"] = True


    # No handoff conditions to reset when memory is turned off.

    #------ Trigger the Review Panel ------------------
    # Note: May be able to run all but the end compiler in parallel to save compute time.
    Var=Variables_Chatroom(context_variables=sender.context_variables, Max_Rounds=10)
    Var.run_Conversation()
    FA=Focus_Area_Assessment(context_variables=sender.context_variables, Max_Rounds=10)
    FA.run_Conversation()
    # RAG=RAG_System(context_variables=sender.context_variables, Max_Rounds=10)
    # RAG.run_Conversation()
    OP=Output_Instruction(context_variables=sender.context_variables, Max_Rounds=15)
    OP.run_Conversation()
    Ctx=Context_Review(context_variables=sender.context_variables, Max_Rounds=10)
    Ctx.run_Conversation()
    CR=Compile_Review(context_variables=sender.context_variables, Max_Rounds=10)
    CR.run_Conversation()
    #----------------------------------------------------
    sender.context_variables["Plan_Review_Count"] += 1 
    return message


def Diff_Hook(
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
    
    a = content
     # Try parsing JSON
    try:
        a = json.loads(a)
    except Exception:
        # Not valid JSON → ignore (likely control message)
        return message
    
    sender.context_variables["Plan_Approach"] = a
    sender.context_variables["Plan_Diff_Reviewed"] = True
    print(f"\n \n Diff Received: {a} \n \n")

    # Reset context variables to ensure handoffs are triggered correctly.
    sender.context_variables["Plan_Fixed"] = False # Reset the Plan Fixed flag to ensure that the handoff to the Plan Fixer is triggered when the Diff Agent updates the plan based on the diff.

    #------ Trigger the Review Panel ------------------
    # Note: May be able to run all but the end compiler in parallel to save compute time.
    Var=Variables_Chatroom(context_variables=sender.context_variables, Max_Rounds=10)
    Var.run_Conversation()
    FA=Focus_Area_Assessment(context_variables=sender.context_variables, Max_Rounds=10)
    FA.run_Conversation()
    # RAG=RAG_System(context_variables=sender.context_variables, Max_Rounds=10)
    # RAG.run_Conversation()
    OP=Output_Instruction(context_variables=sender.context_variables, Max_Rounds=15)
    OP.run_Conversation()
    Ctx=Context_Review(context_variables=sender.context_variables, Max_Rounds=10)
    Ctx.run_Conversation()
    CR=Compile_Review(context_variables=sender.context_variables, Max_Rounds=10)
    CR.run_Conversation()
    #----------------------------------------------------
    sender.context_variables["Plan_Review_Count"] += 1

    return message


    