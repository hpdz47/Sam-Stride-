# These functions are for the agents in the Review Panel in the Planning Phase to reply so that they can update context variables to enable
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
import copy
from Tools.RAG_Tools import RAG_Tool
from Utils.Pydantic_Schema import ReviewResponse, Compile_Response, P, N, I, J

def Review_RAG_Questions_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    """Parse questions from review panel agent and query RAG system."""
    
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
    data = json.loads(content)
    from Utils.Pydantic_Schema import RAGQuestions
    rag_qs = RAGQuestions(**data)
    questions = rag_qs.questions  # Already validated: 3-6 questions
    
    # Query RAG system
    from Tools.RAG_Tools import RAG_Tool
    results = [RAG_Tool(q) for q in questions]
    
    # Format results
    qa_text = "\n".join([
        f"Query: {questions[i]} | Result: {results[i]}" 
        for i in range(len(questions))
    ])
    
    sender.context_variables["Review_RAG_QA"] = qa_text
    sender.context_variables["Review_RAG_Available"] = True
    
    return message

def Review_RAG_Interpret_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    """Extract RAG interpretation from review panel."""
    
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
    
    # Parse JSON if structured output
    try:
        data = json.loads(content)
        interp = data.get("interpretation", data.get("Interpretation", content))
    except:
        interp = content
    
    sender.context_variables["Review_RAG_Interpret"] = interp
    sender.context_variables["Review_RAG_Interpret_Available"] = True
    
    return message


def Compiler_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    """Parse compiled review feedback."""
    
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
    data = json.loads(content)
    
    sender.context_variables["Plan_Feedback"] = data
    sender.context_variables["Review_Compilation_Complete"] = True
    
    return message

def Plan_Score(Score: Annotated[float,"You must use the specialist feedback and the scoring criteria to provide a score out of 10."],
Justification: Annotated[str,"You must provide a concise justification for the score you have given based on the specialist feedback and the scoring criteria."],
context_variables: ContextVariables) -> ReplyResult:

    # NOTE: Justification is purely for debugging and is never passed to another agent or stored in context variables.

    Index = context_variables["Idx"]
    Plan_Idx = context_variables["Plan_Review_Count"]
    context_variables[f"Plan_Score_{Plan_Idx}"][Index] = Score

    context_variables["Scoring_Complete"]= True

    # Context Variables will be reset at the start of the next chatroom run so there is no need to reset them here.
    return ReplyResult(
        message=f"Judgement Provided",
        context_variables=context_variables,
    )

# =========== Hooks =============
def Var_Feedback_Hook(
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
    sender.context_variables["Var_Review"] = PN
    #print(f"\n\n I:Test \n{I} \n\n") # For Debugging.
    sender.context_variables["Var_Improvements"] = I
    sender.context_variables["Var_Review_Available"]=True

    return message

def FA_Feedback_Hook(
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
    sender.context_variables["FA_Reviews"] = PN
    #print(f"\n\n I:Test \n{I} \n\n") # For Debugging.
    sender.context_variables["FA_Improvements"] = I
    sender.context_variables["FA_Reviews_Available"]=True

    return message

def OP_Feedback_Hook(
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
    sender.context_variables["Review_OP_Instruction"] = PN
    #print(f"\n\n I:Test \n{I} \n\n") # For Debugging.
    sender.context_variables["OP_Improvements"] = I
    sender.context_variables["OP_Instruction_Available"]=True

    return message

def CTX_Feedback_Hook(
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
    sender.context_variables["Context_Review"] = PN
    #print(f"\n\n I:Test \n{I} \n\n") # For Debugging.
    sender.context_variables["Ctx_Improvements"] = I
    sender.context_variables["Context_Review_Available"]=True

    return message

def Adapter_Hook(
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
    
    allowed_specialisms = ["Var", "FA", "OP", "Ctx"]
    if sender.context_variables["Specialism"] not in allowed_specialisms:
        raise ValueError(f"Invalid Specialism: {sender.context_variables['Specialism']}. Must be one of {allowed_specialisms}.")
    
    # Only valid JSON reaches here

    if sender.context_variables["Specialism"] == "Var":
        sender.context_variables["Var_Learnings"] = a
    elif sender.context_variables["Specialism"] == "FA":
        sender.context_variables["FA_Learnings"] = a
    elif sender.context_variables["Specialism"] == "OP":
        sender.context_variables["OP_Learnings"] = a
    elif sender.context_variables["Specialism"] == "Ctx":
        sender.context_variables["Context_Learnings"] = a
    sender.context_variables["Reviews_Adapted"] = True

    return message

    