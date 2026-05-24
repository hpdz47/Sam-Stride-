# This function is for the Focus Area and Research Phase agents to replay so they can update context variables to enable
# correct handoffs and to update the shared memory.

#Imports:
from autogen.agentchat.group import ContextVariables, ReplyResult
from autogen import register_function
from typing import Any, Dict, List, Optional, Annotated, Tuple, Union
from autogen import Agent
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function
from Tools.RAG_Tools import RAG_Tool
import json


def Focus_Area_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    """Extract focus area statement from agent message."""
    
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
    
    # Try JSON, fallback to text
    try:
        data = json.loads(content)
        fa = data.get("fa", data.get("FA", data.get("focus_area", content)))
    except:
        fa = content
    
    sender.context_variables["Focus_Area_Statement"] = fa
    sender.context_variables["FA_Available"] = True
    # Reset Context Variables for RAG Agents
    sender.context_variables["FA_RAG_Available"] = False
    sender.context_variables["FA_RAG_Interpretation_Available"] = False
    
    return message

def FA_RAG_Questions_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    """Parse questions from RAG agent and query RAG system."""
    
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
    
    # Handle both dict (already parsed) and string (JSON)
    if isinstance(content, dict):
        data = content
    else:
        data = json.loads(content)
    
    # Extract questions (already validated by response_format)
    questions = data["questions"]
    
    # Query RAG system
    results = [RAG_Tool(q) for q in questions]
    
    # Format results
    qa_text = "\n".join([
        f"Query: {questions[i]} | Result: {results[i]}" 
        for i in range(len(questions))
    ])
    
    sender.context_variables["FA_RAG_QA"] = qa_text
    sender.context_variables["FA_RAG_Available"] = True
    
    return message
    

def FA_RAG_Interpret_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    """Extract RAG interpretation from Focus Area phase."""
    
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
        # Fallback to raw content
        interp = content
    
    sender.context_variables["FA_RAG_Interpretation"] = interp
    sender.context_variables["FA_RAG_Interpretation_Available"] = True
    # Reset Context Variables for Focus Area Agent
    sender.context_variables["FA_Available"]=False
    # Focus Area Agent resets RAG QA Context variables.
    
    return message