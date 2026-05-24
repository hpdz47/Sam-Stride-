# ---------------------Imports -----------------------------
from autogen.agentchat.group import AgentTarget, RevertToUserTarget, OnCondition, StringLLMCondition
from autogen.agentchat.group import OnContextCondition, ExpressionContextCondition, ContextExpression
from autogen.agentchat.group.guardrails import Guardrail, RegexGuardrail, GuardrailResult
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat.group import ContextVariables, ReplyResult
from autogen.agentchat.group import StringContextCondition
from autogen.agentchat import initiate_group_chat
from autogen import GroupChat, GroupChatManager
from autogen.agentchat.group.targets.transition_target import StayTarget, TerminateTarget
from typing import Any, Dict, List, Optional, Annotated
from autogen import UpdateSystemMessage
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
import numpy as np
import pandas as pd
import json
import csv
import os

from vLLM_Configuration import VLLM_Config
from vLLM_Manager import LLM_Manager
from dotenv import load_dotenv
from Singularity_Command_Line_Executor import SingularityCommandLineCodeExecutor
from pathlib import Path

# Import the RAG_Mode function from RAG.py
from RAG import RAG_Mode

load_dotenv()

if __name__ == "__main__":
    # Start the vLLM server
    LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
    
    # Create LLM config
    llm_config = VLLM_Config(
        api_type="openai",
        cache_seed=None,
        temperature=0.3,
        enable_thinking=False,
        LLM_Type="Reasoning"
    ).build_config()
    
    # Create context variables
    context_variables = ContextVariables({
        "RAG_Answer": "",
        "RAG_Query": "",
    })
    
    # Create the RAG agent
    RAG_Agent = ConversableAgent(
        name="RAG_Agent",
        llm_config=llm_config,
        system_message="""
        You are a RAG (Retrieval Augmented Generation) agent. Your job is to answer questions 
        about documents by using the RAG_Mode tool.
        
        When a user asks a question about documents, you MUST use the RAG_Mode function 
        to query the document database and retrieve relevant information.
        
        After receiving the RAG response, provide a clear and helpful answer to the user.
        """,
        human_input_mode="NEVER",
    )
    
    # Register the RAG_Mode function with the agent
    register_function(
        RAG_Mode,
        caller=RAG_Agent,
        executor=RAG_Agent,
        name="RAG_Mode",
        description="Query the RAG system to retrieve information from ingested documents. Use this when you need to answer questions about documents."
    )
    
    # Create pattern for conversation
    pattern = DefaultPattern(
        initial_agent=RAG_Agent,
        agents=[RAG_Agent],
        context_variables=context_variables,
    )
    
    # Run test conversation
    test_query = "Ingest the information in the website with url https://en.wikipedia.org/wiki/High-performance_liquid_chromatography  and give me a brief summary?"
    
    result, final_context, last_agent = initiate_group_chat(
        pattern=pattern,
        messages=test_query,
        max_rounds=5,
    )
    
    # Print results
    print("\n" + "="*50)
    print("RAG Test Results:")
    print("="*50)
    print(f"RAG Answer: {final_context.get('RAG_Answer', 'No answer')}")
    print(f"RAG Query: {final_context.get('RAG_Query', 'No query')}")
    print("\nFull conversation:")
    for msg in result.chat_history:
        print(f"\n[{msg.get('name', 'Unknown')}]: {msg.get('content', '')[:200]}...")
