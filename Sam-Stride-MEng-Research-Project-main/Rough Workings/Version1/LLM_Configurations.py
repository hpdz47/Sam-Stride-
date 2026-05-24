# ---------------------Imports -----------------------------
from autogen import UserProxyAgent, ConversableAgent, LLMConfig
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

from Structured_Responses import GuardrailResponse, PolicyViolation

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