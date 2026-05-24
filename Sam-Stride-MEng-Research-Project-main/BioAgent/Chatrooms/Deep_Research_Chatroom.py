from Tools.Deep_Research import run_deep_research
# ---------------------Imports ------------------------------------------------------------------------
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
from typing import Any, Dict, List, Optional, Annotated, Tuple, Union
from autogen import UpdateSystemMessage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import csv
import os
from Config.vLLM_Configuration import VLLM_Config
from Config.vLLM_Manager import LLM_Manager
from dotenv import load_dotenv
from pathlib import Path
from autogen import Agent
import copy
import pprint
import re
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from Agents.AgentFactory import Agent_Factory
load_dotenv()
#----------------------------------------------------------------------------------

def Deep_Research_Function(task: str, context_variables: ContextVariables) -> ReplyResult:
    run_deep_research(task)
    context_variables["Deep_Research_Complete"] = True
    return ReplyResult(
        message="Deep research completed successfully.",
        context_variables=context_variables,
    )

class DR_Chatroom:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int = 8):
        print("Deep Research Phase in Progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()

        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        self.Chat_Config = VLLM_Config(
            api_type="openai",
            cache_seed=None,
            temperature=0.4,
            enable_thinking=False,
            LLM_Type="Reasoning",
        ).build_config()

        self.DR_Agent = Agent_Factory(
            agent_name="Deep_Research_Agent",
            context_variables=self.context_variables,
        ).BuildAgent()

        register_function(
            Deep_Research_Function,
            caller=self.DR_Agent,
            executor=self.DR_Agent,
            name="Deep_Research_Function",
            description="Run deep research on a detailed scientific research brief and save the report.",
        )

        self.DR_Agent.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Deep_Research_Complete} == True")
                ),
            )
        )
        self.DR_Agent.handoffs.set_after_work(AgentTarget(self.DR_Agent))

    def run_Conversation(self):
        pattern = DefaultPattern(
            initial_agent=self.DR_Agent,
            agents=[self.DR_Agent],
            group_manager_args={"llm_config": self.Chat_Config},
            context_variables=self.context_variables,
        )

        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages=(
                "Use the dataset metadata and user requirements to create a single detailed deep research brief to form the web search"
            ),
            max_rounds=self.Max_Rounds,
        )

        return result, ctx

