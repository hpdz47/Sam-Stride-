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


# ------------------------ Guardrail Agent Structured Output Models ----------------------------------------

class PolicyViolation(BaseModel):
    """Details of a specific policy violation detected."""
    violation_type: str = Field(description="Type of violation (e.g., 'hallucination', 'unauthorized_action', 'incorrect_format', 'out_of_scope')")
    severity: str = Field(description="Severity level: 'low', 'medium', 'high', or 'critical'")
    reason: str = Field(description="Detailed explanation of why this is a policy violation")
    violated_rule: str = Field(description="The specific rule or policy that was violated")

class GuardrailResponse(BaseModel):
    """Structured response from guardrail agent when evaluating agent behavior."""
    compliant: bool = Field(description="True if the response passes all guardrails, False if there are violations")
    violations: List[PolicyViolation] = Field(
        default=[],
        description="List of policy violations detected (empty if compliant=True)"
    )
    action: str = Field(description="Required action: 'approve' (if compliant), 'regenerate' (minor violations), 'terminate' (critical violations), or 'redirect' (redirect to correct agent)")
    feedback_to_agent: str = Field(description="Specific feedback to help the agent correct its behavior if action is 'regenerate' or 'redirect'")
    corrective_prompt: Optional[str] = Field(
        default=None,
        description="A corrective prompt to append to the agent's next attempt if action='regenerate'"
    )
