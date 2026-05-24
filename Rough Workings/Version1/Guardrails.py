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


# Adapter guardrail to accept dict replies and extract content safely
class SafeLLMGuardrail(Guardrail):
    """Wrapper around Guardrail that tolerates dict replies from tools/agents.

    This guardrail handles tool calls and function responses gracefully by:
    1. Skipping validation for tool calls (they're system-level, not agent output)
    2. Extracting text content from various message formats
    3. Converting complex objects to strings for validation
    
    Only validates actual agent text responses, not tool invocations.
    """

    def __init__(self, name: str, condition: str, target: AgentTarget, activation_message: str, llm_config: LLMConfig):
        """Initialize the guardrail with LLM-based validation.
        
        Args:
            name: Name of the guardrail
            condition: The condition to check (as a natural language question)
            target: The target agent to notify if guardrail is triggered
            activation_message: Message to display when guardrail activates
            llm_config: LLM configuration for validation
        """
        super().__init__(name=name, condition=condition, target=target, activation_message=activation_message)
        self.llm_config = llm_config
        self._client = None

    def _get_llm_client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            from openai import OpenAI
            config = self.llm_config.config_list[0]
            # Convert base_url to string if it's a Pydantic HttpUrl object
            base_url = str(config["base_url"]) if config["base_url"] else "http://127.0.0.1:8000/v1"
            self._client = OpenAI(
                api_key=config["api_key"],
                base_url=base_url
            )
        return self._client

    def _to_text(self, val) -> str:
        """Convert any value to a string representation."""
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        # Pydantic models / objects
        try:
            if hasattr(val, "model_dump_json"):
                return val.model_dump_json()
            if hasattr(val, "json"):
                return val.json()
        except Exception:
            pass
        # Fallback to JSON if possible, else str
        try:
            return json.dumps(val, ensure_ascii=False, default=str)
        except Exception:
            return str(val)

    def _is_tool_call_message(self, context) -> bool:
        """Check if this message is a tool call or tool response."""
        if not isinstance(context, dict):
            return False
        
        # Check for tool call indicators
        if "tool_calls" in context and context["tool_calls"]:
            return True
        if "function_call" in context and context["function_call"]:
            return True
        if context.get("role") == "tool":
            return True
        if context.get("name") and not context.get("content"):
            # Function call without content
            return True
        
        # Check if content is None or empty (often indicates tool call)
        content = context.get("content")
        if content is None or (isinstance(content, str) and not content.strip()):
            # But only if there are tool_calls present
            if "tool_calls" in context or "function_call" in context:
                return True
        
        return False

    def check(self, context):  # type: ignore[override]
        """
        Check the context against guardrail conditions.
        Skips validation for tool calls and system messages.
        """
        # Handle dict messages (most common case)
        if isinstance(context, dict):
            # Skip tool calls entirely - they're not agent output to validate
            if self._is_tool_call_message(context):
                # Return a non-activated result (no violation)
                return GuardrailResult(activated=False, guardrail=self)
            
            # Extract content from message dict
            content = context.get("content", None)
            if content is None or (isinstance(content, str) and not content.strip()):
                # Empty content, nothing to validate
                return GuardrailResult(activated=False, guardrail=self)
            
            context_norm = self._to_text(content)
        
        # Handle list of messages
        elif isinstance(context, list):
            # Convert a list of messages into a readable transcript string
            # But skip tool calls
            parts: List[str] = []
            for item in context:
                if isinstance(item, dict):
                    if self._is_tool_call_message(item):
                        continue  # Skip tool calls
                    
                    role = item.get("role", "assistant")
                    content = item.get("content", item)
                    if content and (not isinstance(content, str) or content.strip()):
                        c = self._to_text(content)
                        parts.append(f"{role}: {c}")
                else:
                    parts.append(self._to_text(item))
            
            if not parts:
                # No content to validate
                return GuardrailResult(activated=False, guardrail=self)
            
            context_norm = "\n".join(parts)
        
        # Handle plain strings
        elif isinstance(context, str):
            if not context.strip():
                return GuardrailResult(activated=False, guardrail=self)  # Empty string, nothing to validate
            context_norm = context
        
        # Handle other types
        else:
            context_norm = self._to_text(context)
            if not context_norm.strip():
                return GuardrailResult(activated=False, guardrail=self)

        # Now validate using LLM
        return self._llm_check(context_norm)

    def _llm_check(self, content: str) -> GuardrailResult:
        """Use LLM to evaluate if content violates the guardrail condition."""
        try:
            client = self._get_llm_client()
            config = self.llm_config.config_list[0]
            
            # Create a prompt that asks the LLM to evaluate the condition
            prompt = (
                "You are evaluating agent output for policy compliance.\n\n"
                f"Condition to check: {self.condition}\n\n"
                "Agent output to evaluate:\n"
                f"{content}\n\n"
                "Answer with ONLY 'yes' if the condition is true (violation detected), "
                "or 'no' if the condition is false (no violation).\n"
                "Your answer:"
            )

            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get("temperature", 0.0),
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().lower()
            
            # If LLM says "yes", the condition is met (violation detected)
            activated = "yes" in answer
            
            # Debug: Print guardrail check results and save to file
            if activated:
                print(f"⚠️ GUARDRAIL VIOLATION DETECTED by {self.name}")
                print(f"   Condition: {self.condition}")
                print(f"   LLM Response: {answer}")
                
                # Save violation details to file for debugging
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"guardrail_violation_{timestamp}.txt"
                try:
                    with open(log_file, "w") as f:
                        f.write(f"=== GUARDRAIL VIOLATION LOG ===\n")
                        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                        f.write(f"Guardrail: {self.name}\n")
                        f.write(f"Condition: {self.condition}\n")
                        f.write(f"LLM Response: {answer}\n")
                        f.write(f"\n=== AGENT OUTPUT THAT TRIGGERED VIOLATION ===\n")
                        f.write(content)
                        f.write(f"\n\n=== END OF LOG ===\n")
                    print(f"   Violation details saved to: {log_file}")
                except Exception as e:
                    print(f"   Failed to save violation log: {e}")
            else:
                print(f"✓ Guardrail passed: {self.name}")
            
            return GuardrailResult(activated=activated, guardrail=self)
            
        except Exception as e:
            # If LLM call fails, don't block the agent - return non-activated
            print(f"⚠️ Warning: Guardrail '{self.name}' LLM check failed: {e}")
            import traceback
            traceback.print_exc()
            return GuardrailResult(activated=False, guardrail=self)

