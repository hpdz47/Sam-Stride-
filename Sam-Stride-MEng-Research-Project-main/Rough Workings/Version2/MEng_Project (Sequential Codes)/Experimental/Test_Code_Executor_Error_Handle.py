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
from typing import Any, Dict, List, Optional, Annotated, Tuple, Union
from autogen import UpdateSystemMessage
from pydantic import BaseModel, Field, ValidationError

import json
import csv
import os

from vLLM_Configuration import VLLM_Config
from vLLM_Manager import LLM_Manager
from dotenv import load_dotenv
from Singularity_Command_Line_Executor import SingularityCommandLineCodeExecutor
from pathlib import Path
from autogen import Agent
load_dotenv()

# Setup LLM Manager for Coding server
#LLM_Manager(LLM_Type="Coding").Manage_VLLM()

# Setup executor
#inputs_dir = Path("./Inputs")
work_dir = Path("./Data_Results")
setup_dir = Path("./Singularity_Images")

executor = SingularityCommandLineCodeExecutor(
    image="continuumio/anaconda3",
    timeout=60,
    work_dir=str(work_dir),
    setup_dir=str(setup_dir),
)

# Setup agents
#LLM = VLLM_Config(api_type="openai", cache_seed=None, temperature=0.2, enable_thinking=False, LLM_Type="Coding").build_config()


system_message="""
You must output the following code blocks exactly as shown below:

```python
import ffmpeg
import numpy as np
import matplotlib.pyplot as plt

# Create simple test data
x = np.limspace(0, 10, 100)
y = np.sin(x)

# Create a plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Test Plot')
plt.legend()
plt.savefig('/workspace/test_plot.png')
print("Plot saved successfully!")
print(f"Array sum: {np.sum(y)}")

```

"""
context=ContextVariables({"Send": True,
"Issues": False})
Manager=LLM_Manager(LLM_Type="Coding")
Manager.Manage_VLLM()
LLM_Config=VLLM_Config(api_type="openai", cache_seed=None, temperature=0.2, enable_thinking=False, LLM_Type="Coding").build_config()

Sender_Agent = ConversableAgent(
    name="Sender",
    system_message=system_message,
    llm_config=LLM_Config,
    context_variables=context,
)
executor_agent = ConversableAgent(
    name="Executor",
    llm_config=False,  # No LLM needed
    code_execution_config={"executor": executor},
    human_input_mode="NEVER",
)
def Execution_Results(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool
) -> Union[dict[str, Any], str]:
    """Hook to check code execution results and update context variables."""
    # Extract content from message
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = str(message)
    
    # Check for execution errors
    if "exitcode: 1" in content or "execution failed" in content.lower():
        sender.context_variables["Issues"] = True
        # Modify the message to include error info
        if isinstance(message, dict):
            message["content"] = f"CODE EXECUTION FAILED:\n{content}\n\nPlease fix the code."
            return message
        else:
            return f"CODE EXECUTION FAILED:\n{content}\n\nPlease fix the code."
    
    if "exitcode: 0" in content:
        sender.context_variables["Issues"] = False
        # Modify the message to include success info  
        if isinstance(message, dict):
            message["content"] = f"CODE EXECUTION SUCCESSFUL:\n{content}"
            return message
        else:
            return f"CODE EXECUTION SUCCESSFUL:\n{content}"
    
    # Return message unchanged if no execution result found
    return message
executor_agent.register_hook("process_message_before_send",Execution_Results)

pattern=DefaultPattern(
    initial_agent=Sender_Agent,
    agents=[Sender_Agent, executor_agent],
    group_manager_args={"llm_config": LLM_Config},
    context_variables=context,
)

Sender_Agent.handoffs.set_after_work(AgentTarget(executor_agent))

executor_agent.handoffs.add_after_work(
    OnContextCondition(
        target=AgentTarget(Sender_Agent),
        condition=ExpressionContextCondition(
            expression=ContextExpression("${Issues}==True")
        )
    )
)


result, ctx, _ = initiate_group_chat(
    pattern=pattern,
    messages="Send the message you were given.",
    max_rounds=50,
)
print(result)
print(context)