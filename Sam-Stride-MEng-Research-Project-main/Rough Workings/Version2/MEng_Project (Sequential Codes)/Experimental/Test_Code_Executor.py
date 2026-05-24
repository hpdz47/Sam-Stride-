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

```sh
pip install numpy matplotlib
```

```python
import numpy as np
import matplotlib.pyplot as plt

# Create simple test data
x = np.linspace(0, 10, 100)
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


executor_agent = ConversableAgent(
    name="Executor",
    llm_config=False,  # No LLM needed
    code_execution_config={"executor": executor},
    human_input_mode="NEVER",
)

reply = executor_agent.generate_reply(messages=[{"role": "user", "content": system_message}])
print(reply)
