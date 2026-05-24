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
#----- Functions for Agents
from Tools.Discovery_Phase import EDA_Hook, Profile_Check, Deterministic_EDA
#----------------------------------------------------------------------------------------

class EDA_Analysis:
    def __init__(self,context_variables: ContextVariables, Max_Rounds: int):
        print("Data Discovery in Progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds
        self.inputs_dir = Path("/inputs")

        #------- Agents:
        self.Discovery_Interpreter= Agent_Factory(agent_name="Discovery_Interpreter", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()


        #------ Function Registration:
        self.Discovery_Interpreter.register_hook("process_message_before_send",EDA_Hook)
        #------ Handoffs:
        self.Discovery_Interpreter.handoffs.set_after_work(AgentTarget(self.Discovery_Interpreter))

        self.Discovery_Interpreter.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Interpretation_Available} == True")
                )
            )
        )

    def run_Conversation(self):
        #----- First Run Functions to get initial dataset information:
        # Profile_Check => Metadata (No data security concern here for research stage)
        # Deterministic EDA => Contains numeric summaries of dataset and not suitable for research due to data security).
        Profile_Check(context_variables=self.context_variables, Input_Dir= self.inputs_dir)
        Deterministic_EDA(context_variables=self.context_variables, Input_Dir=self.inputs_dir)
        #---------------------

        pattern=DefaultPattern(
        initial_agent=self.Discovery_Interpreter,
        agents=[self.Discovery_Interpreter],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )

        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Use the EDA results to generate a detailed report that would enable a comprehensive understanding of the dataset without reading it line by line.",
            max_rounds=self.Max_Rounds,
        )

        # NOTE: Uncomment to check context variable. AG2 will overwrite the context variable using the hok when it says [Handing Off to ....].
        # NOTE: Ths check is to make sure that the hook guardrail is working.
        #print(f"""\n \n Testing: \n \n {self.context_variables["EDA_Interpretation"]} """)
        return result, ctx

def main():
    context_variables = ContextVariables({
        # "Discovery_Code": "", (Not Needed ANYMORE !!!!!!)
        # "Discovery_Reviews": "",
        # "Discovery_Code_Updated": False,
        # "Discovery_Reviews_Available": False,
        # "Discovery_Code_Approval": False,
        # "Discovery_Revision_Count": 0,
        # "Issues": False,
        "EDA_Results": [],
        "EDA_Interpretation": "",
        "Interpretation_Available": False,
        "metadata": {},
        # Additional Context Variables for Updated EDA stage.

    })

    eda1=EDA_Analysis(context_variables=context_variables)
    eda1.run_Conversation()
if __name__ == "__main__":
    main()

