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
from Tools.Focus_Area_and_Research_Phase import Focus_Area_Hook, FA_RAG_Questions_Hook, FA_RAG_Interpret_Hook
from Tools.RAG_Tools import ingest
from Deep_Research_Chatroom import DR_Chatroom
#---------------------------

# Deep Research Agent not implemented yet.

class Focus_Area_and_Research_Chatroom:
    def __init__(self,context_variables: ContextVariables, Deep_Research_Mode: bool=False, Max_Rounds:int=30):
        print("Focus Area and Research Phase in Progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM() 
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds
        self.Deep_Research_Mode = Deep_Research_Mode

        ROOT_DIR = Path("/workspace")

        if not Deep_Research_Mode:
            print("Indexing any information provided for the RAG system")
            ingest(f"{ROOT_DIR}/Research_and_Documents/Deep_Research_Report.md")
        else:
            pass
        #------- Agents:
        self.Focus_Area = Agent_Factory(agent_name="Focus_Area", context_variables=self.context_variables).BuildAgent()
        self.RAG_Questioner= Agent_Factory(agent_name="RAG_Questioner", context_variables=self.context_variables).BuildAgent()
        self.RAG_Interpreter= Agent_Factory(agent_name="RAG_Interpreter", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()
        

        # --- Function Registration:
        self.Focus_Area.register_hook("process_message_before_send", Focus_Area_Hook)

        self.RAG_Questioner.register_hook("process_message_before_send", FA_RAG_Questions_Hook)

        self.RAG_Interpreter.register_hook("process_message_before_send", FA_RAG_Interpret_Hook)


        #------ Handoffs:
        self.Focus_Area.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.RAG_Questioner),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${FA_Available} == True")
                )
            )
        )

        self.Focus_Area.handoffs.set_after_work(AgentTarget(self.Focus_Area))

        self.RAG_Questioner.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.RAG_Interpreter),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${FA_RAG_Available} == True")
                )
            )
        )
        self.RAG_Questioner.handoffs.set_after_work(AgentTarget(self.RAG_Questioner))

        self.RAG_Interpreter.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Focus_Area),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${FA_RAG_Interpretation_Available} == True")
                )
            )
        )
        self.RAG_Interpreter.handoffs.set_after_work(AgentTarget(self.RAG_Interpreter))

    def run_Conversation(self):

        if self.Deep_Research_Mode:
            DR1 = DR_Chatroom(context_variables=self.context_variables, Max_Rounds=15)
            DR1.run_Conversation()
       
        pattern=DefaultPattern(
        initial_agent=self.Focus_Area,
        agents=[self.Focus_Area, self.RAG_Questioner, self.RAG_Interpreter],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Discover the data in the directory.",
            max_rounds=self.Max_Rounds,
        )


        # NOTE: Uncomment to check context variable. AG2 will overwrite the context variable using the hok when it says [Handing Off to ....].
        # NOTE: Ths check is to make sure that the hook guardrail is working.
        #print(f"""\n \n Testing: \n \n {self.context_variables["Focus_Area_Statement"]} \n\n {self.context_variables["FA_RAG_QA"]} \n\n {self.context_variables["FA_RAG_Interpretation"]}""")

        return result, ctx

def main():
    context_variables = ContextVariables({
        "EDA_Interpretation": "",
        "User_Requirements": "", # Import from text file to represent a user uploading requirements. This would be used by a User in a ChatGPT-style interface.
        "Focus_Area_Statement": "",
        "FA_Available": False,
        "FA_RAG_QA": "",
        "FA_RAG_Available": False,
        "FA_RAG_Interpretation": "",
        "FA_RAG_Interpretation_Available": False,
    })

    FR1=Focus_Area_and_Research_Chatroom(context_variables=context_variables, Deep_Research_Mode=False, Max_Rounds=15)
    FR1.run_Conversation()
if __name__ == "__main__":
    main()
