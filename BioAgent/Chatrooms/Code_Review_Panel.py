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
from Tools.Code_Review_Panel_Tools import Error_Check_Hook, Enforcement_Hook, Optimisation_Hook, Code_Adapter_Hook, Code_Score

# TODO: Add a RAG System. This should be spread across several RAG systems for dedicated python packages. For Example, one for HPLC-py, one for MOCCA, etc. Questioner and Interpreter is a must. Overall, a final RAG Assessor should decide which python package to be used in the current implementation. It should manage any conflicts before sending to the coder).
# NOTE: Compiler NOT requied here as agents produce a concise list. Unlike the plan, conflicting opinions are not as likely.
# 1 ----------------------------------------------
class Error_Check_Loop:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int):
        print("\nChecking Code for Errors ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        #------ Agents:
        self.Error_Checker = Agent_Factory(agent_name="Error_Checker", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        # ------------ Function Registration:
        self.Error_Checker.register_hook("process_message_before_send",Error_Check_Hook)
        #------------------- Handoffs:
        self.Error_Checker.handoffs.set_after_work(AgentTarget(self.Error_Checker))

        self.Error_Checker.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Errors_Checked} == True")
                )
            )
        )

    def run_Conversation(self):
        # Reset Context Variables First:
        self.context_variables["Code_Errors"] = ""
        self.context_variables["Errors_Checked"] = False

        pattern=DefaultPattern(
        initial_agent=self.Error_Checker,
        agents=[self.Error_Checker],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Check the code for any errors and then record the errors in a concise list.",
            max_rounds=self.Max_Rounds,
        )

        Code_Judge = Code_Judgement(context_variables=self.context_variables, Max_Rounds=15, Specialism="ECL")
        Code_Judge.run_Conversation()

        if self.context_variables["Code_Short_Term_Memory"]: # Short Term Memory ON.
            Code_Adapt = Code_Adapter(context_variables=self.context_variables, Max_Rounds=15, Specialism="ECL")
            Code_Adapt.run_Conversation()
        else:
            print("\n\nCode Short Term Memory is OFF. Skipping Code Adaptation Stage. \n")

        return result, ctx
# 2 ----------------------------------------------
class Plan_Enforcement_Loop:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int):
        print("\nChecking Code follows Plan ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        #------ Agents:
        self.Plan_Enforcer = Agent_Factory(agent_name="Plan_Enforcer", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        # ------------ Function Registration:
        self.Plan_Enforcer.register_hook("process_message_before_send",Enforcement_Hook)
        #------------------- Handoffs:
        self.Plan_Enforcer.handoffs.set_after_work(AgentTarget(self.Plan_Enforcer))

        self.Plan_Enforcer.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Enforcement_Complete} == True")
                )
            )
        )
    def run_Conversation(self):
        # Reset Context Variables First:
        self.context_variables["Plan_Enforcement"] = ""
        self.context_variables["Enforcement_Complete"] = False

        pattern=DefaultPattern(
        initial_agent=self.Plan_Enforcer,
        agents=[self.Plan_Enforcer],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Check the code for any misalignments with the plan and then record the misalignments in a concise list.",
            max_rounds=self.Max_Rounds,
        )

        Code_Judge = Code_Judgement(context_variables=self.context_variables, Max_Rounds=15, Specialism="PEL")
        Code_Judge.run_Conversation()

        if self.context_variables["Code_Short_Term_Memory"]: # Short Term Memory ON.
            Code_Adapt = Code_Adapter(context_variables=self.context_variables, Max_Rounds=15, Specialism="PEL")
            Code_Adapt.run_Conversation()
        else:
            print("\n\nCode Short Term Memory is OFF. Skipping Code Adaptation Stage. \n")

        return result, ctx

# 3 ----------------------------------------------
class Optimisation_Loop:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int):
        print("\nCode Optimisation in progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        #------ Agents:
        self.Code_Optimiser = Agent_Factory(agent_name="Code_Optimiser", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        # ------------ Function Registration:
        self.Code_Optimiser.register_hook("process_message_before_send",Optimisation_Hook)
        #------------------- Handoffs:
        self.Code_Optimiser.handoffs.set_after_work(AgentTarget(self.Code_Optimiser))

        self.Code_Optimiser.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Optimisation_Assessed} == True")
                )
            )
        )
    def run_Conversation(self):
        # Reset Context Variables First:
        self.context_variables["Optimisation_Goals"] = ""
        self.context_variables["Optimisation_Assessed"] = False

        pattern=DefaultPattern(
        initial_agent=self.Code_Optimiser,
        agents=[self.Code_Optimiser],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Check the code for any issues related to speed and efficiency and record the issues in a concise list.",
            max_rounds=self.Max_Rounds,
        )

        Code_Judge = Code_Judgement(context_variables=self.context_variables, Max_Rounds=15, Specialism="OL")
        Code_Judge.run_Conversation()

        if self.context_variables["Code_Short_Term_Memory"]: # Short Term Memory ON.
            Code_Adapt = Code_Adapter(context_variables=self.context_variables, Max_Rounds=15, Specialism="OL")
            Code_Adapt.run_Conversation()
        else:
            print("\n\nCode Short Term Memory is OFF. Skipping Code Adaptation Stage. \n")

        return result, ctx

# 4 ----------------------------------------------
class Code_Judgement:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int, Specialism: str):
        print("Code Judgement in progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds
        self.allowed_specialisms = ["ECL", "PEL", "OL"]
        if Specialism not in self.allowed_specialisms:
            raise ValueError(f"Invalid Specialism. Allowed values are: {self.allowed_specialisms}")
        self.Specialism = Specialism
        self.Idx = None # NOTE: Will be updated based on the specialism area. This is the index that the score will be stored at. See BioAgent for the agent idx assignments.

        # NOTE: Selecting the correct specialist review based on the specialism area required for the judgement.
        # NOTE: This allows the same judgement agent class to be called in different settings for the Plan Review stage.
        if self.Specialism == "ECL":
            self.Idx = 0
            self.context_variables["Idx"] = self.Idx
            self.context_variables["Code_Reviews"] = self.context_variables["Code_Errors"] # Assigns the correct specialist review based on the specialism area required.
        elif self.Specialism == "PEL":
            self.Idx = 1
            self.context_variables["Idx"] = self.Idx
            self.context_variables["Code_Reviews"] = self.context_variables["Plan_Enforcement"]
        elif self.Specialism == "OL":
            self.Idx = 2
            self.context_variables["Idx"] = self.Idx
            self.context_variables["Code_Reviews"] = self.context_variables["Optimisation_Goals"]

        #------ Agents:
        self.Code_Judge = Agent_Factory(agent_name="Code_Judge", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        # ------------ Function Registration:
        register_function(
            Code_Score,
            caller=self.Code_Judge,
            executor=self.Code_Judge,
            name="Code_Score",
            description="Use the specialist feedback to produce a score that reflects the quality of the code."
        )
        #------------------- Handoffs:
        self.Code_Judge.handoffs.set_after_work(AgentTarget(self.Code_Judge))

        self.Code_Judge.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Scoring_Complete} == True")
                )
            )
        )
    def run_Conversation(self):
        # Reset Context Variables First:
        self.context_variables["Scoring_Complete"] = False

        pattern=DefaultPattern(
        initial_agent=self.Code_Judge,
        agents=[self.Code_Judge],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Use the specialist feedback to produce a score that reflects the quality of the code.",
            max_rounds=self.Max_Rounds,
        )
        return result, ctx

# 5  ----------------------------------------------
class Code_Adapter:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int, Specialism: str):
        print("Considering Improvement Directions ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds
        self.allowed_specialisms = ["ECL", "PEL", "OL"]
        if Specialism not in self.allowed_specialisms:
            raise ValueError(f"Invalid Specialism. Allowed values are: {self.allowed_specialisms}")
        self.Specialism = Specialism
        self.Idx = None # NOTE: Will be updated based on the specialism area. This is the index that the score will be stored at. See BioAgent for the agent idx assignments.
        self.Manager = None # NOTE: This will be assigned the correct GitManager based on the specialism area. This allows the same Adapter agent class to be used for different specialism areas in the Plan Review stage.

        # NOTE: Selecting the correct specialist review based on the specialism area required for the judgement.
        # NOTE: This allows the same judgement agent class to be called in different settings for the Plan Review stage.
        if self.Specialism == "ECL":
            self.context_variables["Code_Specialist_Reviews"] = self.context_variables["Code_Errors"] # Only positives and negatives recorded here.
            self.context_variables["Code_Specialist_Improvements"] = self.context_variables["ECL_Improvements"] # Assigns the correct specialist improvements based on the specialism area required.
            self.Manager = self.context_variables["ECL_Manager"] # Assigns the correct GitManager based on the specialism area required.
            self.context_variables["Code_Specialist_Learnings"] = self.context_variables["ECL_Learnings"]
            self.context_variables["Specialism"] = "ECL"
            self.Idx = 0
        elif self.Specialism == "PEL":
            self.context_variables["Code_Specialist_Reviews"] = self.context_variables["Plan_Enforcement"]
            self.context_variables["Code_Specialist_Improvements"] = self.context_variables["PEL_Improvements"]
            self.Manager = self.context_variables["PEL_Manager"]
            self.context_variables["Code_Specialist_Learnings"] = self.context_variables["PEL_Learnings"]
            self.context_variables["Specialism"] = "PEL"
            self.Idx = 1
        elif self.Specialism == "OL":
            self.context_variables["Code_Specialist_Reviews"] = self.context_variables["Optimisation_Goals"]
            self.context_variables["Code_Specialist_Improvements"] = self.context_variables["OL_Improvements"]
            self.Manager = self.context_variables["OL_Manager"]
            self.context_variables["Code_Specialist_Learnings"] = self.context_variables["OL_Learnings"]
            self.context_variables["Specialism"] = "OL"
            self.Idx = 2
        
        #================== Use the Correct GitManager to Update the Memory Here =========================
        # ----- Obtain Relevant Score.
        self.Code_Idx = context_variables["Code_Review_Count"]
        self.Score = context_variables[f"Code_Score_{self.Code_Idx}"][self.Idx]
        # ------- Create dict for Outcome Field:
        self.Outcome = self.context_variables["Code_Specialist_Reviews"]
        #self.Outcome = self.Outcome # Convert the stringified dict back to a dict format.
        self.Outcome["Score"] = self.Score
        self.Improvements = self.context_variables["Code_Specialist_Improvements"]
        self.Outcome["Improvements"] = self.Improvements
        #print(f"""\n\n Testing Adapter Outcome: {self.Outcome} \n\n""") # For Debugging.
        self.Manager.jog_memory() # Must be done to ensure the GitMemory is pointng to the correct repo. GitMemory reads from the repo file to obtain current history.
        self.Approach = self.context_variables["Code_Fix_Approach"]
        self.Manager.commit(approach = self.Approach, outcome = self.Outcome, error_message = None) # Commiting the review to the GitMemory history for that repo.

        #TODO: Fix the approach by adding in some kind of diff agent after plan fixer.
        self.context_variables["Code_Fix_Approach"] = "Fix applied based on last reviews."
        #============================== Obtaining the Correct Review History and Saving to Context Variables for Adapter Agent.
        self.context_variables["Code_History"] = self.Manager.retrieve_history()
        #------ Agents:
        self.Code_Adapter = Agent_Factory(agent_name="Code_Adapter", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        # ------------ Function Registration:
        self.Code_Adapter.register_hook("process_message_before_send",Code_Adapter_Hook) # TODO: Create Hook and Import it.
        #------------------- Handoffs:
        self.Code_Adapter.handoffs.set_after_work(AgentTarget(self.Code_Adapter))

        self.Code_Adapter.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Code_Reviews_Adapted} == True")
                )
            )
        )
    def run_Conversation(self):
        # Reset Context Variables First:
        self.context_variables["Code_Reviews_Adapted"] = False

        pattern=DefaultPattern(
        initial_agent=self.Code_Adapter,
        agents=[self.Code_Adapter],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Use the specialist feedback and the feedback and improvement history to adapt the instructions for improving the code accordingly.",
            max_rounds=self.Max_Rounds,
        )
        return result, ctx

def main():
    context_variables = ContextVariables({
        "Code_Errors": "",
        "Errors_Checked": False,
        "Plan_Enforcement": "",
        "Enforcement_Complete": False,
        "Optimisation_Goals": "",
        "Optimisation_Assessed": False,
        "Code_Short_Term_Memory": False,
        # Git Based Review Loops:
        "Code_Specialist_Learnings": "",
        "Code_History": "",
        "Code_Specialist_Reviews": "",
        "Code_Specialist_Improvements": "",
        "PEL_Learnings": "",
        "PEL_Improvements": "",
        "ECL_Learnings": "",
        "ECL_Improvements": "",
        "OL_Learnings": "",
        "OL_Improvements": "",
        "Code_Fix_Approach": "First code commit. Code created. No previous code versions to apply fixes to.",
        "Code_Reviews_Adapted": False,
        "Code_Reviews": "",

    })
    # Unit Tests Here:

if __name__ == "__main__":
    main()