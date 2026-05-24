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
from Utils.Git_Manager import GitManager
#----- Functions for Agents
from Tools.Review_Panel_Tools import Review_RAG_Questions_Hook, Review_RAG_Interpret_Hook, Compiler_Hook, Plan_Score, Var_Feedback_Hook, FA_Feedback_Hook, OP_Feedback_Hook, CTX_Feedback_Hook, Adapter_Hook
#----------------------------

# 1 ---------------------------------------
class Variables_Chatroom:
    def __init__(self,context_variables: ContextVariables, Max_Rounds:int=30):
        print("Variables Review in Progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        #------- Agents:
        self.Variable_Checker = Agent_Factory(agent_name="Variable_Checker", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        #------ Functions:
        self.Variable_Checker.register_hook("process_message_before_send",Var_Feedback_Hook)
        #----- Handoffs:
        self.Variable_Checker.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Var_Review_Available} == True")
                )
            )
        )
        self.Variable_Checker.handoffs.set_after_work(AgentTarget(self.Variable_Checker))

    def run_Conversation(self):
        # Reset Context Variables First:
        self.context_variables["Var_Review"] = {}
        self.context_variables["Var_Review_Available"] = False
        pattern=DefaultPattern(
        initial_agent=self.Variable_Checker,
        agents=[self.Variable_Checker],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Review the current analysis plan with a focus on the suitability of the variables suggested for each analysis step. Provide detailed feedback on any issues identified and suggestions for improvement. Consider the structure of the dataset in your review.",
            max_rounds=self.Max_Rounds,
        )

        Judge = Judgement(context_variables=self.context_variables, Max_Rounds=15, Specialism="Var")
        Judge.run_Conversation()
        
        if self.context_variables["Short_Term_Memory"] == True:
            Adapt = Adapter(context_variables=self.context_variables, Max_Rounds=15, Specialism="Var")
            Adapt.run_Conversation()
        else:
            print("\n Short Term Memory is disabled, skipping the adapter \n")
        return result, ctx
# 2----------------------------------------------
class Focus_Area_Assessment:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int = 30):
        print("Checking User Requirements and Assessing Alignment ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        #------ Agents:
        self.Focus_Area_Assessor = Agent_Factory(agent_name="Focus_Area_Assessor", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        # ------------ Function Registration:
        self.Focus_Area_Assessor.register_hook("process_message_before_send",FA_Feedback_Hook)
        #------------------- Handoffs:
        self.Focus_Area_Assessor.handoffs.set_after_work(AgentTarget(self.Focus_Area_Assessor))

        self.Focus_Area_Assessor.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${FA_Reviews_Available} == True")
                )
            )
        )
    def run_Conversation(self):
        # Reset Context Variables First:
        self.context_variables["FA_Reviews"] = ""
        self.context_variables["FA_Reviews_Available"] = False

        pattern=DefaultPattern(
        initial_agent=self.Focus_Area_Assessor,
        agents=[self.Focus_Area_Assessor],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Review the current analysis plan with a focus on the suitability of the focus areas suggested for each analysis step. Provide detailed feedback on any issues identified and suggestions for improvement.",
            max_rounds=self.Max_Rounds,
        )

        Judge = Judgement(context_variables=self.context_variables, Max_Rounds=15, Specialism="FA")
        Judge.run_Conversation()

        if self.context_variables["Short_Term_Memory"] == True:
            Adapt = Adapter(context_variables=self.context_variables, Max_Rounds=15, Specialism="FA")
            Adapt.run_Conversation()
        else:
            print("\n Short Term Memory is disabled, skipping the adapter \n")
        return result, ctx
# 3 ---------------------------------------
class RAG_System:
    def __init__(self,context_variables: ContextVariables, Max_Rounds:int=30):
        print("Researching Documents ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        #------- Agents:
        self.Review_RAG_Questioner = Agent_Factory(agent_name="Review_RAG_Questioner", context_variables=self.context_variables).BuildAgent()
        self.Review_RAG_Interpreter = Agent_Factory(agent_name="Review_RAG_Interpreter", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        #------ Functions:
        self.Review_RAG_Questioner.register_hook("process_message_before_send", Review_RAG_Questions_Hook)

        self.Review_RAG_Interpreter.register_hook("process_message_before_send", Review_RAG_Interpret_Hook)
        #----- Handoffs:
        self.Review_RAG_Questioner.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Review_RAG_Interpreter),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Review_RAG_Available} == True")
                )
            )
        )
        self.Review_RAG_Questioner.handoffs.set_after_work(AgentTarget(self.Review_RAG_Questioner))

        self.Review_RAG_Interpreter.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Review_RAG_Interpret_Available} == True")
                )
            )
        )
        self.Review_RAG_Interpreter.handoffs.set_after_work(AgentTarget(self.Review_RAG_Interpreter))

    def run_Conversation(self):
        # Reset Context Variables First:
        self.context_variables["Review_RAG_QA"] = ""
        self.context_variables["Review_RAG_Available"] = False
        self.context_variables["Review_RAG_Interpret"] = ""
        self.context_variables["Review_RAG_Interpret_Available"] = False
        pattern=DefaultPattern(
        initial_agent=self.Review_RAG_Questioner,
        agents=[self.Review_RAG_Questioner, self.Review_RAG_Interpreter],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Ask domain specific questions to gather crucial information required to support the analysis plan.",
            max_rounds=self.Max_Rounds,
        )
        return result, ctx
# 4 ---------------------------------------
class Output_Instruction():
    def __init__(self,context_variables: ContextVariables, Max_Rounds:int=30):
        print("Reviewing Output Format Requirements ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        #------- Agents:
        self.Review_OP = Agent_Factory(agent_name="Review_OP", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        #------ Functions:
        self.Review_OP.register_hook("process_message_before_send",OP_Feedback_Hook)
        #----- Handoffs:
        self.Review_OP.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${OP_Instruction_Available} == True")
                )
            )
        )
        self.Review_OP.handoffs.set_after_work(AgentTarget(self.Review_OP))

    def run_Conversation(self):
        # Reset Context Variables First:
        self.context_variables["Review_OP_Instruction"] = ""
        self.context_variables["OP_Instruction_Available"] = False
        pattern=DefaultPattern(
        initial_agent=self.Review_OP,
        agents=[self.Review_OP],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Review the output format requirements for the data analysis plan considering the structure of the dataset.",
            max_rounds=self.Max_Rounds,
        )

        Judge = Judgement(context_variables=self.context_variables, Max_Rounds=15, Specialism="OP")
        Judge.run_Conversation()

        if self.context_variables["Short_Term_Memory"] == True:
            Adapt = Adapter(context_variables=self.context_variables, Max_Rounds=15, Specialism="OP")
            Adapt.run_Conversation()
        else:
            print("\n Short Term Memory is disabled, skipping the adapter \n")
        return result, ctx

# 5 ---------------------------------------------
class Context_Review:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int = 30):
        print("Assessing Contextual Alignment ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        #------ Agents:
        self.Context_Assessor = Agent_Factory(agent_name="Context_Assessor", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        # ------------ Function Registration:
        self.Context_Assessor.register_hook("process_message_before_send",CTX_Feedback_Hook)
        #------------------- Handoffs:
        self.Context_Assessor.handoffs.set_after_work(AgentTarget(self.Context_Assessor))

        self.Context_Assessor.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Context_Review_Available} == True")
                )
            )
        )
    def run_Conversation(self):
        # Reset Context Variables First:
        self.context_variables["Context_Review"] = ""
        self.context_variables["Context_Review_Available"] = False

        pattern=DefaultPattern(
        initial_agent=self.Context_Assessor,
        agents=[self.Context_Assessor],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Provide an analysis of the contextual alignment between the analysis type and the implementation instructions in the plan.",
            max_rounds=self.Max_Rounds,
        )

        Judge = Judgement(context_variables=self.context_variables, Max_Rounds=15, Specialism="Ctx")
        Judge.run_Conversation()
        
        if self.context_variables["Short_Term_Memory"] == True:
            Adapt = Adapter(context_variables=self.context_variables, Max_Rounds=15, Specialism="Ctx")
            Adapt.run_Conversation()
        else:
            print("\n Short Term Memory is disabled, skipping the adapter \n")
        return result, ctx

# 6 ----------------------------------------------
class Compile_Review:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int = 30):
        print("Compiling Plan Reviews ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        #------ Agents:
        self.Compiler = Agent_Factory(agent_name="Compiler", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        # ------------ Function Registration:
        self.Compiler.register_hook("process_message_before_send", Compiler_Hook)
        #------------------- Handoffs:
        self.Compiler.handoffs.set_after_work(AgentTarget(self.Compiler))

        self.Compiler.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Review_Compilation_Complete} == True")
                )
            )
        )
    def run_Conversation(self):
        
        # Reset Context Variables First:
        self.context_variables["Plan_Feedback"] = ""
        self.context_variables["Review_Compilation_Complete"] = False

        pattern=DefaultPattern(
        initial_agent=self.Compiler,
        agents=[self.Compiler],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Compile all feedback provided by individual specalists into detailed instructions for improving the data analysis plan.",
            max_rounds=self.Max_Rounds,
        )
        return result, ctx

# 7 ----------------------------------------------
class Judgement:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int, Specialism: str):
        print("Plan Judgement in progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds
        self.allowed_specialisms = ["Var", "FA", "OP", "Ctx"]
        if Specialism not in self.allowed_specialisms:
            raise ValueError(f"Invalid Specialism. Allowed values are: {self.allowed_specialisms}")
        self.Specialism = Specialism
        self.Idx = None # NOTE: Will be updated based on the specialism area. This is the index that the score will be stored at. See BioAgent for the agent idx assignments.

        # NOTE: Slecting the correct specialist review based on the specialism area required for the judgement.
        # NOTE: This allows the same judgement agent class to be called in different settings for the Plan Review stage.
        if self.Specialism == "Var":
            self.Idx = 0
            self.context_variables["Idx"] = self.Idx
            self.context_variables["Reviews"] = self.context_variables["Var_Review"] # Assigns the correct specialist review based on the specialism area required.
        elif self.Specialism == "FA":
            self.Idx = 1
            self.context_variables["Idx"] = self.Idx
            self.context_variables["Reviews"] = self.context_variables["FA_Reviews"]
        elif self.Specialism == "OP":
            self.Idx = 2
            self.context_variables["Idx"] = self.Idx
            self.context_variables["Reviews"] = self.context_variables["Review_OP_Instruction"]
        elif self.Specialism == "Ctx":
            self.Idx = 3
            self.context_variables["Idx"] = self.Idx
            self.context_variables["Reviews"] = self.context_variables["Context_Review"]
            # Here Judge does not need extra information.

        #------ Agents:
        self.Plan_Judge = Agent_Factory(agent_name="Plan_Judge", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        # ------------ Function Registration:
        register_function(
            Plan_Score,
            caller=self.Plan_Judge,
            executor=self.Plan_Judge,
            name="Plan_Score",
            description="Use the specialist feedback to produce a score that reflects the quality of the plan."
        )
        #------------------- Handoffs:
        self.Plan_Judge.handoffs.set_after_work(AgentTarget(self.Plan_Judge))

        self.Plan_Judge.handoffs.add_context_condition(
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
        initial_agent=self.Plan_Judge,
        agents=[self.Plan_Judge],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Use the specialist feedback to produce a score that reflects the quality of the plan.",
            max_rounds=self.Max_Rounds,
        )
        return result, ctx

# 8  ----------------------------------------------
class Adapter:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int, Specialism: str):
        print("Considering Improvement Directions ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds
        self.allowed_specialisms = ["Var", "FA", "OP", "Ctx"]
        if Specialism not in self.allowed_specialisms:
            raise ValueError(f"Invalid Specialism. Allowed values are: {self.allowed_specialisms}")
        self.Specialism = Specialism
        self.Idx = None # NOTE: Will be updated based on the specialism area. This is the index that the score will be stored at. See BioAgent for the agent idx assignments.
        self.Manager = None # NOTE: This will be assigned the correct GitManager based on the specialism area. This allows the same Adapter agent class to be used for different specialism areas in the Plan Review stage.

        # NOTE: Selecting the correct specialist review based on the specialism area required for the judgement.
        # NOTE: This allows the same judgement agent class to be called in different settings for the Plan Review stage.
        if self.Specialism == "Var":
            self.context_variables["Specialist_Reviews"] = self.context_variables["Var_Review"] # Assigns the correct specialist review based on the specialism area required.
            self.context_variables["Specialist_Improvements"] = self.context_variables["Var_Improvements"] # Assigns the correct specialist improvements based on the specialism area required.
            self.Manager = self.context_variables["Var_Manager"] # Assigns the correct GitManager based on the specialism area required.
            self.context_variables["Specialist_Learnings"] = self.context_variables["Var_Learnings"]
            self.context_variables["Specialism"] = "Var"
            self.Idx = 0
        elif self.Specialism == "FA":
            self.context_variables["Specialist_Reviews"] = self.context_variables["FA_Reviews"]
            self.context_variables["Specialist_Improvements"] = self.context_variables["FA_Improvements"]
            self.Manager = self.context_variables["FA_Manager"]
            self.context_variables["Specialist_Learnings"] = self.context_variables["FA_Learnings"]
            self.context_variables["Specialism"] = "FA"
            self.Idx = 1
        elif self.Specialism == "OP":
            self.context_variables["Specialist_Reviews"] = self.context_variables["Review_OP_Instruction"]
            self.context_variables["Specialist_Improvements"] = self.context_variables["OP_Improvements"]
            self.Manager = self.context_variables["Output_Manager"]
            self.context_variables["Specialist_Learnings"] = self.context_variables["OP_Learnings"]
            self.context_variables["Specialism"] = "OP"
            self.Idx = 2
        elif self.Specialism == "Ctx":
            self.context_variables["Specialist_Reviews"] = self.context_variables["Context_Review"]
            self.context_variables["Specialist_Improvements"] = self.context_variables["Ctx_Improvements"]
            self.Manager = self.context_variables["Context_Manager"]
            self.context_variables["Specialist_Learnings"] = self.context_variables["Context_Learnings"]
            self.context_variables["Specialism"] = "Ctx"
            self.Idx = 3
        
        #================== Use the Correct GitManager to Update the Memory Here =========================
        # ----- Obtain Relevant Score.
        self.Plan_Idx = context_variables["Plan_Review_Count"]
        self.Score = context_variables[f"Plan_Score_{self.Plan_Idx}"][self.Idx]
        # ------- Create dict for Outcome Field:
        self.Outcome = self.context_variables["Specialist_Reviews"]
        self.Outcome["Score"] = self.Score
        self.Improvements = self.context_variables["Specialist_Improvements"]
        self.Outcome["Improvements"] = self.Improvements
        print(f"""\n\n Testing Adapter Outcome: {self.Outcome} \n\n""") # For Debugging.
        self.Manager.jog_memory() # Must be done to ensure the GitMemory is pointng to the correct repo. GitMemory reads from the repo file to obtain current history.
        self.Approach = self.context_variables["Plan_Fix_Approach"]
        self.Manager.commit(approach = self.Approach, outcome = self.Outcome, error_message = None) # Commiting the review to the GitMemory history for that repo.

        #TODO: Fix the approach by adding in some kind of diff agent after plan fixer.
        self.context_variables["Plan_Fix_Approach"] = "Fix applied based on last reviews."
        #============================== Obtaining the Correct Review History and Saving to Context Variables for Adapter Agent.
        self.context_variables["Plan_History"] = self.Manager.retrieve_history()
        #print(f"""\n\n Testing: Plan History \n{self.context_variables["Plan_History"]}\n\n """) # For Debugging.
        #------ Agents:
        self.Plan_Adapter = Agent_Factory(agent_name="Plan_Adapter", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        # ------------ Function Registration:
        self.Plan_Adapter.register_hook("process_message_before_send",Adapter_Hook)
        #------------------- Handoffs:
        self.Plan_Adapter.handoffs.set_after_work(AgentTarget(self.Plan_Adapter))

        self.Plan_Adapter.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Reviews_Adapted} == True")
                )
            )
        )
    def run_Conversation(self):
        # Reset Context Variables First:
        self.context_variables["Reviews_Adapted"] = False

        pattern=DefaultPattern(
        initial_agent=self.Plan_Adapter,
        agents=[self.Plan_Adapter],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Use the specialist feedback and the feedback and improvement history to adapt the instructions for improving the plan accordingly.",
            max_rounds=self.Max_Rounds,
        )
        return result, ctx

def main():
    context_variables = ContextVariables({
        "Var_Review": {},
        "Var_Review_Available": False,
        "FA_Reviews": {},
        "FA_Reviews_Available": False,
        "Review_RAG_QA": "",
        "Review_RAG_Available": False,
        "Review_RAG_Interpret": "",
        "Review_RAG_Interpret_Available": False,
        "Review_OP_Instruction": {},
        "OP_Instruction_Available": False,
        "Plan_Feedback": {},
        "Review_Compilation_Complete": False,
        "Context_Review": {}, 
        "Context_Review_Available": False, 
        "Reviews": {},
        "Scoring_Complete": False, 
        "Idx": 0,
        "Plan_History": {}, 
        "Specialist_Reviews": {},
        "Specialism": "",
        "Reviews_Adapted": False,
        "Plan_Fix_Approach": "First plan commit. Plan created. No previous plan versions to apply fixes to.",
        "Var_Learnings": {},
        "FA_Learnings": {},
        "OP_Learnings": {},
        "Context_Learnings": {},
        "Var_Improvements": {},
        "Specialist_Improvements": {},
        "FA_Improvements": {},
        "OP_Improvements": {},
        "Ctx_Improvements": {},
        "Specialist_Learnings": {},
    })
    # Unit Tests Here:

    Review=Variables_Chatroom(context_variables=context_variables, Max_Rounds=15)
    Review.run_Conversation()
    