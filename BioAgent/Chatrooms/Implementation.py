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
from Utils.Secure_LocalCommandLineExecutor import SecureLocalCommandLineExecutor
from pathlib import Path
from autogen import Agent
import copy
import pprint
import re
import shutil
import ast
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from Agents.AgentFactory import Agent_Factory
load_dotenv()

from Utils.Git_Manager import GitManager
from Utils.Git_Memory_Handling import GitMemory
from Utils.Code_Repo_Management import create_code_git_managers
#----- Functions for Agents
from Tools.Implementation_Phase import Coder_Hook, Code_Fixer_Hook, Code_Fixer_Hook_No_Memory, Code_Diff_Hook, Summary_Hook, Git_Control_Hook, Debug_Hook, Installations_Hook, Analysis_Hook

#================ Notes for File ===========================
#The granularity has been ignored for now as it makes sense 
#              for 1 step at a time.
#===========================================================

#=================== GitMemory and GitManager Note ==========================
# NOTE: GitMemory will be created as a Singleton in BioAgent, then pass to
# NOTE: be coupled up with GitManager. GitManager will be instantiated in
# NOTE: FINAL class that ties all other classes together, and the instance will
# NOTE: be passed to the control phase so that it can use git and pass its functions.
#==============================================================================

#==================== Results Note ===================================================
# NOTE: The main results are now aimed at the code success rate and the code debug
# NOTE: attempts. If code is successful, the result will be saved as a 1, otherwise
# NOTE: it will be saved as a 0. If the code is saved as a 0, the max retries can
# NOTE: artificially increase the debug attempts metric and obscure it, hence debug
# NOTE: attempts will not be recorded in this case.
#======================================================================================

# Evaluation Metric Code:
def cyclomatic_complexity(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0

    complexity = 1  # Initial Complexity

    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp, ast.With, ast.AsyncWith, ast.Assert)):
            complexity += 1
        
        elif isinstance(node, ast.Match):
            complexity += len(node.cases) - 1
            for case in node.cases:
                if case.guard is not None:
                    complexity += 1
        
        elif isinstance(node, ast.comprehension):
            complexity += 1
            complexity += len(node.ifs)

        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1

    return complexity

def nesting_depth(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0

    nesting_nodes = (ast.If, ast.For, ast.While, ast.FunctionDef)

    def depth(node):
        current = 1 if isinstance(node, nesting_nodes) else 0
        child_depths = [depth(child) for child in ast.iter_child_nodes(node)]
        return current + (max(child_depths) if child_depths else 0)

    return depth(tree)

def code_size(code: str) -> int:
    """
    Return the number of statement nodes in the code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0

    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            count += 1

    return count


#1 ----- Initial_Loop
class Initial_Loop:
    def __init__(self,context_variables: ContextVariables, Max_Rounds:int, Plan_Step:int, Max_Review_Loops: int, LLM_Name: str, Review_Agent_Number: int, Git_Memory: GitMemory):
        print("\nInitial Code-Review Loop in progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.LLM_Name = LLM_Name
        self.LLM_Name = self.LLM_Name.replace('/','_') # This ensures no / in LLM name as it is confusing for file naming
        self.ROOT_DIR = Path("/workspace")
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds # This is to prevent an infinite loop in case there are any erros calling the functions.
        self.Max_Review_Loops = Max_Review_Loops # This is to prevent reviewer from causing excessive time wastage. (-1 is to account for when the variable is updated)
        # Need to put Max_Review_Loops into the context variables to enable the handoff to work:
        self.context_variables["Max_Review_Loops"] = self.Max_Review_Loops
        self.Plan_Step = Plan_Step -1 # To enable the selection of only the relevant part of the plan to
        #                            pass to the agent. The -1 is to account for the 0 indexing in python.
        self.context_variables["Plan_Section"]= self.context_variables["Plan"]["Plan_Section"][self.Plan_Step]
        print(self.context_variables["Plan_Section"]) # For Debugging.
        # Reset Code context variables in case of artifacts from previous runs.
        self.context_variables["Code_Review_Count"] = 0 # To keep track of code reviews.
        self.context_variables["Code"] = ""
        self.context_variables["Code_Updated"] = False 
        self.context_variables["Code_Fixed"] = False
        self.context_variables["Code_Diffs"] = ""
        self.context_variables["Code_Diffs_Analysed"] = False

        # GitManager Setup function for review panel. (Done here so that Repo folder reset can be done at the Global Iteration Level).
        create_code_git_managers(context_variables=self.context_variables, Root_Dir=self.ROOT_DIR, Git_Memory=Git_Memory)
        #          - Git Managers have been added to context variables so the review panel can access them.

        #----------- Resetting Scoring Context Variables:
        self.Review_Agent_Number = Review_Agent_Number
        for idx in range(self.Max_Review_Loops):
            self.context_variables[f"Code_Score_{idx}"] = [0]*self.Review_Agent_Number # Resets all scores as they have been stroed in permanent files if data contained is important. This prevents artifacts from previous global iterations.
        
        #------- Agents:
        self.Coder = Agent_Factory(agent_name="Coder", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Code_Fixer = Agent_Factory(agent_name="Code_Fixer", context_variables=self.context_variables).BuildAgent()
        self.Code_Diff = Agent_Factory(agent_name="Code_Diff", context_variables=self.context_variables).BuildAgent()

        # NOTE: Reviewers are embedded in a review panel.

        #------ Functions:
        self.Coder.register_hook("process_message_before_send", Coder_Hook)

        if self.context_variables["Code_Short_Term_Memory"]: # Sort Term Memory ON.
            self.Code_Fixer.register_hook("process_message_before_send", Code_Fixer_Hook)
        else: # Short Term Memory OFF.
            self.Code_Fixer.register_hook("process_message_before_send", Code_Fixer_Hook_No_Memory)

        self.Code_Diff.register_hook("process_message_before_send", Code_Diff_Hook)


        #----- Handoffs:
        self.Coder.handoffs.set_after_work(AgentTarget(self.Coder))

        self.Coder.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Code_Fixer),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Code_Updated} == True")
                )
            )
        )

        if self.context_variables["Code_Short_Term_Memory"]: # Short Term Memory ON

            self.Code_Fixer.handoffs.set_after_work(AgentTarget(self.Code_Fixer)) 

            # Max Review Loops Condition:
            self.Code_Fixer.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Code_Review_Count} >= ${Max_Review_Loops}")
                    )
                )   
            )
            # NOTE: AG2 order for handoffs is important as it checks these conditions in order (Top -> Bottom).

            self.Code_Fixer.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Code_Diff),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Code_Fixed} == True")
                    )
                )   
            )

            self.Code_Diff.handoffs.set_after_work(AgentTarget(self.Code_Diff))

            self.Code_Diff.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Code_Fixer),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Code_Diffs_Analysed} == True")
                    )
                )   
            )
            
        else: # No Short Term Memory
            self.Code_Fixer.handoffs.set_after_work(AgentTarget(self.Code_Fixer)) 

            # Max Review Loops Condition:
            self.Code_Fixer.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Code_Review_Count} >= ${Max_Review_Loops}")
                    )
                )   
            )
            
    def run_Conversation(self):
        #------------------------ Scoring Folder Setup -----------------------------------
        RESULTS_DIR = self.ROOT_DIR/f"Results" # We want a generc results folder that contains all results for all LLM types.
        if not RESULTS_DIR.exists():
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        LLM_RESULTS_DIR = RESULTS_DIR/f"{self.LLM_Name}" # All results associated with a specifc LLM will be stored here.
        if not LLM_RESULTS_DIR.exists():
            LLM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ITER_FOLDER_DIR = LLM_RESULTS_DIR/f"Coding_Local_Iterations"
        if not ITER_FOLDER_DIR.exists():
            ITER_FOLDER_DIR.mkdir(parents=True, exist_ok=True)
        
        # New Results (Based on code success rate and number of retries) =========================================
        if self.context_variables["Short_Term_Memory"]:
            self.Planning_State = "M" # Stands for Memory On
        else:
            self.Planning_State = "N" # Stands for No Memory
        if self.context_variables["Code_Short_Term_Memory"]:
            self.Coding_State = "M"
        else:
            self.Coding_State = "N"
        self.ANALYSIS_RESULTS_DIR = LLM_RESULTS_DIR/f"""Memory_Ablation_{self.context_variables["Max_Plan_Reviews"]}_{self.context_variables["Max_Review_Loops"]}"""/f"""{self.Planning_State}{self.Coding_State}"""
        
        if not self.ANALYSIS_RESULTS_DIR.exists():
            self.ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        self.ANALYSIS_DIR = self.ANALYSIS_RESULTS_DIR/f"""Code_Analysis_{self.Planning_State}_{self.Coding_State}.csv"""
        if not self.ANALYSIS_DIR.exists():
            # Writing Column Headings to Results CSV File (ONLY if file doesn't already exist)
            with open(self.ANALYSIS_DIR, "w") as f:
                    writer = csv.writer(f)
                    row = ["Success","Debug_Attempts", "Run_ID", "Code_Size", "Cyclomatic_Complexity", "Nesting_Depth"] # These are the main metrics that will be used for analysis.
                    writer.writerow(row)


        # NOTE: This initialises the coding results scoring folders. The global iteration number does not matter.
        # NOTE: It also does not matter if this is an entirely separate run. All that matters is the scoring of the code
        # NOTE: based on iteration number, to assess if initial coding loop encourages better nitial code before debug loop.
        #--------------------------------------------------------------------------
        # NOTE: As the chatroom iterates using only a single class instantiation and method call, the scores need to be recorded by dynamically
        # NOTE: creating context variables in BioAgent based on Max_Review_Loops and then giving each agent in review panel access to a specific list
        # NOTE: index to record their results in their functions. These functions would rely on the use of Code_Review_Count to get the correct index.
        # NOTE: File writing would be done after this agent conversation.

        if self.context_variables["Code_Short_Term_Memory"]: # Short Term Memory ON.
            agents = [self.Coder,self.Code_Fixer,self.Code_Diff]
        else: # Short Term Memory OFF.
            agents = [self.Coder,self.Code_Fixer]

        pattern=DefaultPattern(
        initial_agent=self.Coder,
        agents=agents,
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Implement the plan using Python code. The code should be updated iteratively, but you MUST attempt to write the full code in the first attempt.",
            max_rounds=self.Max_Rounds,
        )

        # ------------------ Writing Scores to Correct Files Based on Iteration Number and LLM Type ----------------------
        for i in range(self.Max_Review_Loops):
            file_path = ITER_FOLDER_DIR / f"Scores_Iter{i+1}.csv"
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                row = self.context_variables[f"Code_Score_{i}"]
                writer.writerow(row)
        #----------------------------------------------------------------------------------
        return result, ctx, self.ANALYSIS_DIR


# 2 ----- Execution Loop:
class Execution_Loop:
    def __init__(self,context_variables: ContextVariables, Max_Rounds:int):
        print("\nExecution in progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.ROOT_DIR = Path("/workspace")
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds # This is to prevent an infinite loop in case there are any erros calling the functions.
        
        # Context Variable Reset;
        # Only the context variables for when the summariser has called its reply function AND the successful code output variable should be reset here.
        self.context_variables["Error_Summarised"]=False
        self.context_variables["Success"]=False
        # Pip Should not be reset here as it is a condition that is needed later in run conversation method.

        #------- Agents:

        # Code Executor Setup -------
        # NOTE: The scratchpad will be used for the code executor during testing. This will get clustered with code and will never be cleaned. Successful code is stored in the repo and will be executed in a clean environment.
        #--- Setting the Path:
        self.inputs_dir=Path("/inputs") # Read-only bind mount to container.
        self.work_dir=self.ROOT_DIR/"ScratchPad"
        self.pip_dir=self.ROOT_DIR/"Container_Packages"

        # Creates folders if not already present.
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True, exist_ok=True)
        if not self.pip_dir.exists():
            self.pip_dir.mkdir(parents=True, exist_ok=True)

        self.executor = SecureLocalCommandLineExecutor(
            timeout = 600,
            work_dir = str(self.work_dir),
            inputs_dir = str(self.inputs_dir),
            pip_install_dir = str(self.pip_dir),
            execution_policies = {"python": True,
            "bash": False,
            "powershell": False,
            "shell": False,
            "sh": True,
            "pwsh": False,
            "ps1": False}
        )
        self.Code_Executor_Agent = ConversableAgent("Code_Executor_Agent",
        llm_config=False,  # Turn off LLM for this agent.
        code_execution_config={"executor": self.executor,
        "last_n_messages": 3},  # Use the Singularity Command Line Executor.
        human_input_mode="NEVER",
        )
        #----------------------------
        self.Summariser = Agent_Factory(agent_name="Summariser", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        #------ Functions:
        self.Summariser.register_hook("process_message_before_send", Summary_Hook)
        #----- Handoffs:
        self.Code_Executor_Agent.handoffs.set_after_work(AgentTarget(self.Summariser)) # Executor always hands off as it has no LLM and is deterministic.

        self.Summariser.handoffs.set_after_work(AgentTarget(self.Summariser)) # If context condition not satisfied it is due to error with summariser calling function. Return to summariser to try again.

        self.Summariser.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Error_Summarised} == True & (${Success} == False or ${Success} == True)")
                )
            )
        )
        # To move on, the Success value must be set to a valid boolean value. The function must have been called successfully as well.

    def run_Conversation(self):
        pattern=DefaultPattern(
        initial_agent=self.Code_Executor_Agent,
        agents=[self.Code_Executor_Agent, self.Summariser],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        if self.context_variables["Pip"]:
            messages=f"""```sh \n{self.context_variables["Pip_Code"]} \n``` \n```python \n{self.context_variables["Code"]} \n```"""
        else:
            messages= f"```python \n{self.context_variables["Code"]} \n```"
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages=messages, #This code block format is requred for the SingularityCommandLineExecutor to work properly.
            max_rounds=self.Max_Rounds,
        )
        return result, ctx

#3 ----- Controller Agent:
class Control_Strategy:
    def __init__(self,context_variables: ContextVariables, Max_Rounds:int, Diff:str, History:str):
        print("\nCode Control Strategy in progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds # This is to prevent an infinite loop in case there are any erros calling the functions.
        self.Diff = Diff
        self.History = History

        # Update Context Variables with Diff and History for LLM access:
        self.context_variables["Diff"] = Diff
        self.context_variables["History"] = History

        # Reset Context Variables for Control Strategy handoff logic:
        self.context_variables["Rollback"] = False
        self.context_variables["Controller_Finished"] = False

        #------- Agents:
        self.Controller = Agent_Factory(agent_name="Controller", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        #------ Functions:
        self.Controller.register_hook("process_message_before_send", Git_Control_Hook)

        #----- Handoffs:
        self.Controller.handoffs.set_after_work(AgentTarget(self.Controller)) # No other agent in this conversation to handoff to. 
        
        # Exit to Next Stage:
        self.Controller.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Controller_Finished} == True")
                )
            )
        )
    def run_Conversation(self):
        pattern=DefaultPattern(
        initial_agent=self.Controller,
        agents=[self.Controller],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Review the code, the git history and diffs and determine if the repo needs to be rolled back or if debug agent needs any advice to steer it in the correct direction.",
            max_rounds=self.Max_Rounds,
        )
        return result, ctx

#4 ----- Pip Manager:
class Pip_Manager:
    def __init__(self,context_variables: ContextVariables, Max_Rounds:int):
        print("\nInstalling Required Python Packages ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds # This is to prevent an infinite loop in case there are any erros calling the functions.

        # Reset Context Variables for Pip_Manager handoff logic:
        self.context_variables["Packages_Managed"] = False

        #------- Agents:
        self.Pip = Agent_Factory(agent_name="Pip", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        #------ Functions:
        self.Pip.register_hook("process_message_before_send", Installations_Hook)
        
        #----- Handoffs:
        self.Pip.handoffs.set_after_work(AgentTarget(self.Pip)) # No other agent in this conversation to handoff to. 
        
        # Exit to Next Stage:
        self.Pip.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Packages_Managed} == True")
                )
            )
        )
    def run_Conversation(self):
        pattern=DefaultPattern(
        initial_agent=self.Pip,
        agents=[self.Pip],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="You must look at the error message and write shell code for managing pip installations.",
            max_rounds=self.Max_Rounds,
        )
        return result, ctx

#5 ----- Debug Agent:
class Debug_Loop:
    def __init__(self,context_variables: ContextVariables, Max_Rounds:int, Diff:str, History:str):
        print("\nDebugging in progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds # This is to prevent an infinite loop in case there are any erros calling the functions.
        self.Diff = Diff
        self.History = History

        # Update Context Variables with Diff and History for LLM access:
        self.context_variables["Diff"] = Diff
        self.context_variables["History"] = History

        # Reset Context Variables for Debug_Loop handoff logic:
        self.context_variables["Debug_Finished"] = False

        #------- Agents:
        self.Debugging = Agent_Factory(agent_name="Debugging", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        #------ Functions:
        self.Debugging.register_hook("process_message_before_send", Debug_Hook)
    
        #----- Handoffs:
        self.Debugging.handoffs.set_after_work(AgentTarget(self.Debugging)) # No other agent in this conversation to handoff to. 
        
        # Exit to Next Stage:
        self.Debugging.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Debug_Finished} == True")
                )
            )
        )
    def run_Conversation(self):
        pattern=DefaultPattern(
        initial_agent=self.Debugging,
        agents=[self.Debugging],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="You must review the code and the git history to update the code with the fix. Use the advice from the controller to help with this. Once fixed, detail your approach in a concise sentence for the next git commit.",
            max_rounds=self.Max_Rounds,
        )
        return result, ctx

#5 ----- :
class Failure_Analyst_Loop:
    def __init__(self,context_variables: ContextVariables, Max_Rounds:int, Diff:str, History:str):
        print("\nFailure Analysis in progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds # This is to prevent an infinite loop in case there are any erros calling the functions.

        # Reset Context Variables for Failure_Analyst_Loop handoff logic:
        self.context_variables["Failure_Analysed"] = False

        #------- Agents:
        self.Analyst = Agent_Factory(agent_name="Analyst", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        #------ Functions:
        self.Analyst.register_hook("process_message_before_send", Analysis_Hook)
        
        #----- Handoffs:
        self.Analyst.handoffs.set_after_work(AgentTarget(self.Analyst)) # No other agent in this conversation to handoff to. 
        
        # Exit to Next Stage:
        self.Analyst.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Failure_Analysed} == True")
                )
            )
        )
    def run_Conversation(self):
        pattern=DefaultPattern(
        initial_agent=self.Analyst,
        agents=[self.Analyst],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="You must review the code and the git history and instruct the debugging agent on how to fix the code.",
            max_rounds=self.Max_Rounds,
        )
        return result, ctx

class Implementation_Loop:
    def __init__(self,context_variables: ContextVariables, Max_Retries:int, Max_Review_Loops: int, Git_Memory: GitMemory, Review_Agent_Number: int):
        print("\nImplementation Loop in progress ... \n")
        # LLM not needed for ths class as other classes already ensure they have the correct LLM setup.
        self.context_variables = context_variables
        self.Max_Retries = Max_Retries # This is to prevent an infinite loop in case there are any erros calling the functions.
        self.context_variables["Max_Retries"] = Max_Retries
        self.Max_Review_Loops = Max_Review_Loops # This is to prevent reviewer from causing excessive time wastage.
        self.Review_Agent_Number = Review_Agent_Number
        self.Git_Memory= Git_Memory # This is to enable git memory handling across the implementation loop. 
        self.Git_Manager= None # Method will create GitManager and update instance variable.
        self.LLM_Name = os.getenv("LLM_Model_Reasoning") # This extracts the LLM name from teh .env file to enable scoring assocated with that LLM type.

        # Final Code Folder (Only for code that runs successfully)
        self.CODE_DIR = Path("/workspace/Code")
        if self.CODE_DIR.exists():
            shutil.rmtree(self.CODE_DIR) # If the fle exists, delete the contents and start again by making a blank directory.
            self.CODE_DIR.mkdir(parents=True, exist_ok=True)
        else:
            self.CODE_DIR.mkdir(parents=True, exist_ok=True)
    def solve(self):
        for step in range(1,self.context_variables["Plan"]["Number_of_Steps"]+1):
            # Intial code and review loop. Limited to ensure time is not wasted.
            Initial = Initial_Loop(context_variables=self.context_variables, Max_Rounds=100, Plan_Step=step, Max_Review_Loops=self.Max_Review_Loops, LLM_Name=self.LLM_Name, Review_Agent_Number=self.Review_Agent_Number, Git_Memory=self.Git_Memory)
            _,_,folder = Initial.run_Conversation()
            self.ANALYSIS_DIR = folder
            Retries=0 # Initialise for each step.
            
            self.Git_Manager = GitManager(plan_step=step, Git_Memory=self.Git_Memory, context_variables=self.context_variables, Hard_Reset=False)
            
            # For first execution ONLY.
            EL=Execution_Loop(context_variables=self.context_variables, Max_Rounds=10)
            EL.run_Conversation() # Execute to get the execution results and summary.

            # 1 Chance for Installation Before First Commit:
            if self.context_variables["Pip"]: # If first error is due to Pip error, then sort his out before first commit and re-run code.
                Pip_Manage=Pip_Manager(context_variables=self.context_variables, Max_Rounds=10)
                Pip_Manage.run_Conversation() # This will manage the pip installations based on the error message.

                EL=Execution_Loop(context_variables=self.context_variables, Max_Rounds=10)
                EL.run_Conversation() # Execute to get the execution results and summary.
            else:
                pass
            # If successful on first attempt:
            if self.context_variables["Success"]:
                self.Git_Manager.write_code(self.context_variables["Code"]) # This ensures that the latest code is always in the repo for the Control Strategy to access and make commits with.
                self.Git_Manager.record_result("Success") # Records the successful attempt before moving to the next plan step.
                self.Git_Manager.commit(approach= "First Code Commit", outcome="Success", error_message= "None") # If the code works on the first attempt, this ensures it is committed.
                    
                # Write the Successful Code to the Code File for re-running:
                FILE_DIR = Path(self.CODE_DIR/f"Code_{step}.py")
                Code = self.Git_Manager.read_code()
                with open(FILE_DIR, "w", newline="") as c:
                    c.write(Code)
                
                # Calculating Coding Score Metrics ============================

                # Statement Node Count (Code Size - How many logically useful nodes):
                C_Size = code_size(Code)

                # Cyclomatic Complexity (Logical Complexity - Number of decision points):
                Cyc_Complexity = cyclomatic_complexity(Code)

                # Nesting Depth (Complexity)
                ND = nesting_depth(Code)

                # Run ID
                R = self.context_variables["Run_ID"]
                
                # Record Results =====================================
                with open(self.ANALYSIS_DIR, "a", newline="") as f:
                    writer = csv.writer(f)
                    row = [1,Retries, R, C_Size, Cyc_Complexity, ND]  # Success = 1. Retries only recorded if code successful.
                    writer.writerow(row)
                        
                continue # Move to next plan step if successful.

            # Before Sendng to Control_Strategy, we need to write the code to the repo, then
            # make a commit and obtain latest diff. 
            # NOTE: We need ths in a large loop that only exits when implementation is successful or Max_Retries is reached.
            while self.context_variables["Success"] == False:
                self.Git_Manager.write_code(self.context_variables["Code"]) # This ensures that the latest code is always in the repo for the Control Strategy to access and make commits with.
                self.Git_Manager.commit(approach=self.context_variables["Approach"], outcome=self.context_variables["Outcome"], error_message=self.context_variables["Summary"])
                diff = self.Git_Manager.retrieve_diff()
                history = self.Git_Manager.retrieve_history()
                # Now, we can pass the diff and the history to the Control Strategy and the Debug Loop.

                # Max Retries Escape:
                if self.context_variables["Max_Retries"] < Retries:
                    self.Git_Manager.record_result("Fail") # Records the failed attempt before moving to the next plan step.
                    # Recording Results ==================================
                    # NOTE: In this case, this would be considered a failure.
                    R = self.context_variables["Run_ID"]
                    with open(self.ANALYSIS_DIR, "a", newline="") as f:
                        writer = csv.writer(f)
                        row = [0, None, R, None, None, None] # Failure = 0. Retries not recorded when failure occurs.
                        writer.writerow(row)
                    break

                CS = Control_Strategy(context_variables=self.context_variables, Max_Rounds=10, Diff=diff, History=history)
                CS.run_Conversation() 
                # Rollback depends on CS output.
                if self.context_variables["Rollback"]:
                    self.Git_Manager.rollback() # This will rollback to the last commit, which is the state of the code before the failed implementation attempt. This ensures a clean state for the next attempt.
                    self.context_variables["Rollback_Count"] +=1
                    diff = self.Git_Manager.retrieve_diff() # Get the diff after rollback to update the context for the next Control Strategy run.
                    history = self.Git_Manager.retrieve_history() # Get the history after rollback to update the context for the next Control Strategy run.
                    # ----- To Synchronise Context Variables, we need to read from Repo again ----
                    code = self.Git_Manager.read_code()
                    self.context_variables["Code"] = code # Get the diff after the rollback.
                else:
                    pass
                if self.context_variables["Pip"]:
                    Pip = Pip_Manager(context_variables=self.context_variables, Max_Rounds=10)
                    Pip.run_Conversation() # This will manage the pip installations if there is a pip error. This will just add the Pip manager to the workflow, but won't break the main flow if it is wrong to call Pip Agent.
                
                EA = Failure_Analyst_Loop(context_variables=self.context_variables, Max_Rounds=10, Diff=diff, History=history)
                EA.run_Conversation() # This will provide the debugging agent with instructions on how to fix.

                DL = Debug_Loop(context_variables=self.context_variables, Max_Rounds=10, Diff=diff, History=history)
                DL.run_Conversation()
                Retries +=1
                print(f"\n\nRetry Number: {Retries} \n")

                EL=Execution_Loop(context_variables=self.context_variables, Max_Rounds=10)
                EL.run_Conversation() # Execute to get the execution results and summary.
                # Code will now go around in this loop for this plan step until it is successful or Max Retries is reached.

                if self.context_variables["Success"]:
                    self.Git_Manager.write_code(self.context_variables["Code"]) # This ensures that the latest code is always in the repo for the Control Strategy to access and make commits with.
                    self.Git_Manager.record_result("Success") # If successful, record the successful attempt in the repo.
                    self.Git_Manager.commit(approach=self.context_variables["Approach"], outcome=self.context_variables["Outcome"], error_message=self.context_variables["Summary"])
                    
                    # Write the Successful Code to the Code File for re-running:
                    FILE_DIR = Path(self.CODE_DIR/f"Code_{step}.py")
                    Code = self.Git_Manager.read_code()
                    with open(FILE_DIR, "w", newline="") as c:
                        c.write(Code)
                    
                    # Calculating Coding Score Metrics ============================

                    # Statement Node Count (Code Size - How many logically useful nodes):
                    C_Size = code_size(Code)

                    # Cyclomatic Complexity (Logical Complexity - Number of decision points):
                    Cyc_Complexity = cyclomatic_complexity(Code)

                    # Nesting Depth (Complexity)
                    ND = nesting_depth(Code)

                    # Run ID
                    R = self.context_variables["Run_ID"]
                    
                    # Record Results =====================================
                    with open(self.ANALYSIS_DIR, "a", newline="") as f:
                        writer = csv.writer(f)
                        row = [1,Retries, R, C_Size, Cyc_Complexity, ND]  # Success = 1. Retries only recorded if code successful.
                        writer.writerow(row)
                else:
                    pass

def main():
    context_variables = ContextVariables({
        "Max_Review_Loops": 0,
        "Plan_Section": {},
        "Code_Review_Count": 0,
        "Code": "",
        "Code_Errors": "",
        "Plan_Enforcement":"",
        "Optimisation_Goals":"",
        "Error_Summarised": False,
        "Success": False,
        "Summary":"",
        "Approach":"Initial code commit.",
        "Outcome":"",
        "Rollback_Count": 0, # Gives Total Rollbacks in code run. This is to assess the usage of Git and determine how agents are learning.
        "Max_Retries": 10,
        "History": "",
        "Diff": "",
        "Suggestions": "",
        "Controller_Finished": False,
        "Rollback": False,
        "Debug_Finished": False,
        # ---- Pip Manager Context Variables:
        "Packages_Managed": False,
        "Pip_Code": "",
        "Pip": False,
        #------- Error Analyst Context Variables:
        "Failure_Analysed": False,
        "Instructions": "",
        #-------- Git Based Loops:
        "Code_Updated": False,
        "Code_Diffs_Analysed": False,
        "Code_Fixed": False,
        "Code_Diffs": "",
        "Code_Approach": "",

    })
    Init=Initial_Loop(context_variables=context_variables, Max_Rounds=15, Plan_Step=1, Max_Review_Loops=3)
if __name__ == "__main__":
    main()

