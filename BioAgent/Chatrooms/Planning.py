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
from Utils.Git_Memory_Handling import GitMemory
from Utils.Plan_Repo_Management import create_plan_git_managers
#----- Functions for Agents
from Tools.Planning_Phase import Planner_Hook, Plan_Hook, Diff_Hook, Plan_Hook_No_Memory
#----------------------------


class Planning_Chatroom:
    def __init__(self,context_variables: ContextVariables, Max_Plan_Steps: int, Max_Rounds:int, Max_Plan_Reviews:int, Review_Agent_Number:int, Git_Memory: GitMemory):
        print("Planning Phase in Progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Plan_Steps = Max_Plan_Steps
        self.Max_Rounds = Max_Rounds
        self.context_variables["Max_Plan_Steps"] = Max_Plan_Steps
        self.Max_Plan_Reviews = Max_Plan_Reviews # This is the main plan iteration control. Need to set the Max_Rounds to be high to allow any user input. Max_Rounds acts as a safety limit.
        self.context_variables["Max_Plan_Reviews"] = Max_Plan_Reviews
        self.Review_Agent_Number = Review_Agent_Number # Used to reset the context variables used for scoring.
        self.ROOT_DIR = Path("/workspace")
        # LLM Name Extarction to storing results associated with LLM type.
        self.LLM_Name = os.getenv("LLM_Model_Reasoning") # This extracts the LLM name from teh .env file to enable scoring assocated with that LLM type.
        self.LLM_Name = self.LLM_Name.replace("/","_") # Slashes are reserved for file paths only (But often in LLM model name)

        #---- Reset Scoring Context Variables:
        for idx in range(self.Max_Plan_Reviews):
            self.context_variables[f"Plan_Score_{idx}"] = [0]*self.Review_Agent_Number

        # GitManager Setup function for review panel. (Done here so that Repo folder reset can be done at the Global Iteration Level).
        create_plan_git_managers(context_variables=self.context_variables, Root_Dir=self.ROOT_DIR, Git_Memory=Git_Memory)
        #          - Git Managers have been added to context variables so the review panel can access them.
        #------- Agents:
        self.Planner = Agent_Factory(agent_name="Planner", context_variables=self.context_variables).BuildAgent()
        self.Plan_Fixer = Agent_Factory(agent_name="Plan_Fixer", context_variables=self.context_variables).BuildAgent()
        self.Plan_Diff = Agent_Factory(agent_name="Plan_Diff", context_variables=self.context_variables).BuildAgent()

        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        if self.context_variables["Short_Term_Memory"] == True:
            #------ Functions:
            self.Planner.register_hook("process_message_before_send", Planner_Hook)

            self.Plan_Fixer.register_hook("process_message_before_send",Plan_Hook)

            self.Plan_Diff.register_hook("process_message_before_send", Diff_Hook) # This is added to ensure that when the Diff Agent updates the plan based on the diff, the same context variables are updated to trigger the relevant handoffs and update the plan in the review panel.
            
            #----- Handoffs:
            self.Planner.handoffs.add_context_condition(
                OnContextCondition(
                    target=AgentTarget(self.Plan_Fixer),
                    condition=ExpressionContextCondition(
                        expression=ContextExpression("${Plan_Updated} == True")
                    )
                )
            )
            self.Planner.handoffs.set_after_work(AgentTarget(self.Planner))

            # Plan fixer now implemented to separate the role of main architect and implementing reviews from other agents.
            self.Plan_Fixer.handoffs.add_context_condition( # Other agents are embedded in the function to enable parallelism in future.
                OnContextCondition(
                    target=TerminateTarget(),
                    condition=ExpressionContextCondition(
                        expression=ContextExpression("${Plan_Review_Count} >= ${Max_Plan_Reviews}")
                    )
                )
            )

            self.Plan_Fixer.handoffs.add_context_condition( # Other agents are embedded in the function to enable parallelism in future.
                OnContextCondition(
                    target=AgentTarget(self.Plan_Diff),
                    condition=ExpressionContextCondition(
                        expression=ContextExpression("${Plan_Fixed} == True")
                    )
                )
            )

            self.Plan_Fixer.handoffs.set_after_work(AgentTarget(self.Plan_Fixer))

            self.Plan_Diff.handoffs.add_context_condition( # Other agents are embedded in the function to enable parallelism in future.
                OnContextCondition(
                    target=AgentTarget(self.Plan_Fixer),
                    condition=ExpressionContextCondition(
                        expression=ContextExpression("${Plan_Diff_Reviewed} == True")
                    )
                )
            )

            self.Plan_Diff.handoffs.set_after_work(AgentTarget(self.Plan_Diff))
        else:
            #------ Functions:
            self.Planner.register_hook("process_message_before_send", Planner_Hook)

            self.Plan_Fixer.register_hook("process_message_before_send",Plan_Hook_No_Memory)
            
            #----- Handoffs:
            self.Planner.handoffs.add_context_condition(
                OnContextCondition(
                    target=AgentTarget(self.Plan_Fixer),
                    condition=ExpressionContextCondition(
                        expression=ContextExpression("${Plan_Updated} == True")
                    )
                )
            )
            self.Planner.handoffs.set_after_work(AgentTarget(self.Planner))

            # Plan fixer now implemented to separate the role of main architect and implementing reviews from other agents.
            self.Plan_Fixer.handoffs.add_context_condition( # Other agents are embedded in the function to enable parallelism in future.
                OnContextCondition(
                    target=TerminateTarget(),
                    condition=ExpressionContextCondition(
                        expression=ContextExpression("${Plan_Review_Count} >= ${Max_Plan_Reviews}")
                    )
                )
            )

            self.Plan_Fixer.handoffs.set_after_work(AgentTarget(self.Plan_Fixer)) # Always skips diff agent when short term memory is disabled.


    def run_Conversation(self):
        #------------- Scoring Folders Setup --------------
        RESULTS_DIR = self.ROOT_DIR/f"Results" # We want a generc results folder that contains all results for all LLM types.
        if not RESULTS_DIR.exists():
            RESULTS_DIR.mkdir()
        LLM_RESULTS_DIR = RESULTS_DIR/f"{self.LLM_Name}" # All results associated with a specifc LLM will be stored here.
        if not LLM_RESULTS_DIR.exists():
            LLM_RESULTS_DIR.mkdir()
        ITER_FOLDER_DIR = LLM_RESULTS_DIR/f"Planning_Local_Iterations"
        if not ITER_FOLDER_DIR.exists():
            ITER_FOLDER_DIR.mkdir()
        #--------------------------------------------------
        if self.context_variables["Short_Term_Memory"] == True:
            agents=[self.Planner, self.Plan_Fixer, self.Plan_Diff]
        else:
            agents=[self.Planner, self.Plan_Fixer] # Plan Diff is not used when short term memory is disabled as it also provides memory of plan changes.
        pattern=DefaultPattern(
        initial_agent=self.Planner,
        agents=agents,
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Create a detailed plan to address the user requirements and focus areas. Update the plan iteratively based on the context and any new information.",
            max_rounds=self.Max_Rounds,
        )
        # ------------------ Writing Scores to Correct Files Based on Iteration Number and LLM Type ----------------------
        for i in range(self.Max_Plan_Reviews):
            file_path = ITER_FOLDER_DIR / f"Scores_Iter{i+1}.csv"
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                row = self.context_variables[f"Plan_Score_{i}"]
                writer.writerow(row)
        #----------------------------------------------------------------------------------

        # NOTE: Uncomment to check context variable. AG2 will overwrite the context variable using the hok when it says [Handing Off to ....].
        # NOTE: Ths check is to make sure that the hook guardrail is working.
        print(f"""\n \n Testing: \n \n {self.context_variables["Plan"]} \n\n {self.context_variables["Plan_Diffs"]} """)
        return result, ctx

def main():
    context_variables = ContextVariables({
        "Plan": {},
        "Plan_Updated": False,
        "Plan_Fixed": False,
        "Plan_Feedback": "",
        "Feedback_Available": False,
        "Max_Plan_Steps": 0,
        # Max_Plan_Reviews tracking:
        "Plan_Review_Count": 0,
        "Max_Plan_Reviews": 0,
        # Plan Diff Tracking:
        "Plan_Approach": {},
        "Plan_Diffs": "",
        "Plan_Diff_Reviewed": False,
        "All_Plans":[],

    })

    Plan=Planning_Chatroom(context_variables=context_variables, Max_Plan_Steps=3, Max_Rounds=30)
    Plan.run_Conversation()

if __name__ == "__main__":
    main()
