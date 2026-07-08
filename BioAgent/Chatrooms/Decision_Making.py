# ---------------------Imports ------------------------------------------------------------------------
from autogen.agentchat.group import AgentTarget, RevertToUserTarget, OnCondition, StringLLMCondition
from autogen.agentchat.group import OnContextCondition, ExpressionContextCondition, ContextExpression
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import ContextVariables, ReplyResult
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.targets.transition_target import TerminateTarget
from typing import Any, Dict, List, Optional, Annotated, Tuple, Union
import os
from pathlib import Path
from Config.vLLM_Configuration import VLLM_Config
from Config.vLLM_Manager import LLM_Manager
from dotenv import load_dotenv
from Agents.AgentFactory import Agent_Factory
load_dotenv()
# ----- Functions for Agents
from Tools.Decision_Phase import Run_Screening, Screening_Hook
# ----------------------------------------------------------------------------------------


class Decision_Making:
    """Host-guest screening decision phase.

    Mirrors ``Chatrooms/Data_Discovery.py``: the deterministic verdicts are
    computed by validated tools FIRST (``Run_Screening``), then a single
    interpreter agent narrates the outcome. The LLM never changes a verdict.
    """

    def __init__(
        self,
        context_variables: ContextVariables,
        Max_Rounds: int,
        Workflow_Name: str = "SUPRAMOL-SCREENING",
        Use_Processed: bool = True,
    ):
        print("Decision Making (Host-Guest Screening) in Progress ... \n")
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds
        self.Workflow_Name = Workflow_Name
        self.Use_Processed = Use_Processed
        self.inputs_dir = Path("/inputs")

        # ------- Agents:
        self.Screening_Interpreter = Agent_Factory(
            agent_name="Screening_Interpreter",
            context_variables=self.context_variables,
        ).BuildAgent()
        self.Chat_Config = VLLM_Config(
            api_type="openai",
            cache_seed=None,
            temperature=0.3,
            enable_thinking=False,
            LLM_Type="Reasoning",
        ).build_config()

        # ------ Function Registration:
        self.Screening_Interpreter.register_hook(
            "process_message_before_send", Screening_Hook
        )
        # ------ Handoffs:
        self.Screening_Interpreter.handoffs.set_after_work(
            AgentTarget(self.Screening_Interpreter)
        )
        self.Screening_Interpreter.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Screening_Interpreted} == True")
                ),
            )
        )

    def run_Conversation(self):
        # ----- First run the deterministic screening tools to populate verdicts.
        # (No LLM involved in the verdict — same pattern as Data_Discovery.py:67-68.)
        Run_Screening(
            context_variables=self.context_variables,
            Input_Dir=self.inputs_dir,
            name=self.Workflow_Name,
            use_processed=self.Use_Processed,
        )
        # ---------------------

        pattern = DefaultPattern(
            initial_agent=self.Screening_Interpreter,
            agents=[self.Screening_Interpreter],
            group_manager_args={"llm_config": self.Chat_Config},
            context_variables=self.context_variables,
        )

        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages=(
                "Interpret the deterministic host-guest screening results and "
                "report which samples formed the target assembly. Do not change "
                "any verdict."
            ),
            max_rounds=self.Max_Rounds,
        )
        return result, ctx


def main():
    context_variables = ContextVariables(
        {
            "Screening_Verdicts": {},
            "Screening_Summary": [],
            "Screening_Available": False,
            "Screening_Interpretation": "",
            "Screening_Interpreted": False,
            "Focus_Area_Statement": "",
        }
    )
    dm = Decision_Making(context_variables=context_variables, Max_Rounds=10)
    dm.run_Conversation()


if __name__ == "__main__":
    main()
