# ---------------------Imports ------------------------------------------------------------------------
from autogen.agentchat.group import AgentTarget, RevertToUserTarget, OnCondition, StringLLMCondition
from autogen.agentchat.group import OnContextCondition, ExpressionContextCondition, ContextExpression
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function
from autogen.agentchat.group import ContextVariables, ReplyResult
from typing import Any, Dict, List, Optional, Annotated, Tuple, Union
from autogen import UpdateSystemMessage
import json
import csv
import os
from dotenv import load_dotenv
import base64
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
#----------------------------------------------------------------------------------------
class Agent_Base():
    # This base class is designed to help create agents efficiently by providing a common strucuture
    # that can be inherted in all agents.
    def __init__(self,name: str,llm_config: LLMConfig, system_message: str, Update_System_Message: Optional[str] = None, context_variables: Optional[ContextVariables] = None, Multimodal: Optional[bool] = False):
        self.name=name
        self.llm_config=llm_config # Composition Not Used here in case strucutred responses are needed for specific agents.
        self.system_message=system_message
        self.human_input_mode="NEVER" # Hard Coded as this is an AUTONMOUS system.
        if Update_System_Message:
            Updated_Message = [UpdateSystemMessage(Update_System_Message)]
        
        if Multimodal == False:
            self._agent=ConversableAgent(
                name=self.name,
                llm_config=self.llm_config,
                system_message=self.system_message,
                human_input_mode=self.human_input_mode,
                update_agent_state_before_reply=Updated_Message if Update_System_Message else None,
                context_variables=context_variables

            )

        else:
            self._agent=MultimodalConversableAgent(
                name=self.name,
                llm_config=self.llm_config,
                system_message=self.system_message,
                human_input_mode=self.human_input_mode,
                update_agent_state_before_reply=Updated_Message if Update_System_Message else None,
                context_variables=context_variables
            )


    @property
    def agent(self):
        return self._agent # Getter for retrieving the agent instance.