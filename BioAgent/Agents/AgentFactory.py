import yaml
from pathlib import Path
from typing import Optional, Any
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.group import ContextVariables
from Config.vLLM_Configuration import VLLM_Config
from Agents.AgentBase import Agent_Base
from Utils.Pydantic_Schema import PlanResponse, ReviewResponse, LearningResponse, DiffResponse, CodeReviewResponse, RAGQuestions, Compile_Response, SummaryResponse, GitControlResponse, DebugResponse, VLResponse, MarkdownResponse, ReportResponse, ScreeningVerdict, ScreeningInterpretationResponse, NMRAnalysisResponse, MSAnalysisResponse
import base64
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
class Agent_Factory():
    # This class is designed to take the AgentBase and the configuration files from each agent folder
    # to be able to create the correct agent in each chatroom more efficiently. Changes to agent configurations
    # can be made to their config files and the everything downstream will be updated efficiently.
    def __init__(self, agent_name: str, context_variables: ContextVariables):
        self.agent_name=agent_name
        self.context_variables=context_variables
        self.folder=None
        # Loading the YAML file with agent names and mappings.
        self.AGENTS_PATH = Path(__file__).parent
        with open(self.AGENTS_PATH/"agent_mapping.yml") as f:
            agent_mapping = yaml.safe_load(f)
        for key, value in agent_mapping.items():
            if self.agent_name==key:
                self.folder=value
                break
        else: 
            raise ValueError(f"Agent name {self.agent_name} not found in agent mapping. \n The agent name must be from this list: {list(agent_mapping.keys())}")
        # Set the main path to the correct agent folder.
        self.config_path= self.AGENTS_PATH/self.folder/"config.yml"
        self.system_message_path= self.AGENTS_PATH/self.folder/"system_message.yml"

    def BuildAgent(self) -> ConversableAgent:
        # Loading the system message and config files for the correct agent:
        with open(self.config_path) as c:
            config_data = yaml.safe_load(c)
        with open(self.system_message_path) as s:
            system_messages = yaml.safe_load(s)
        # Building the LLM Config for the agent:
        agent_name = config_data["name"]
        api_type = config_data["api_type"]
        temperature = config_data["temperature"]
        if config_data.get("response_format", None) == "PlanResponse": # Note that the plan response is the name of the schema. All schema available are imported at the top of the file.
            response_format = PlanResponse
        elif config_data.get("response_format", None) == "ReviewResponse":
            response_format = ReviewResponse
        elif config_data.get("response_format", None) == "LearningResponse":
            response_format = LearningResponse
        elif config_data.get("response_format", None) == "DiffResponse":
            response_format = DiffResponse
        elif config_data.get("response_format", None) == "CodeReviewResponse":
            response_format = CodeReviewResponse
        elif config_data.get("response_format", None) == "RAGQuestions":
            response_format = RAGQuestions
        elif config_data.get("response_format", None) == "Compile_Response":
            response_format = Compile_Response
        elif config_data.get("response_format", None) == "SummaryResponse":
            response_format = SummaryResponse
        elif config_data.get("response_format", None) == "GitControlResponse":
            response_format = GitControlResponse
        elif config_data.get("response_format", None) == "DebugResponse":
            response_format = DebugResponse
        elif config_data.get("response_format", None) == "VLResponse":
            response_format = VLResponse
        elif config_data.get("response_format", None) == "MarkdownResponse":
            response_format = MarkdownResponse
        elif config_data.get("response_format", None) == "ReportResponse":
            response_format = ReportResponse
        elif config_data.get("response_format", None) == "ScreeningVerdict":
            response_format = ScreeningVerdict
        elif config_data.get("response_format", None) == "ScreeningInterpretationResponse":
            response_format = ScreeningInterpretationResponse
        elif config_data.get("response_format", None) == "NMRAnalysisResponse":
            response_format = NMRAnalysisResponse
        elif config_data.get("response_format", None) == "MSAnalysisResponse":
            response_format = MSAnalysisResponse
        else:
            response_format = None
        llm_type = config_data["llm_type"]
        enable_thinking = config_data["enable_thinking"]
        Multimodal = config_data["Multimodal"]

        llm_config=VLLM_Config(api_type=api_type,
                             temperature=temperature,
                             enable_thinking=enable_thinking,
                             response_format=response_format,
                             LLM_Type=llm_type).build_config()
        # Building the agent from system message and llm config:
        system_message = system_messages["system_message"]
        updated_system_message = system_messages.get("updated_system_message",None)

        if Multimodal:
            Agent_Model = Agent_Base(name=agent_name,
            llm_config=llm_config,
            system_message=system_message,
            Update_System_Message=updated_system_message,
            context_variables=self.context_variables,
            Multimodal = True)
        else:
            Agent_Model = Agent_Base(name=agent_name,
            llm_config=llm_config,
            system_message=system_message,
            Update_System_Message=updated_system_message,
            context_variables=self.context_variables)
            
        return Agent_Model.agent