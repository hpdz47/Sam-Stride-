from typing import Any, Dict, List, Optional, Annotated
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function
import os
from dotenv import load_dotenv
load_dotenv("/app/.env")

class VLLM_Config():
    def __init__(self,*,
                 api_type: str, cache_seed: Optional[str]=None,
                  temperature: float, enable_thinking: bool,
                  response_format: Optional[Any]=None,
                  LLM_Type: str) -> None:
        self.api_type = api_type
        if LLM_Type=="Reasoning":
            self.model = os.environ["LLM_Model_Reasoning"]
            self.base_url = "http://127.0.0.1:8002/v1"
        elif LLM_Type=="Coding":
            self.model = os.environ["LLM_Model_Coding"]
            self.base_url = "http://127.0.0.1:8002/v1"
        elif LLM_Type=="VL":
            self.model = os.environ["LLM_Model_Visual"]
            self.base_url = "http://127.0.0.1:8002/v1"
        elif LLM_Type=="Judge":
                self.model = os.environ["LLM_Model_Judge"]
                self.base_url = "http://127.0.0.1:8002/v1"
        else:
            raise ValueError(f"Invalid LLM Type: {LLM_Type}")
        self.api_key = os.environ["API_KEY"]
        self.cache_seed = cache_seed
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.response_format = response_format
    def build_config(self) -> LLMConfig:
        config={
            "api_type": self.api_type,
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "cache_seed": self.cache_seed,
            "temperature": self.temperature,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
            "response_format": self.response_format if self.response_format else None,
            "max_retries": 5, # Hard Coded and not adjustable for if the LLM API is unresponsive (whether localhost or cloud API).
            "timeout": 300, # Hard coded to stop agents from trying to produce long verbose outputs that take too long to generate causing latency issues.
            #"max_tokens": 6000,
            #"frequency_penalty": 0.3,
        }
        return LLMConfig(config_list=[config]) # Instance of LLMConfig() created and returned to instantiate a config object.
