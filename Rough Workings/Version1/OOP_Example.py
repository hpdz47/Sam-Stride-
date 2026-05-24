"""Minimal two-agent MAS scaffold using AG2 with a vLLM backend.

This script demonstrates an object-oriented structure for wiring up a
planner and reviewer agent that share context and run inside a simple
group chat workflow. The code keeps AG2- and vLLM-specific plumbing in
dedicated classes so the orchestration remains clear and easy to extend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.patterns import AutoPattern


class VLLMConfigFactory:
	"""Builds an AG2 LLMConfig that points to a running vLLM endpoint."""

	def __init__(
		self,
		*,
		model_env: str = "LLM_Model",
		api_key_env: str = "API_KEY",
		base_url: str = "http://127.0.0.1:8000/v1",
		temperature: float = 0.2,
		enable_thinking: bool = False,
	) -> None:
		self._model_env = model_env
		self._api_key_env = api_key_env
		self._base_url = base_url
		self._temperature = temperature
		self._enable_thinking = enable_thinking

	def build(self) -> LLMConfig:
		model_name = os.environ.get(self._model_env, "qwen2.5-72b-instruct")
		api_key = os.environ.get(self._api_key_env, "EMPTY")
		config: Dict[str, Any] = {
			"api_type": "openai",
			"model": model_name,
			"api_key": api_key,
			"base_url": self._base_url,
			"temperature": self._temperature,
			"cache_seed": None,
			"extra_body": {
				"chat_template_kwargs": {"enable_thinking": self._enable_thinking}
			},
		}
		return LLMConfig(config_list=[config])


@dataclass(frozen=True)
class AgentBlueprint:
	"""Static configuration for an agent role."""

	name: str
	system_message: str


class BaseLLMAgent:
	"""Wraps a ConversableAgent with shared context."""

	def __init__(
		self,
		blueprint: AgentBlueprint,
		llm_config: LLMConfig,
		context: ContextVariables,
	) -> None:
		self._blueprint = blueprint
		self._context = context
		self._llm_config = llm_config
		self._agent = ConversableAgent(
			name=blueprint.name,
			system_message=blueprint.system_message,
			llm_config=llm_config,
			human_input_mode="NEVER",
			context_variables=context,
		)

	@property
	def agent(self) -> ConversableAgent:
		return self._agent

	@property
	def name(self) -> str:
		return self._blueprint.name


class PlannerAgent(BaseLLMAgent):
	"""Planner role with domain-specific responsibilities."""


class ReviewerAgent(BaseLLMAgent):
	"""Reviewer role that critiques planner output."""


class PlanningSession:
	"""Coordinates the planner and reviewer agents inside a group chat."""

	def __init__(
		self,
		*,
		planner_prompt: Optional[str] = None,
		reviewer_prompt: Optional[str] = None,
		context_seed: Optional[Dict[str, Any]] = None,
	) -> None:
		factory = VLLMConfigFactory()
		self._llm_config = factory.build()
		seed_data = context_seed or {"plan_drafts": []}
		self._context = ContextVariables(data=seed_data)

		planner_blueprint = AgentBlueprint(
			name="PlannerAgent",
			system_message=planner_prompt
			or """You create structured, step-by-step execution plans.""",
		)
		reviewer_blueprint = AgentBlueprint(
			name="ReviewerAgent",
			system_message=reviewer_prompt
			or """You review plans for risks, gaps, and compliance issues.""",
		)

		self.planner = PlannerAgent(planner_blueprint, self._llm_config, self._context)
		self.reviewer = ReviewerAgent(
			reviewer_blueprint,
			self._llm_config,
			self._context,
		)

	@property
	def context(self) -> ContextVariables:
		return self._context

	def run(self, task_instruction: str, *, max_rounds: int = 6) -> str:
		"""Starts a group chat between planner and reviewer and returns the final reply."""

		group_chat = GroupChat(agents=[self.planner.agent, self.reviewer.agent])
		manager = GroupChatManager(
			groupchat=group_chat,
			llm_config=self._llm_config,
			context_variables=self._context,
		)
		pattern = AutoPattern(
			initial_agent=self.planner.agent,
			agents=[self.planner.agent, self.reviewer.agent],
			group_manager_args={"llm_config": self._llm_config},
			context_variables=self._context,
		)
		result, _context, _trace = initiate_group_chat(
			pattern=pattern,
			messages=task_instruction,
			max_rounds=max_rounds,
		)
		final_response = result.final_response if hasattr(result, "final_response") else str(result)
		self._context = _context or self._context
		manager.stop()
		return final_response


def main() -> None:
	session = PlanningSession(
		planner_prompt=(
			"You are the planner for a multi-agent system. Generate a numbered plan "
			"with actionable steps that another agent can execute."
		),
		reviewer_prompt=(
			"You are the reviewer. Evaluate the planner's proposal, highlight issues, and "
			"suggest revisions if needed."
		),
		context_seed={"plan_drafts": [], "review_notes": []},
	)
	response = session.run(
		"Draft a plan for cleaning and analyzing a mass spectrometry dataset.",
		max_rounds=4,
	)
	print("=== Final Response ===")
	print(response)


if __name__ == "__main__":
	main()

