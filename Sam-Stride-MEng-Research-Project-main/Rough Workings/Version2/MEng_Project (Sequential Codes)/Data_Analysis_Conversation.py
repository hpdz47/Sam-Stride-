# ---------------------Imports -----------------------------
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
from pydantic import BaseModel, Field, ValidationError
import numpy as np
import pandas as pd
import json
import csv
import os

from vLLM_Configuration import VLLM_Config
from vLLM_Manager import LLM_Manager
from dotenv import load_dotenv
from Singularity_Command_Line_Executor import SingularityCommandLineCodeExecutor
from pathlib import Path
from autogen import Agent
import copy
import pprint
import re
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from Code_Support_System import RAG_Enhancer, Plan_Enforcer, Error_Checking_System

load_dotenv()
#==========================================================================
class Code_Response(BaseModel):
    Shell_Code: str = Field(..., description="The Shell code to be updated.")
    Python_Code: str = Field(..., description="The Python code to be updated.")



#==========================================================================

class Agent_Base():
    def __init__(self,name: str,llm_config: LLMConfig, system_message: str, Update_System_Message: Optional[str] = None, context_variables: Optional[ContextVariables] = None):
        self.name=name
        self.llm_config=llm_config # Composition Not Used here in case strucutred responses are needed for specific agents.
        self.system_message=system_message
        self.human_input_mode="NEVER" # Hard Coded as this is an AUTONMOUS system.
        self.context_variables=context_variables
        if Update_System_Message:
            Updated_Message = [UpdateSystemMessage(Update_System_Message)]
        self._agent=ConversableAgent(
            name=self.name,
            llm_config=self.llm_config,
            system_message=self.system_message,
            human_input_mode=self.human_input_mode,
            update_agent_state_before_reply=Updated_Message if Update_System_Message else None,
            context_variables=self.context_variables,
        )

    @property
    def agent(self) -> ConversableAgent:
        return self._agent # Getter for retrieving the agent instance.

# Setup Specific Agents by Inheritance of main Conversable Agent Setup.

class Coder_Agent(Agent_Base):
    pass
class Code_Reviewer_Agent(Agent_Base):
    pass
class Code_Executor_Agent(Agent_Base):
    pass
class Report_Writer_Agent(Agent_Base):
    pass

#-------------------------------------------------------------------------------
#1----------------------
class Coding_Initializer():
    Type: str=""
    Step:int =0
    Step_Group:list
    def __init__(self,context_variables: ContextVariables, Analysis_Type:str, Iter:int, Max_Rounds:int, Step_Group:list):
        # Start the vLLM servers FIRST!!!!
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()
        self.context_variables = context_variables
        self.Analysis_Type=Analysis_Type
        Coding_Initializer.Type=Analysis_Type
        self.Iter=Iter
        Coding_Initializer.Step=Iter
        self.Max_Rounds=Max_Rounds
        self.correct_format = """{{'Code': {{'Shell_Code': 'ABC Code', 'Python_Code': 'XYZ Code'}}}}"""
        self.Step_Group=Step_Group
        Coding_Initializer.Step_Group=Step_Group

        self.start_index=self.Step_Group[Iter][0]
        self.end_index=self.Step_Group[Iter][1] +1 #Makes the end inclusive.

        def format_plan_steps(plan_group):
            return "\n".join(
                "Step {Step_Number}\n"
                "- Analysis_Type: {Analysis_Type}\n"
                "- Data_File: {Data_File}\n"
                "- Variables: {Variables}\n"
                "- Context: {Context}\n"
                "- Output_Format: {Output_Format}\n"
                .format(**step)
                for step in plan_group
            )


        if self.Analysis_Type =="Usability":
            self.Plan_Group= self.context_variables["Usability_Plan"]["Plan_Section"][self.start_index:self.end_index]
            steps = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in self.Plan_Group
            ]
            self.plan_key = format_plan_steps(steps)
            print(f"Plan Group {self.Plan_Group}")
            print(f"self.plan_key {self.plan_key}")
            self.suggestion_key="{Code_Reviews_Available}" # (T/F) Does not need to be unique as it is reset each time externally.
            self.Updated="{Code_Updated}" # (T/F) Does not need to be unique as it is reset each time externally.
            self.Code=f"""{{{self.Analysis_Type}_Code_{self.Iter}}}""" # Gets correct code for each intstance of the sub-chatroom.
            self.Filename="Usability.md"

        elif self.Analysis_Type =="HPLC":
            self.Plan_Group= self.context_variables["HPLC_Analysis_Plan"]["Plan_Section"][self.start_index:self.end_index]
            steps = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in self.Plan_Group
            ]
            self.plan_key = format_plan_steps(steps)
            self.suggestion_key="{Code_Reviews_Available}" # (T/F) Does not need to be unique as it is reset each time externally.
            self.Updated="{Code_Updated}" # (T/F) Does not need to be unique as it is reset each time externally.
            self.Code=f"""{{{self.Analysis_Type}_Code_{self.Iter}}}""" # Gets correct code for each intstance of the sub-chatroom.
            self.Filename="HPLC.md"
            
        elif self.Analysis_Type =="MS":
            self.Plan_Group= self.context_variables["Mass_Spectrometry_Plan"]["Plan_Section"][self.start_index:self.end_index]
            steps = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in self.Plan_Group
            ]
            self.plan_key = format_plan_steps(steps)
            self.suggestion_key="{Code_Reviews_Available}" # (T/F) Does not need to be unique as it is reset each time externally.
            self.Updated="{Code_Updated}" # (T/F) Does not need to be unique as it is reset each time externally.
            self.Code = f"""{{{self.Analysis_Type}_Code_{self.Iter}}}"""# Gets correct code for each intstance of the sub-chatroom.
            self.Filename="MS.md"
            
        else:
            raise ValueError(f"Invalid Analysis Type: {self.Analysis_Type}. Must be Usability, HPLC or MS")
        self.RAG_key="{RAG_Enhancements}" # Does not need to be unique as it is reset each time externally.
        self.Plan_Enforcer_key="{Enforcer_Feedback}" # Does not need to be unique as it is reset each time externally.
        self.Error_Checker_key="{Errors_Spotted}" # Does not need to be unique as it is reset each time externally.

        self.Coder_Agent_System_Message="""
        You are a Coder Agent that specialises in writing Python code to implement data analysis plans.
        """
        self.Coder_Agent_Update_System_Message=f"""
        ----------- ROLE ---------------
        You are a Chemistry Data Analysis Coding Agent. Your job is to write python code that implements a data analysis plan that has been provided.
        You are using datasets that are very large (hundreds of MBs) and so you must ensure your code analyses the data in a memory-efficient manner. Consider the operations that you are trying to run
        and the n-notation. For example something on O(n^3) is not acceptable here. You may wish to have parallel execution of code. You may wish to use existing python libraries that work more efficiently with large databases or other coding approaches.
        You will be advised on the best practices and relevant python packages to use and you may be required to write some shell code to install dependencies.
        You will be provied with some feedback on the code that you must use to update and improve the code.
        You may not change the plan or falsify any information.
        You have been provided with the main focus area of the topic of data analysis. This is only for background context and some aspects may not be relevant to the specific plan you have been given.
        Other aspects may be very relevant and so you must use your judgement to implement the plan effectively.
        -------------------------------
        ----------- Task -----------------
        You MUST use the Code_Update function to update the code to ensure that it follows the plan and output instructions exactly. You must never attempt to
        change the plan provided.
        - To call the function that correct syntax is: {self.correct_format}
        CRITICAL: When calling Code_Update, you MUST provide BOTH fields:
        - Shell_Code: Shell/bash commands for installing dependencies (e.g., pip install packages). You are only allowed to install from PyPI.
        - Python_Code: The actual Python code for data analysis
        If no shell commands are needed, provide an empty string for Shell_Code, but you MUST still include the field.
        - You MUST NEVER use ```sh or ```python in your code, the code executor will handle this automatically.
        ------------------------------------------------
        --------- CRITICAL FILE PATHS and Output Instructions -------------------
        - Input data files are located in: /inputs/ (read-only directory)
        - All output files (results, graphs, CSV files) must be saved to: /workspace/
        ----------------------------------------------------
        ---------- Context -----------------
        *** The Plan To be Implemented in Code *** 
        {self.plan_key}
        **************
        *** Instructions for Improving Code ***
        Plan Adherence: {self.Plan_Enforcer_key}
        Bugs to Fix: {self.Error_Checker_key}
        **************
        *** Current Code that Requires Updating *** 
        {self.Code}
        **************
        *** Domain Knowledge and Best Practices for Data Analysis ***
        {self.RAG_key}
        **************
        *** Focus Area Statement (For Context Only) ***
        {{Focus_Area_{Coding_Initializer.Type}}}
        """
        self.Code_Reviewer_Agent_System_Message="""
        You are a Code Reviewer that specialises in reviewing Python code for data analysis.
        """
        self.Code_Reviewer_Agent_Update_System_Message=f"""
        ----------- ROLE ---------------
        Your job is to call the Code_Review function.
        ---------------------------------
        """
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()

        Coder_Agent_Name="Coder"
        Code_Reviewer_Agent_Name="CodeReviewer"

        Coder_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.4,enable_thinking=False,LLM_Type="Reasoning").build_config()
        Code_Reviewer_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

                

        self.Coder=Coder_Agent(Coder_Agent_Name,
        Coder_LLM_Config,
        self.Coder_Agent_System_Message,
        self.Coder_Agent_Update_System_Message,self.context_variables)
        self.Code_Reviewer=Code_Reviewer_Agent(Code_Reviewer_Agent_Name,
        Code_Reviewer_LLM_Config,
        self.Code_Reviewer_Agent_System_Message,
        self.Code_Reviewer_Agent_Update_System_Message,self.context_variables)  

        # Handoffs -----------
        self.Coder.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Code_Reviewer.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression(f"${self.Updated} == True")
                )
            )
        )
        self.Coder.agent.handoffs.set_after_work(AgentTarget(self.Coder.agent))

        self.Code_Reviewer.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Coder.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Code_Reviews_Available} == True") # Approval flag removed or testing purposes.
                )
            )
        )

        # Sending to executor after approval has been removed here.
        self.Code_Reviewer.agent.handoffs.set_after_work(AgentTarget(self.Code_Reviewer.agent))

        register_function(
            Code_Update,
            caller=self.Coder.agent,
            executor=self.Coder.agent,
            name="Code_Update",
            description="Update the code to follow the Plan Provided, within the context of the metadata provided."
        )
        register_function(
            Code_Review,
            caller=self.Code_Reviewer.agent,
            executor=self.Code_Reviewer.agent,
            name="Code_Review",
            description="Review the code in context variables and set flags for handoff."
        )
        
    def run_Conversation(self):
        # Transforming Message History (To Limit)----
        context_handling = transform_messages.TransformMessages(
            transforms=[transforms.MessageHistoryLimiter(max_messages=3)])
        context_handling.add_to_agent(self.Coder.agent)
        context_handling.add_to_agent(self.Code_Reviewer.agent)
        #---------------------------
        pattern=DefaultPattern(
        initial_agent=self.Coder.agent,
        agents=[self.Coder.agent,
                self.Code_Reviewer.agent],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )


        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages="Write python code to analyse the data according to the plan provided. Shell code may be necessary to install any dependencies.",
            max_rounds=self.Max_Rounds,
        )
#-------------------------------------------------------------------------------
#2----------------------
# This class does not need to be specific to the any version of analysis as the class variables from the Coding_Initializer are used.
class Coder_Executor():
    def __init__(self,context_variables: ContextVariables, Max_Rounds:int):
        self.context_variables = context_variables
        self.Code=f"""{{{Coding_Initializer.Type}_Code_{Coding_Initializer.Step}}}""" # Gets correct code for each intstance of the sub-chatroom.
        self.Max_Rounds=Max_Rounds
        LLM_Manager(LLM_Type="Coding").Manage_VLLM()
        # Reset Context Variables for Debugging Stage
        self.context_variables["Debug_Updated"] = False

        self.Coding_System_Message="""
        You are a Code refinement and debugging agent.
        """
        self.Coding_Update_System_Message=f"""
        ----------- ROLE ---------------
        You are a coding debugger agent. You will recieve code that has been executed and where execution has failed.
        Your job is to debug the code and fix any issues that are causing the code to fail execution. You will be provided with both the code
        and the error messages. You will use this information to fix the code and then you will then need to suggest the new code to use.
        The code was written to carry out a specific plan and so you must never change the main objective of the code. Major changes will not be tolerated.
        You MUST call the Debug function to provide the updated code. You MUST provide both the Shell code and the Python code.
        You have been provided with information about the variable names and file names that are available. This is to ensure consistency in the code and to avoid errors related to incorrect variable names or file paths.
        --------------------------------
        ------------- Current Code to Debug ------------
        {self.Code}       
        ------------------------------------------------- 
        ------------- Variable and File Name Information ---------
        {{metadata}}
        ---------------------------------------------------------
        """
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Coding").build_config()
        Coding_Agent_Name="Coding"
        Coding_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.4,enable_thinking=False,LLM_Type="Coding").build_config()
        self.Coding=Coder_Agent(Coding_Agent_Name,Coding_LLM_Config,
        self.Coding_System_Message,
        self.Coding_Update_System_Message,self.context_variables)

        # Setup Executor -------------------------
        #              ++++++++++++++++++++++Setup Correct File Paths +++++++++++++
        Results_Folder=Path("./Data_Results") # Path for the temporary files. (Delete after use).
        if not Results_Folder.exists():
            Results_Folder.mkdir(parents=True, exist_ok=True)
        Usability_Results_Folder=Path("./Data_Results/Usability_Results")
        if not Usability_Results_Folder.exists():
            Usability_Results_Folder.mkdir(parents=True, exist_ok=True)
        HPLC_Results_Folder=Path("./Data_Results/HPLC_Results")
        if not HPLC_Results_Folder.exists():
            HPLC_Results_Folder.mkdir(parents=True, exist_ok=True)
        MS_Results_Folder=Path("./Data_Results/MS_Results")
        if not MS_Results_Folder.exists():
            MS_Results_Folder.mkdir(parents=True, exist_ok=True)

        #              ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.inputs_dir=Path("./Inputs")
        if Coding_Initializer.Type =="Usability":
            self.work_dir=Usability_Results_Folder
        elif Coding_Initializer.Type =="HPLC":
            self.work_dir=HPLC_Results_Folder
        elif Coding_Initializer.Type =="MS":
            self.work_dir=MS_Results_Folder
        else:
            raise ValueError(f"Invalid Analysis Type: {Coding_Initializer.Type}. Must be Usability, HPLC or MS")
        self.setup_dir=Path("./Singularity_Images")
        self.pip_dir=Path("./Pip_Install_Coder_Agent")  # Directory to use instead of writable-tmpfs for pip installs.
        self.executor = SingularityCommandLineCodeExecutor(
            image="continuumio/anaconda3",
            timeout=60,
            work_dir=str(self.work_dir),
            setup_dir=str(self.setup_dir),
            inputs_dir=str(self.inputs_dir),
            pip_install_dir=str(self.pip_dir),
        )
        self.Code_Executor_Agent = ConversableAgent("Code_Executor_Agent",
        llm_config=False,  # Turn off LLM for this agent.
        code_execution_config={"executor": self.executor,
        "last_n_messages": 3},  # Use the docker command line code executor.
        human_input_mode="NEVER",
        context_variables=self.context_variables,
        )
        # Handoffs ----------
        self.Code_Executor_Agent.handoffs.add_after_work( # This is the only add_after_work needed.
            OnContextCondition(
                target=AgentTarget(self.Coding.agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Issues} == True")
                )
            )
        )

        self.Code_Executor_Agent.handoffs.add_after_work( # This is the only add_after_work needed.
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Issues} == False")
                )
            )
        )

        self.Coding.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=AgentTarget(self.Code_Executor_Agent),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${Debug_Updated}==True")
                )
            )
        )
        self.Coding.agent.handoffs.set_after_work(AgentTarget(self.Code_Executor_Agent))

        register_function(
            Debug,
            caller=self.Coding.agent,
            executor=self.Coding.agent,
            name="Debug",
            description="Debug the code based on execution errors and provide updated code.",
        )
        self.Code_Executor_Agent.register_hook("process_message_before_send",Execution_Results)
    def run_Conversation(self):
        # Transforming Message History (To Limit)----
        context_handling = transform_messages.TransformMessages(
            transforms=[transforms.MessageHistoryLimiter(max_messages=3)])
        context_handling.add_to_agent(self.Coding.agent)
        #---------------------------
        pattern=DefaultPattern(
        initial_agent=self.Code_Executor_Agent,
        agents=[self.Code_Executor_Agent, self.Coding.agent],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )


        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages=f"""```sh \n{self.context_variables[f"{Coding_Initializer.Type}_Shell_Code_{Coding_Initializer.Step}"]} \n``` \n\n```python \n{self.context_variables[f"{Coding_Initializer.Type}_Python_Code_{Coding_Initializer.Step}"]} \n```""",
            max_rounds=self.Max_Rounds,
        )

def Debug(
    Code: Annotated[Code_Response, "The Python code to be updated."],
    context_variables: ContextVariables) -> ReplyResult:

    context_variables[f"Debug_Updated"] = True
    context_variables[f"{Coding_Initializer.Type}_Shell_Code_{Coding_Initializer.Step}"] = f"{Code.Shell_Code}"
    context_variables[f"{Coding_Initializer.Type}_Python_Code_{Coding_Initializer.Step}"] = f"{Code.Python_Code}"
    context_variables[f"{Coding_Initializer.Type}_Code_{Coding_Initializer.Step}"]=f"#Shell_Code: \n {Code.Shell_Code}\n\n #Python_Code: \n {Code.Python_Code}"
    # Resetting Code Reviewer Context variables to be used with the necessary handoffs.
    context_variables["Issues"] = False
    #--- Message Must be Repeated for the Executor to Work Properly ----
    message=f"""```sh \n{context_variables[f"{Coding_Initializer.Type}_Shell_Code_{Coding_Initializer.Step}"]} \n``` \n\n```python \n{context_variables[f"{Coding_Initializer.Type}_Python_Code_{Coding_Initializer.Step}"]} \n```"""
    return ReplyResult(
        message=message,
        context_variables=context_variables,
    )


#---------------------------------------------------------------------------------
#3----------------------
class Data_Analysis_Chat():
    def __init__(self,context_variables: ContextVariables, Step_Size:int):
        self.context_variables = context_variables
        self.step_size=Step_Size
        if self.step_size <=0:
            raise ValueError("Step Size must be a positive integer greater than 0.")

        self.Usability_Number_of_Steps=self.context_variables["Usability_Plan"]["Number_of_Steps"]
        self.HPLC_Number_of_Steps=self.context_variables["HPLC_Analysis_Plan"]["Number_of_Steps"]
        self.MS_Number_of_Steps=self.context_variables["Mass_Spectrometry_Plan"]["Number_of_Steps"]

        def Groups(N, step_size):
            groups = []
            for start in range(0, N, step_size):
                end = min(start + step_size - 1, N - 1)
                groups.append((start, end))
            return groups
        self.Usability_Groups=Groups(self.Usability_Number_of_Steps,self.step_size)
        self.HPLC_Groups=Groups(self.HPLC_Number_of_Steps,self.step_size)
        self.MS_Groups=Groups(self.MS_Number_of_Steps,self.step_size)

        self.Usability_Length=len(self.Usability_Groups)
        self.HPLC_Length=len(self.HPLC_Groups)
        self.MS_Length=len(self.MS_Groups)


    def run_Conversation(self):
        # Commented out for now !!!!!!!!!!!!!!!!!!
        # for i in range(self.Usability_Length):
        #     self.context_variables[f"Usability_Code_{i}"] = ""
        #     self.context_variables[f"Usability_Shell_Code_{i}"] = ""
        #     self.context_variables[f"Usability_Python_Code_{i}"] = ""
        #     self.context_variables["Approval"]=False
        #     RAG1=RAG_Enhancer(context_variables=self.context_variables, Analysis_Type="Usability", Max_Rounds=10)
        #     RAG1.run_Conversation()
        #     coding_chat= Coding_Chat(self.context_variables,Analysis_Type="Usability",Iter=i,Max_Rounds=60, Step_Group=self.Usability_Groups)
        #     coding_chat.run_Conversation()
           
        # Clean Context Variables:
        self.context_variables["Issues"] = False
        self.context_variables["Code_Reviews_Available"] = False
        self.context_variables["Code_Updated"] = False
        self.context_variables["Code_Reviews"] = ""
        self.context_variables["Approval"]=False
        for i in range(self.HPLC_Length):
            self.context_variables[f"HPLC_Code_{i}"] = ""
            self.context_variables[f"HPLC_Shell_Code_{i}"] = ""
            self.context_variables[f"HPLC_Python_Code_{i}"] = ""
            self.context_variables["Approval"]=False
            coding_chat= Coding_Initializer(self.context_variables,Analysis_Type="HPLC",Iter=i,Max_Rounds=20, Step_Group=self.HPLC_Groups)
            coding_chat.run_Conversation()
            Debug_Execute=Coder_Executor(context_variables=self.context_variables, Max_Rounds=30)
            Debug_Execute.run_Conversation()
           
        # # Clean Context Variables:
        # self.context_variables["Issues"] = False
        # self.context_variables["Code_Reviews_Available"] = False
        # self.context_variables["Code_Updated"] = False
        # self.context_variables["Code_Reviews"] = ""
        # self.context_variables["Approval"]=False
        # for i in range(self.MS_Length):
        #     self.context_variables[f"MS_Code_{i}"] = ""
        #     self.context_variables[f"MS_Shell_Code_{i}"] = ""
        #     self.context_variables[f"MS_Python_Code_{i}"] = ""
        #     self.context_variables["Approval"]=False
        #     coding_chat= Coding_Chat(self.context_variables,Analysis_Type="MS",Iter=i,Max_Rounds=100, Step_Group=self.MS_Groups)
        #     coding_chat.run_Conversation()
            
        # # Clean Context Variables:
        # self.context_variables["Issues"] = False
        # self.context_variables["Code_Reviews_Available"] = False
        # self.context_variables["Code_Updated"] = False
        # self.context_variables["Code_Reviews"] = ""
        # self.context_variables["Approval"]=False

#==================== Functions to Pass to Agents =======================
def Code_Update(
    Code: Annotated[Code_Response, "The Python code to be updated."],
    context_variables: ContextVariables) -> ReplyResult:
    """Update the code to follow the Plan Provided, within the context of the metadata provided."""

    context_variables[f"Code_Updated"] = True
    context_variables[f"{Coding_Initializer.Type}_Shell_Code_{Coding_Initializer.Step}"] = f"{Code.Shell_Code}"
    context_variables[f"{Coding_Initializer.Type}_Python_Code_{Coding_Initializer.Step}"] = f"{Code.Python_Code}"
    context_variables[f"{Coding_Initializer.Type}_Code_{Coding_Initializer.Step}"]=f"#Shell_Code: \n {Code.Shell_Code}\n\n #Python_Code: \n {Code.Python_Code}"

    # Resetting Code Reviewer Context variables to be used with the necessary handoffs.
    context_variables[f"Code_Reviews_Available"] = False
    context_variables["Issues"] = False
    with open("Debug.json", "w") as f:
        json.dump(context_variables.model_dump(), f, indent=2) # Useful only for debugging purposes when code hasn't executed properly.

    return ReplyResult(
        message=f"Code stored successfully. Ready for review.",
        context_variables=context_variables,
    )

def Code_Review(context_variables: ContextVariables) -> ReplyResult:
    # """Review the code in context variables and set flags for handoff."""
    # context_variables[f"Code_Reviews"] = Code_Reviews

    # if Test_Approval_Status:
    #     context_variables["Approval"]=True
    #     message=f"""```sh \n{context_variables[f"{Coding_Chat.Type}_Shell_Code_{Coding_Chat.Step}"]} \n``` \n\n```python \n{context_variables[f"{Coding_Chat.Type}_Python_Code_{Coding_Chat.Step}"]} \n```  """
    context_variables[f"Code_Reviews_Available"] = True
    # Resetting Coder Context variables to be used with the necessary handoffs.
    context_variables[f"Code_Updated"] = False

    #----- Activate Review Panel ------
    RAG=RAG_Enhancer(context_variables=context_variables, Analysis_Type=Coding_Initializer.Type, Max_Rounds=10,Iter=Coding_Initializer.Step, Step_Group=Coding_Initializer.Step_Group)
    RAG.run_Conversation()
    PF=Plan_Enforcer(context_variables=context_variables, Analysis_Type=Coding_Initializer.Type, Max_Rounds=10, Iteration=1, Step_Group=Coding_Initializer.Step_Group, Iter=Coding_Initializer.Step)
    PF.run_Conversation()
    EC=Error_Checking_System(context_variables=context_variables, Analysis_Type=Coding_Initializer.Type, Max_Rounds=10, Iteration=1, Step_Group=Coding_Initializer.Step_Group, Iter=Coding_Initializer.Step)
    EC.run_Conversation()
    #----------------------------------

    return ReplyResult(
        message="The code has been reviewed and requires updating.",
        context_variables=context_variables,
    )

def Execution_Results(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:
    """Hook to check code execution results and update context variables."""
    # Extract content from message
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = str(message)
    
    # Check for execution errors
    if "exitcode: 1" in content or "execution failed" in content.lower() or "error occurred:" in content:
        sender.context_variables["Issues"] = True
        sender.context_variables["Approval"] = False # Retracts approval to ensure that the code changes are reviewed in light of the plan to ensure no large scale changes are made.
        sender.context_variables["Debug_Updated"] = False
        # Modify the message to include error info
        if isinstance(message, dict):
            message["content"] = f"CODE EXECUTION FAILED:\n{content}\n\nPlease fix the code."
            return message
        else:
            return f"CODE EXECUTION FAILED:\n{content}\n\nPlease fix the code."
    
    if "exitcode: 0" in content:
        sender.context_variables["Issues"] = False
        # Modify the message to include success info  
        if isinstance(message, dict):
            message["content"] = f"CODE EXECUTION SUCCESSFUL:\n{content}"
            return message
        else:
            return f"CODE EXECUTION SUCCESSFUL:\n{content}"
    
    # Return message unchanged if no execution result found
    return message


def main():
    # This code is just for unit testing, the Planning_Conversation.json file needs to be
    #loaded into the Current Working Directory by running the Planning_Conversation.py
    #file first.
    #------------------------------------------------------
    with open("Planning_Conversation.json", "r") as f:
        CV=json.load(f)
        # The data is nested under "data" key
        data_section = CV.get("data", {})
        #metadata_value = data_section.get("metadata", {})
        #plan_value = data_section.get("Plan", {})
        
    with open("Data_Analysis_Conversation.json", "r") as f:
        CV1=json.load(f)
        data_section1 = CV1.get("data", {})
        Usability_Plan= data_section1.get("Usability_Plan", {})
        HPLC_Analysis_Plan= data_section1.get("HPLC_Analysis_Plan", {})
        Mass_Spectrometry_Plan= data_section1.get("Mass_Spectrometry_Plan", {})
        #DS=data_section1.get("Data_Discovery_Summary", {})
    #------------------------------------------------------
    context=ContextVariables({
        "metadata": "'metadata chromatography_combined.csv' \n Variables: ['Unnamed: 0', 'start_well', 'end_well', 'Sample_Code', 'process_part', 'run', 'chromatography_stage', 'COLUMN_IN_USE', 'Resin_type', 'column', 'Column_Volume_(ml)', 'Load_volume_(mL)', 'Load_pH', 'Unicorn_server_result_file', 'Exported_run_data_file', 'Unicorn_Method_Name', 'Chromatography_Plate_ID', 'Plate_start', 'Plate_end', 'Chromatography_Plate_ID_2', 'Plate2_start', 'Plate2_end', 'run_no', 'Run_Name', 'Excel_Filename', 'Shortened_Code_for_Bioaccord_Samples', 'Run_variables_file', 'Operator_name_(Full)', 'Date', 'Upstream_sample', 'Aliquot_code', 'Time_samples_removed_from_fridge', 'Time_sample_returned_to_fridge', 'IDBS_experiment_number', 'Sample_start_volume_(mL)', 'Sample_volume_after_dilution_with_HPW_(mL)', 'Sample_final_volume_after_pH_titration_(mL)', 'Titration_agent', 'Start_pH', 'Final_pH', 'Total_protein_concentration_(bradford)_(mg/mL)', 'Conducivity_(mS)', 'Concentration_(mg/mL)', 'Filter_type_used', 'Load_sample_start_ID', 'Load_sample_start_IDBS_name', 'Load_sample_end_IDBS_name', 'EQ_and_wash_buffer', 'EQ_and_Wash_Buffer_Inlet', 'Elution_buffer', 'Elution_Buffer_Inlet', 'Sanitisation_buffer', 'Other_buffer', 'All_columns_loaded_from_same_material', 'Notes', 'fraction_volume', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise'] \n -----'MS_combined.csv' ------ \n ['Unnamed: 0', 'Unique_Peak_ID', 'Unique_MS_Sample_ID', 'Sample_Code', 'run_code', 'process_part', 'run', 'MS_method', 'column', 'chromatography_stage', 'start_well', 'end_well', 'Replicate', 'Type', 'Molecule ID', 'Component', 'Observed_TIC_RT_(mins)', 'Observed_UV_RT_(mins)', 'Observed RT delta (mins)', 'Response', '%_of_response', 'Observed_neutral_mass_(Da)', 'Observed_m/z', 'Spectrum_type', 'Expected_mass_(Da)', 'Mass_error_(ppm)', 'Alternative_assignments', 'fraction_volume', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise']\n ------'chromatography_combined.csv'------ \n ['Unnamed: 0', 'run_no', 'run', 'Fraction_unique_ID', 'column', 'volume_ml', 'UV_1_280_ml', 'UV_1_280_mAU', 'Cond_ml', 'Cond_mS_cm', 'Conc_B_ml', 'Conc_B_%', 'Injection_ml', 'Injection_Injection', 'Run_Log_ml', 'Run_Log_Logbook', 'Fraction_ml', 'Fraction_Fraction', 'UV_1_280_CUT_TEMP_100_BASEM_ml', 'UV_1_280_CUT_TEMP_100_BASEM_mAU', 'UV_2_260_ml', 'UV_2_260_mAU', 'pH_ml', 'pH_pH', 'DeltaC_pressure_ml', 'DeltaC_pressure_MPa', 'System_flow_ml', 'System_flow_ml_min', 'Sample_flow_ml', 'Sample_flow_ml_min', 'Sample_Code', 'chromatography_stage', 'chromatography_stage_order', 'Fraction_number', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise', 'fraction_volume']",
        "Plan": "",
        "Usability_Plan": Usability_Plan,
        "HPLC_Analysis_Plan": HPLC_Analysis_Plan,
        "Mass_Spectrometry_Plan": Mass_Spectrometry_Plan,
        "Issues": False,
        "Data_Discovery_Summary": "",
        "Univariate_EDA_Report": "### Dataset Interpretation Report\n\nThis report provides a detailed interpretation of the three files in the dataset: `metadata chromatography_combined.csv`, `MS_combined.csv`, and `chromatography_combined.csv`. Each file is analyzed independently to understand its structure, content, and relevance for downstream analysis.\n\n---\n\n### 1. `metadata chromatography_combined.csv`\n\n**File Structure and Type:**\n- This is a CSV file with a data shape of 354 rows and 58 columns.\n- The file contains metadata related to chromatography runs, including experimental parameters, sample details, buffer information, and operational notes.\n\n**Key Observations:**\n- **Data Types:** The file contains a mix of `int64`, `float64`, and `object` (string) data types. Notably, several columns (e.g., `start_well`, `end_well`, `Sample_Code`) are of type `object`, indicating categorical or textual data.\n- **Missing Values:** Many columns (e.g., `Time_samples_removed_from_fridge`, `Time_sample_returned_to_fridge`, `Notes`) have no observed values (all NaNs), suggesting they may be unused or incomplete.\n- **Dense Data:** The dataset is dense, with most numerical columns having meaningful values. The `Unnamed: 0` column appears to be an index, with values ranging from 0 to 49.\n- **Categorical Variables:** Several columns are highly categorical:\n  - `Resin_type` (4 unique values)\n  - `chromatography_stage` (16 unique values)\n  - `Run_Name`, `Excel_Filename`, `Operator_name_(Full)` (11, 11, and 1 unique values respectively), indicating a small number of experimental runs.\n- **Numerical Variables:** Key experimental parameters include:\n  - `Load_volume_(mL)` (mean ~112 mL, range 35–188 mL)\n  - `Sample_start_volume_(mL)` (mean ~166 mL, range 90–270 mL)\n  - `Total_protein_concentration_(bradford)_(mg/mL)` (mean ~0.16 mg/mL, range 0.10–0.44 mg/mL)\n  - `Start_pH` and `Final_pH` (mean ~7.1 and ~6.2, respectively)\n- **Grouping:** The data is grouped by `run`, `run_no`, and `Sample_Code`, suggesting that each row corresponds to a unique experimental run or sample.\n- **Relevance:** This file serves as a comprehensive metadata source for chromatography experiments, linking sample, process, and operational details. It is essential for contextualizing the other two datasets.\n\n**Conclusion:** This file is a dense, structured metadata file critical for understanding the experimental setup. It should be used to enrich and validate the other datasets.\n\n---\n\n### 2. `MS_combined.csv`\n\n**File Structure and Type:**\n- This is a large CSV file (620.35 MB) with 1,912,029 rows and 30 columns.\n- It contains mass spectrometry (MS) data, likely from peptide or protein identification and quantification.\n\n**Key Observations:**\n- **Data Types:** Mixed types, with `int64`, `float64`, and `object` columns. The `Unnamed: 0` column appears to be an index.\n- **Missing Values:** Several columns (e.g., `Type`, `Molecule ID`, `Component`, `Expected_mass_(Da)`, `Mass_error_(ppm)`, `Alternative_assignments`) have no observed values (all NaNs), indicating they may be unused or incomplete.\n- **Sparse Data:** Despite its large size, the dataset is sparse in certain key columns, particularly those related to molecular identification.\n- **Categorical Variables:**\n  - `Unique_MS_Sample_ID` (283 unique values)\n  - `Sample_Code` (274 unique values)\n  - `chromatography_stage` (19 unique values)\n  - `Spectrum_type` (3 unique values)\n- **Numerical Variables:** Key MS features include:\n  - `Observed_TIC_RT_(mins)` (mean ~2.33 mins, range ~0.62–3.03 mins)\n  - `Observed_UV_RT_(mins)` (mean ~1.98 mins, range ~0.74–2.72 mins)\n  - `Response` (mean ~9,920, range ~4.86–15.25M)\n  - `Observed_neutral_mass_(Da)` (mean ~162.5 kDa, range ~400–314 kDa)\n  - `Observed_m/z` (mean ~733 Da, range ~505–1,263 Da)\n- **Grouping:** Data is grouped by `Unique_MS_Sample_ID`, `Sample_Code`, and `run_code`, suggesting a hierarchical structure where multiple MS spectra are associated with a single sample.\n- **Relevance:** This file contains high-resolution MS data, likely from LC-MS/MS experiments. It is essential for downstream analysis such as peptide identification, quantification, and post-translational modification (PTM) analysis.\n\n**Conclusion:** This file is a large, sparse dataset containing detailed MS data. It is highly relevant for proteomics analysis but requires careful handling due to missing values in key identification columns.\n\n---\n\n### 3. `chromatography_combined.csv`\n\n**File Structure and Type:**\n- This is a large CSV file (257.96 MB) with 1,084,069 rows and 37 columns.\n- It contains chromatographic data, including UV, conductivity, pH, flow, and volume measurements across fractions.\n\n**Key Observations:**\n- **Data Types:** Mixed types, with `int64`, `float64`, and `object` columns. The `Unnamed: 0` column is likely an index.\n- **Missing Values:** Some columns (e.g., `Injection_Injection`, `Fraction_Fraction`) have no observed values (all NaNs), suggesting they may be unused.\n- **Dense Data:** The dataset is dense, with most numerical columns having meaningful values. The `Fraction_number` and `Sample_Code` columns suggest a hierarchical structure where multiple fractions are generated per sample.\n- **Categorical Variables:**\n  - `column` (4 unique values)\n  - `chromatography_stage` (16 unique values)\n  - `Run_Log_Logbook` (17 unique values)\n- **Numerical Variables:** Key chromatographic parameters include:\n  - `volume_ml` (mean ~126 mL, range ~-33.8–335.8 mL)\n  - `UV_1_280_ml` and `UV_1_280_mAU` (mean ~126 mL and ~250 mAU, respectively)\n  - `Cond_ml` and `Cond_mS_cm` (mean ~126 mL and ~15.8 mS/cm)\n  - `pH_ml` and `pH_pH` (mean ~132.5 mL and ~6.28 pH)\n  - `fraction_volume` (mean ~20.9 mL, range ~0.008–66.0 mL)\n- **Grouping:** Data is grouped by `Sample_Code`, `run_no`, and `Fraction_number`, indicating that each row corresponds to a fraction collected during a chromatography run.\n- **Relevance:** This file contains high-resolution chromatographic profiles, essential for understanding elution patterns, peak detection, and fraction collection. It is critical for integrating with MS data to link protein identity to chromatographic behavior.\n\n**Conclusion:** This file is a dense, structured chromatographic dataset that provides detailed fraction-level data. It is indispensable for correlating MS results with chromatographic elution profiles.\n\n---\n\n### Overall Summary\n\n- The dataset consists of three interrelated files:\n  1. `metadata chromatography_combined.csv`: Metadata for experimental runs.\n  2. `chromatography_combined.csv`: High-resolution chromatographic profiles (fraction-level).\n  3. `MS_combined.csv`: High-resolution MS data (peptide/protein-level).\n\n- **Integration Potential:** The three files can be integrated using common keys such as `Sample_Code`, `run_no`, and `chromatography_stage`. This integration enables a comprehensive analysis of protein behavior across chromatography and MS platforms.\n\n- **Gaps and Limitations:**\n  - Several columns in `MS_combined.csv` are entirely missing (all NaNs), which may limit downstream identification.\n  - Some columns in all files (e.g., `Notes`, `Other_buffer`) are unused or incomplete.\n  - Further validation is required to confirm the consistency of `Sample_Code` and `run_no` across files.\n\n- **Recommendations for Further Analysis:**\n  - Perform data merging using `Sample_Code` and `run_no` to align chromatography and MS data.\n  - Investigate the reasons for missing values in `MS_combined.csv`.\n  - Validate the integrity of `fraction_volume` and related columns across files.\n  - Conduct a time-series analysis of chromatographic profiles to identify elution patterns.\n\nThis dataset is well-structured for integrative proteomics analysis, provided that missing data issues are addressed.",

        # Specific to Data Analysis Chatroom:
        "Code_Reviews_Available": False,
        "Code_Updated": False,
        "Code_Reviews": "",
        "Issues": False,
        "Usability_Shell_Codes": "",
        "HPLC_Shell_Codes": "",
        "MS_Shell_Codes": "",
        "Usability_Python_Codes": "",
        "HPLC_Python_Codes": "",
        "MS_Python_Codes": "",
        "Usability_Shell_Code":"",
        "Usability_Python_Code":"",
        "HPLC_Shell_Code":"",
        "HPLC_Python_Code":"",
        "MS_Shell_Code":"",
        "MS_Python_Code":"",
        "Approval": False,
        #----------------------
        "Focus_Area_Usability": "The focus area is the assessment of data quality, reliability, and relevance of High Performance Liquid Chromatography (HPLC) and Mass Spectrometry (MS) data generated during a biopharmaceutical manufacturing process. Data analysis in this context involves evaluating the integrity of chromatographic and MS datasets—specifically, identifying anomalies such as sensor drift, noise, missing values, and inconsistent metadata—to ensure that the data accurately reflects the biological and process characteristics of the product. The primary goal is to verify that the data is fit for purpose in supporting critical quality attributes (CQAs), such as purity, identity, and structural integrity of the biopharmaceutical product. This includes validating the consistency of sample tracking across files, detecting outliers in elution profiles or MS response, and ensuring that MS identifications are supported by reliable chromatographic co-elution. Appropriate data analysis includes SNR Calculation, Chromatographic Peak Shape Analysis, MS Signal Consistency Check, Metadata Alignment Validation, Missing Data Pattern Analysis, Chromatography-MS Integration, Sensor Drift Detection, Outlier Detection in Response Intensity, RT Alignment Validation, and Data Completeness Assessment. The context is that the datasets are large, high-resolution, and hierarchical, with multiple levels of sample and run metadata; therefore, analysis must account for the expected variability in elution times and response intensities due to process dynamics, while distinguishing true anomalies from normal process variation.",
        "Focus_Area_HPLC": "**Statement**\nHPLC data analysis in the context of biopharmaceutical manufacturing focuses on characterizing chromatographic profiles from fraction-level data to assess product quality, purity, and consistency. The primary objective is to identify and quantify target protein peaks, detect impurities or degradation products, and ensure process reproducibility across runs. This analysis relies on the integrity, quality, and relevance of chromatographic data, including UV absorbance, conductivity, pH, and volume measurements, to support process understanding, release testing, and regulatory compliance.\n\n**Suggested Data Analysis**\nPeak Detection, Peak Integration, Chromatographic Alignment, Time-Series Analysis, Multivariate Analysis (PCA), Process Variability Assessment, Fraction Classification, Elution Profile Comparison, Outlier Detection, Run-to-Run Consistency Analysis\n\n**Context**\nThe chromatography data is collected at the fraction level, with each row representing a fraction from a specific run and sample. Data is grouped by `Sample_Code`, `run_no`, and `Fraction_number`, enabling analysis at the fraction, run, and process stage levels. The data is dense and high-resolution, making it suitable for detailed peak profiling and comparison across stages (e.g., capture, polishing). Analysis should account for variations in load volume, pH, buffer composition, and resin type, as these are captured in the metadata and can influence elution behavior. Integration with metadata (e.g., `chromatography_stage`, `Resin_type`, `Start_pH`) is essential for contextual interpretation.",
        "Focus_Area_MS": "**Statement**\nMass spectrometry (MS) data analysis in biopharmaceutical manufacturing aims to identify and quantify proteins and peptides, assess their structural integrity, and detect critical quality attributes (CQAs) such as post-translational modifications (PTMs), deamidation, oxidation, and truncations. This analysis is essential for ensuring product consistency, safety, and efficacy, with the primary result being a reliable characterization of the biologic’s molecular identity and purity. Data quality, reliability, and relevance are ensured by validating spectral integrity, alignment across runs, and consistency with chromatographic elution profiles.\n\n**Suggested Data Analysis**\n- Peak Detection and Integration\n- Protein/Peptide Identification (via database search)\n- Quantification (label-free or label-based)\n- Post-Translational Modification (PTM) Analysis\n- Mass Error Assessment\n- Fragmentation Pattern Analysis\n- Multivariate Analysis (PCA, PLS-DA)\n- Batch Effect Correction\n- Temporal Profile Analysis (by chromatography stage)\n- Spectral Quality Control (SQC)\n\n**Context**\nThe MS data is collected across multiple chromatography stages and runs, with samples linked via `Sample_Code`, `run_code`, and `chromatography_stage`. Integration with chromatography data enables correlation of MS signals with elution behavior, enhancing confidence in identifications. The presence of missing values in key identification columns (e.g., `Molecule ID`, `Expected_mass_(Da)`) necessitates careful data filtering and quality control. Analysis must account for run-to-run variability and ensure that only high-quality, reproducible spectra are used for downstream characterization.",
        #----- Coding Support System --------
        "Code_QA_Available": False,
        "Coding_RAG_QA": "",
        "RAG_Enhancements": "",
        "RAG_Enhancement_Available": False,
        #-------- Iter 2 of Code Support System --------
        "Enforcer_Feedback":"",
        "Enforcer_Used": False,
        #---------- Iter 3 of Code Support System --------
        "Errors_Checked": False,
        "Errors_Spotted": "",
        "Debug_Updated": False,

    })
    data_analysis_chat= Data_Analysis_Chat(context,Step_Size=1)
    data_analysis_chat.run_Conversation()

    # Note: Logic can be implemented to adjust the step size based on the number of steps in the plan. But, the planing chat
    # should be encouraged to write a plan consisting of < 10 steps for best results.

    with open("Data_Analysis_Conversation.json", "w") as f:
        json.dump(context.model_dump(), f, indent=2)

if __name__ =="__main__":
    main()