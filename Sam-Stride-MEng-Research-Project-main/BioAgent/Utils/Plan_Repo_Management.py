from __future__ import annotations
# This script s for setting up the GitManagers for the planning review panel phase.
# One GitManager is required for each review agent type to allow for short-term memory management during the planning phase.
# Each GitManager will point t its own unque repository, and the GitMemory can work wit multiple GitManagers
# as long as the .set_step() method is used to load the relevant progress for each reviewer type.

from pathlib import Path
import json
from typing import List, Dict, Optional, Union, Any
import atexit
import logging
import uuid
from hashlib import md5
from time import sleep
from types import TracebackType
import os
import subprocess
from Utils.Git_Memory_Handling import GitMemory
from Utils.Git_Manager import GitManager
import shutil
from autogen.agentchat.group import ContextVariables, ReplyResult

def create_plan_git_managers(context_variables: ContextVariables, Root_Dir: Path, Git_Memory: GitMemory) -> None:
    """
    This function accepts the context variables so that it can create te GitManagers at the Planning phase.
    If the planning phase is repeated, then the GitManagers should be re-created and overwrite previous
    global iteration attempts.

    - Root_Dir: This will be used to set the root directory so that the paths for each reviewer type
                can be set correctly and passed to each GitManager.
    - Context Variables: This is used to allow access to these GitManagers from any file that is relevant.
    - Git Memory: This is used to allow the GitManagers to access the Singleton GitMemory instance which 
                  is passed down from BioAgent.
    """
    # Variable Checker:
    Var_GitManager = GitManager(plan_step = "Variable_Checker", Git_Memory=Git_Memory, context_variables=context_variables, Hard_Reset=True) # Only Hard Reset on the frst creation. This is CRUCIAL.
    context_variables["Var_Manager"] = Var_GitManager

    # Focus Area Assessor:
    FA_GitManager = GitManager(plan_step = "Focus_Area_Assessor", Git_Memory=Git_Memory, context_variables=context_variables, Hard_Reset=False) # Hard Reset = False as it was already done by the first GitManager creation of this cycle.
    context_variables["FA_Manager"] = FA_GitManager

    # Output Reviewer:
    Output_GitManager = GitManager(plan_step = "Output_Reviewer", Git_Memory=Git_Memory, context_variables=context_variables, Hard_Reset=False) # Hard Reset = False as it was already done by the first GitManager creation of this cycle.
    context_variables["Output_Manager"] = Output_GitManager

    # Context Reviewer:
    Context_GitManager = GitManager(plan_step = "Context_Reviewer", Git_Memory=Git_Memory, context_variables=context_variables, Hard_Reset=False) # Hard Reset = False as it was already done by the first GitManager creation of this cycle.
    context_variables["Context_Manager"] = Context_GitManager

    # Plan Manager (For Diff Agent):
    Plan_GitManager = GitManager(plan_step = "Plan_Manager", Git_Memory=Git_Memory, context_variables=context_variables, Hard_Reset=False) # Hard Reset = False as it was already done by the first GitManager creation of this cycle.
    context_variables["Plan_Manager"] = Plan_GitManager
    