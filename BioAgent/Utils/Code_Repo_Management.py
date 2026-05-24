from __future__ import annotations
# This script s for setting up the GitManagers for the coding review panel phase.
# One GitManager is required for each review agent type to allow for short-term memory management during the coding review phase.
# Each GitManager will point to its own unique repository, and the GitMemory can work wit multiple GitManagers
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

def create_code_git_managers(context_variables: ContextVariables, Root_Dir: Path, Git_Memory: GitMemory) -> None:
    """
    This function accepts the context variables so that it can create te GitManagers at the Implementation phase.
    If the implementation phase is repeated, then the GitManagers should be re-created and overwrite previous
    global iteration attempts.

    - Root_Dir: This will be used to set the root directory so that the paths for each reviewer type
                can be set correctly and passed to each GitManager.
    - Context Variables: This is used to allow access to these GitManagers from any file that is relevant.
    - Git Memory: This is used to allow the GitManagers to access the Singleton GitMemory instance which 
                  is passed down from BioAgent.
    """
    # Error Check Loop:
    ECL_GitManager = GitManager(plan_step = "Error_Correction", Git_Memory=Git_Memory, context_variables=context_variables, Hard_Reset=True) # Only Hard Reset on the frst creation. This is CRUCIAL.
    context_variables["ECL_Manager"] = ECL_GitManager

    # Plan Enforcement Loop:
    PEL_GitManager = GitManager(plan_step = "Plan_Enforcement", Git_Memory=Git_Memory, context_variables=context_variables, Hard_Reset=False) # Hard Reset = False as it was already done by the first GitManager creation of this cycle.
    context_variables["PEL_Manager"] = PEL_GitManager

    # Optimisation Loop:
    OL_GitManager = GitManager(plan_step = "Optimisation", Git_Memory=Git_Memory, context_variables=context_variables, Hard_Reset=False) # Hard Reset = False as it was already done by the first GitManager creation of this cycle.
    context_variables["OL_Manager"] = OL_GitManager

    # Code Manager (For Diff Agent):
    Code_GitManager = GitManager(plan_step = "Code_Manager", Git_Memory=Git_Memory, context_variables=context_variables, Hard_Reset=False) # Hard Reset = False as it was already done by the first GitManager creation of this cycle.
    context_variables["Code_Manager"] = Code_GitManager