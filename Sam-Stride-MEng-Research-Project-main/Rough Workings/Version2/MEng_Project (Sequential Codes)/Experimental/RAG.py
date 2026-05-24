# ---------------------Imports -----------------------------
from autogen.agentchat.group import AgentTarget, RevertToUserTarget, OnCondition, StringLLMCondition
from autogen.agentchat.group import OnContextCondition, ExpressionContextCondition, ContextExpression
from autogen.agentchat.group.guardrails import Guardrail, RegexGuardrail, GuardrailResult
from autogen import UserProxyAgent, ConversableAgent, LLMConfig, register_function, AssistantAgent
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat.group import ContextVariables, ReplyResult
from autogen.agentchat.group import StringContextCondition
from autogen.agentchat import initiate_group_chat
from autogen import GroupChat, GroupChatManager
from autogen.agentchat.group.targets.transition_target import StayTarget, TerminateTarget
from typing import Any, Dict, List, Optional, Annotated
from autogen import UpdateSystemMessage
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
import numpy as np
import pandas as pd
import json
import csv
import os
from autogen.tools.experimental import DeepResearchTool

from vLLM_Configuration import VLLM_Config
from vLLM_Manager import LLM_Manager
from dotenv import load_dotenv
import subprocess
from pathlib import Path
import logging
import socket
import shutil
import time

def RAG_Mode(query: Annotated[str,"The Query to ask the RAG agent about documents"], context_variables: ContextVariables)-> ReplyResult:
     # ---- Setting up the Singularity Container with Python installed.
    Python_Image_Docker="docker://mcr.microsoft.com/playwright/python:v1.57.0-noble" 
    
    Python_Image_Singularity="Python_Image.sif"
    # Include in dependencies with BioAgent:
    RAG_scripts_dir=Path("./RAG_Scripts") # Always exists as it has the python files needed for the RAG agent.

    # Not required prior to running BioAgent.
    setup_dir=Path("./Singularity_Images") # Path for the SIF.
    pip_dir=Path("./Pip_Packages") # Path for the Pip Packages. (Only used for when the sessiondir max size is too small).
    temp_dir=Path("./RAG_Temp") # Path for the temporary files. (Delete after use).
    DB_dir=Path("./RAG_DB") # Path for the ChromaDB.
    Files_dir=Path("./Deep_Research_Reports") # Path for the Files.
    hf_cache_dir=Path("./HF_Cache") # Path for the Hugging Face Cache.
    if not hf_cache_dir.exists():
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
    if not DB_dir.exists():
        DB_dir.mkdir(parents=True, exist_ok=True)
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)
    if not setup_dir.exists():
        setup_dir.mkdir(parents=True, exist_ok=True)
    # pip_dir will be made later in script.

    python_singularity_path=setup_dir/f"{Python_Image_Singularity}"

    if not python_singularity_path.exists():
        logging.info(f"Singularity image not found.  Pulling {Python_Image_Docker}...")
        result = subprocess.run(
            ["singularity", "pull", str(python_singularity_path), Python_Image_Docker],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise ValueError(f"Failed to pull image {Python_Image_Docker}: {result.stderr}")

    container_name="RAG_Container"
    # This container differs from the contianer used in the Singularity COmmand Line Executor script (persistent container),
    # This container is set to be ephemeral to ensure all files are deleted after the research is complete, or if it errors.
    # This will prevent the need for cleanup functions and destroy any possibly malicious files.

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    #print(f"Hostname: {hostname} | IP Address: {ip_address}")

    # If Pip_Packages not setup yet:
    if not pip_dir.exists():
        logging.info(f"Pip packages not found. Installing...")
        pip_dir.mkdir(parents=True, exist_ok=True)
        setup_cmd = [
            "singularity", "exec",
            "--containall",
            "--no-home",
            "--bind", f"{pip_dir.resolve()}:/pip",  # Read-write for installation. Made read only later in script.
            "--env", "TMPDIR=/pip",
            str(python_singularity_path),
            "bash", "-c",
            "pip install --no-cache-dir --target=/pip ag2[openai,rag] sentence-transformers llama-index-llms-openai-like",
        ]
        subprocess.run(setup_cmd)
        logging.info(f"Pip packages installed successfully.")
        
        # Now the code can be run with all pip installations already installed.
        instance_start_cmd = [
        "singularity", "exec",
        "--nv", # Enables GPU support.
        "--containall", # Full isolation from the host system.
        "--no-home",    # Full isolation from the home directory.
        "--network=bridge", # Setes a virtual network interface for the container to prevent any network attacks to host.
        "--bind", f"{pip_dir.resolve()}:/pip:ro", # Read only for security.
        "--bind", f"{RAG_scripts_dir.resolve()}:/RAG:ro", # Read only for security.
        "--bind", f"{temp_dir.resolve()}:/temp",
        "--bind", f"{DB_dir.resolve()}:/DB",
        "--bind", f"{Files_dir.resolve()}:/Files:ro",
        "--bind", f"{hf_cache_dir.resolve()}:/hf_cache",
        "--env","PYTHONPATH=/pip",
        #"--env","SINGULARITYENV_PYTHONPATH=/pip",
        "--env", "TMPDIR=/temp",
        "--env", "HF_HOME=/hf_cache",
        "--env", "TRANSFORMERS_CACHE=/hf_cache",
        "--env", "SENTENCE_TRANSFORMERS_HOME=/hf_cache",
        "--env", "LLAMA_INDEX_CACHE_DIR=/hf_cache",  # Add this line
        #"--env", "PYTHONUSERBASE=/pip",
        str(python_singularity_path),
        "python", "/RAG/RAG_Singularity_Script.py", f"'{query}'", f"'{ip_address}'",
        ]
        result = subprocess.run(
            instance_start_cmd,
            text=True)
        Response=result.stdout # This wll be returned so that the agent can see the response.

        if result.returncode != 0:
            raise ValueError(f"Failed to start Singularity instance: {result.stderr}")
        else:
            logging.info(f"Singularity instance started successfully.")

        # Delete temp files for security.
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logging.info(f"Deleted temp folder: {temp_dir}")
    else:   
        # Now the code can be run with all pip installations already installed.
        instance_start_cmd = [
        "singularity", "exec",
        "--nv", # Enables GPU support.
        "--containall", # Full isolation from the host system.
        "--no-home",    # Full isolation from the home directory.
        "--network=bridge", # Setes a virtual network interface for the container to prevent any network attacks to host.
        "--bind", f"{pip_dir.resolve()}:/pip:ro", # Read only for security.
        "--bind", f"{RAG_scripts_dir.resolve()}:/RAG:ro", # Read only for security.
        "--bind", f"{temp_dir.resolve()}:/temp",
        "--bind", f"{DB_dir.resolve()}:/DB",
        "--bind", f"{Files_dir.resolve()}:/Files:ro",
        "--bind", f"{hf_cache_dir.resolve()}:/hf_cache",
        "--env","PYTHONPATH=/pip",
        #"--env","SINGULARITYENV_PYTHONPATH=/pip",
        "--env", "TMPDIR=/temp",
        "--env", "HF_HOME=/hf_cache",
        "--env", "TRANSFORMERS_CACHE=/hf_cache",
        "--env", "LLAMA_INDEX_CACHE_DIR=/hf_cache",  # Add this line
        "--env", "SENTENCE_TRANSFORMERS_HOME=/hf_cache",
        #"--env", "PYTHONUSERBASE=/pip",
        str(python_singularity_path),
        "python", "/RAG/RAG_Singularity_Script.py", f"'{query}'", f"'{ip_address}'",
        ]
        result = subprocess.run(
            instance_start_cmd,
            text=True)
        Response=result.stdout # This wll be returned so that the agent can see the response.

        if result.returncode != 0:
            raise ValueError(f"Failed to start Singularity instance: {result.stderr}")
        else:
            logging.info(f"Singularity instance started successfully.")

        # Delete temp files for security.
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logging.info(f"Deleted temp folder: {temp_dir}")

    return ReplyResult(
        message=Response,
        context_variables=context_variables,
    )