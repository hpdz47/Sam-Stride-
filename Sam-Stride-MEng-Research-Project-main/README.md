# Specialised AI Assistants for the Interpretation of Biopharmaceutical Data

![alt text](Final_Diagram(Github).png)

## Purpose
This repository provides code that implements an LLM-based Multi-Agent System (MAS) for automating data analysis. The purpose is to allow the reasoning behind current AI LLM's to have more functionality than they can have independently. By using agents and providing them with tools, tasks such as Data Analysis can be carried out without needing domain-specific knowledge. This system requires an input dataset (.csv) and a brief user requirement to allow it to perform data discovery, planning, coding and data analysis execution resulting in a PDF report (via Latex). This system was initially built for deployment in a BioPharmaceutical manufacturng environment but t is a general data analysis system that can be used in any domain.

## Setup
- This code code is desgned to run in a single container instance (Docker, Singularity, Apptainer). This allows the code to be run on different computing platforms and provides a layer of isolation due to the execution of LLM-generated code.

- AG2 requires the code to appear as though it is coming from an OpenAI API Endpoint to allow for tool usage. The system sets up a vLLM server (127.0.0.0 --port 8002).

- Main tools used are vLLM (Which installs models form HuggingFace) and AG2. To allow for tool usage, the correct tool parser must be used (when setting up vLLM server) and there must be a JSON file that gives the LLM clear instructions. Some HuggingFace models have this as part of tokenizer_config.JSON, but other models may include a specialised .jinja2 file for this purpose.

## Code Layout
- Setup files: batch_job.sh (Specific to HPC cluster or computing platform used), .env (To avoid hardcoded API Keys). The .env file contains server API tokens, huggingface tokens (as required) and LLM model name to be used. The LLM model can be switched easily with a valid HuggingFace model to allow a different LLM to power the MAS.
- Agents Folder: This contains directories for each agent in the system for ease of modification of settings (via config yaml file) and of system message (via system message yaml file). Each agent can be constructed and returned via the Agent Factory function, separating agent setup from the orchestration layer.
- Config: All LLM configuration is handled by the VLLM Config file and the VLLM Manager, allowing LLM model types to be switched as required.
- Chatrooms: This contains all orchestration code for creating the Multi-Agent System, allowing the responses of individual agents to be saved to context variables and passed to the relevant agents in the pipeline.
- Tools: This contains code for hooks and function calls. The hooks can capture the agent response (free-form or pydantic) and save to context variables to allow other agents to see the response.
- Utils: This contains all utility code files such as pydantic schema, Secure Local Command Line Executor (for code files) and the git repo management/memory systems that are used in the Planning and Coding stages.
- Outputs: The outputs folder is the main workspace (read-write) for the system. All other directories are bind mounted as read-only to ensure minimal LLM tampering. The outputs folder contains the fles needed by the MAS during the workflow as well as the end results folder.
  - Results: The results folder uses Memory_Ablation_XY. X = Number of Planning self-refinement loops and Y = Number of Coding self-refinement loops. NN = No Memory on both stages, MN = Memory on Planning, No Memory on Coding, MM = Memory on both stages and NM = No Memory on Plannning and Memory on Coding. These are all adjustable via the BioAgent interface.
- Inputs: The Inputs folder (named exactly as shown with capital 'I') can be used to upload any .csv file that teh system should analyse. It is advisable to only use 1 file at a time to prevent system confusion.
- BioAgent: BioAgent is the main entrypoint to the system. There are many different configuration options that can be selected. The key options are to toggle memory On/Off, adjusting the number of self-refinement loops, adjusting the maximum number of plan steps and adjusting the debug attempts during the coding stage.


