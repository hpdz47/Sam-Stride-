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
from typing import Any, Dict, List, Optional, Annotated, Tuple, Union
from autogen import UpdateSystemMessage
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
import numpy as np
import pandas as pd
import json
import csv
import os
from autogen.tools.experimental import DeepResearchTool
from autogen.tools.experimental import ReliableTool

from vLLM_Configuration import VLLM_Config
from vLLM_Manager import LLM_Manager
from dotenv import load_dotenv
import subprocess
from pathlib import Path
import logging
import socket
import shutil
import time
from Data_Discovery_Conversation import Data_Discovery_Chat
from RAG_Tools import RAG_Tool
import copy
import pprint
import re
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from autogen import Agent
#===========================================================
load_dotenv()

# ------------------Structured Output-------------------
class Step(BaseModel):
    Statement: str = Field(..., description="Definition of what data analysis means withn the focus area and what the end result is tasked with finding.")
    Analysis_Suggestions: List[str] = Field(..., description="Suggest clearly and concisely the name of the data analysis that could be performed within the focus area.")
    Context: str = Field(..., description="Explain any important context linked to the datsets that is relevant to the focus area.")

class PlanResponse(BaseModel):
    Focus_Area: List[Step] = Field(..., description="A list of focus area statements. ")
#-----------------------

class Agent_Base():
    def __init__(self,name: str,llm_config: LLMConfig, system_message: str, Update_System_Message: Optional[str] = None, context_variables: Optional[ContextVariables] = None):
        self.name=name
        self.llm_config=llm_config # Composition Not Used here in case strucutred responses are needed for specific agents.
        self.system_message=system_message
        self.human_input_mode="NEVER" # Hard Coded as this is an AUTONMOUS system.
        if Update_System_Message:
            Updated_Message = [UpdateSystemMessage(Update_System_Message)]
        self._agent=ConversableAgent(
            name=self.name,
            llm_config=self.llm_config,
            system_message=self.system_message,
            human_input_mode=self.human_input_mode,
            update_agent_state_before_reply=Updated_Message if Update_System_Message else None,
            context_variables=context_variables

        )

    @property
    def agent(self) -> ConversableAgent:
        return self._agent # Getter for retrieving the agent instance.

class Focus_Area_Agent(Agent_Base):
    pass

class Focus_Chat():
    Type: ""
    def __init__(self, context_variables: ContextVariables, Analysis_Type: str, Max_Rounds: int):
        self.context_variables=context_variables
        self.Analysis_Type=Analysis_Type
        Focus_Chat.Type=Analysis_Type
        self.Max_Rounds=Max_Rounds
        LLM_Manager(LLM_Type="Reasoning").Manage_VLLM()

        if self.Analysis_Type=="Usability":
            self.Focus_Agent_System_Message="""
            Your role is to define a focus area for data analysis
            """
            self.Focus_Agent_Update_System_Message="""
            --------- ROLE ------------
            You are an expert scientist that provides a clear and concise statement that defines the focus area for data analysis.
            Your statement will be used by a planner agent to help them decide what data analysis to perform.
            You must define what the term data analysis means in the context of the focus area. This is essential to allow other agents to understand the scope and objectives of the analysis.
            For the focus area you are given, you must explain the purpose of collecting and analysing the data. What is the main result that the data analysis is tasked with finding?
            You supplement your statement with examples of suitable data analysis that could be performed within the focus area.
            You must make sure not to encourage cross-file analysis in your focus area statement as this analysis is typically concerned with usability within individual data files.
            Your statement must be a single informative paragraph.

            --------- Focus Area ------------
            The context for the focus area is in assessing the usability and quality of High Performance Liquid Chromatography (HPLC) and Mass Spectrometry (MS) data.
            The data comes from a Biopharmaceutical manufacturing process and is crucial for ensuring product quality and safety.
            The data files available are related to HPLC and MS analysis of samples taken during the manufacturing process.
            Your focus area statement must consider the importance of data quality, reliability, and relevance to the manufacturing process.

            --------- Task ------------
            - You must call the Focus_Area_Report function to provde your focus area statement.
            - You must ensure that your focus area statement is relevant to the data files available. You have access to the metadata and descriptions of the data files.
            You have access to reports generated from Exploratory Data Analysis (EDA) that focused on univariate and multivarate EDA.
            - You must provide some examples of appropriate data analysis as an example of the expectations for the focus area. Although, you must not
            constrain the planner to only use these examples.
            ----------------------------
             ------- Output Format ------------
            Your output must include the following subsections:
            - Statement: You must provide a clear definition of what data analysis means within the context of the foucs area, given the data file reports and metadata available. You must explain what the end goal of the data analyss is. For example (illustrative not exhaustive): The quality of both HPLC and MS data must be assessed to ensure sensor accuracy and reliablity. Sensor drift and excessive noise should be detected and reported so that corrective actions can be taken to ensure data integrity throughout the biopharmaceutical manufacturing process.
            - Suggested Data Analysis: A list of suitable data analysis techniques that could be performed within the focus area. You should aim for up to 10 suggestions (short names not long explanations of analysis). For example (illustrative only), "SNR Calculation", "Sensor Drift Analysis", etc.
            - Context: Any important context linked to the datasets that is relevant to the focus. For example (illustrative only), "The dataset contains data that is meant to contain sudden peaks and estimation of outliers or noise should take into account that this is expected."

            The output should be of the following form:
            **Statement**
            <Your Statement>
            **Suggested Data Analysis**
            <List of Analysis Suggestions>
            **Context**
            <Your Context>
            ----------------------------------
            ------ Context -------------
            ** Current Focus Area Statement **
            {Focus_Area_Usability}
            *******************
            ***** Information about the dataset (Exploratory Data Analysis reports) *****
            {Univariate_EDA_Report} 
            {Multivariate_EDA_Report}
            ****** Variables Available (Per File) ******
            {metadata}
            ********************
            """
        elif self.Analysis_Type=="HPLC":
            self.Focus_Agent_System_Message="""
            Your role is to define a focus area for data analysis
            """
            self.Focus_Agent_Update_System_Message="""
            --------- ROLE ------------
            You are an expert scientist that provides a clear and concise statement that defines the focus area for data analysis.
            Your statement will be used by a planner agent to help them decide what data analysis to perform.
            You must define what the term data analysis means in the context of the focus area. This is essential to allow other agents to understand the scope and objectives of the analysis.
            For the focus area you are given, you must explain the purpose of collecting and analysing the data. What is the main result that the data analysis is tasked with finding?
            You supplement your statement with examples of suitable data analysis that could be performed within the focus area.
            You must be careful not to encourage cross file analysis in your focus area statement as this analysis is specifically concerned with HPLC data only.
            Your statement must be a single informative paragraph.

            --------- Focus Area ------------
            The context for the focus area is in analysing High Performance Liquid Chromatography (HPLC) data.
            The data comes from a Biopharmaceutical manufacturing process and is crucial for ensuring product quality and safety.
            The data files available are related to HPLC analysis of samples taken during the manufacturing process. Your suggestions are only allowed to relate to any HPLC data. MS data is not to be considered in this focus area.
            Your focus area statement must consider the importance of data quality, reliability, and relevance to the manufacturing process.

            --------- Task ------------
            - You must call the Focus_Area_Report function to provde your focus area statement.
            - You must ensure that your focus area statement is relevant to the data files available. You have access to the metadata and descriptions of the data files.
            You have access to reports generated from Exploratory Data Analysis (EDA) that focused on univariate and multivarate EDA.
            - You must provide some examples of appropriate data analysis as an example of the expectations for the focus area. Although, you must not
            constrain the planner to only use these examples.
            ----------------------------
            ------- Output Format ------------
            Your output must include the following subsections:
            - Statement: A concise statement defining what HPLC analysis means and what the main objectives are. For example (illustrative not exhaustive): HPLC data analysis seeks to characterise chromatograms to identify compounds present and their concentrations. There are many different techniques of varying complexity and efficacy that can be used to ensure accurate compound identification and quantification.
            - Suggested Data Analysis: A list of suitable data analysis techniques that could be performed within the focus area. You should aim for up to 10 suggestions (short names not long explanations of analysis). For example (illustrative only), "Regression Analysis", "Clustering Analysis", "Peak Deconvolution", etc.
            - Context: Any important context linked to the datasets that is relevant to the focus. For example (illustrative only), "The HPLC data contains multple runs and so analysis should be performed on each run or data should be plotted by run to help identify what compounds are present."

            The output should be of the following form:
            **Statement**
            <Your Statement>
            **Suggested Data Analysis**
            <List of Analysis Suggestions>
            **Context**
            <Your Context>
            ----------------------------------
            ------ Context -------------
            ** Current Focus Area Statement **
            {Focus_Area_HPLC}
            *******************
            ***** Information about the dataset (Exploratory Data Analysis reports) *****
            {Univariate_EDA_Report} 
            {Multivariate_EDA_Report}
            ****** Variables Available (Per File) ******
            {metadata}
            ********************
            """
        elif self.Analysis_Type=="MS":
            self.Focus_Agent_System_Message="""
            Your role is to define a focus area for data analysis
            """
            self.Focus_Agent_Update_System_Message="""
            --------- ROLE ------------
            You are an expert scientist that provides a clear and concise statement that defines the focus area for data analysis.
            Your statement will be used by a planner agent to help them decide what data analysis to perform.
            You must define what the term data analysis means in the context of the focus area. This is essential to allow other agents to understand the scope and objectives of the analysis.
            For the focus area you are given, you must explain the purpose of collecting and analysing the data. What is the main result that the data analysis is tasked with finding?
            You supplement your statement with examples of suitable data analysis that could be performed within the focus area.
            You must be careful not to encourage cross file analysis in your focus area statement as this analysis is specifically concerned with MS data only.
            Your statement must be a single informative paragraph.
            --------- Focus Area ------------
            The context for the focus area is in analysing Mass Spectrometry (MS) data.
            The data comes from a Biopharmaceutical manufacturing process and is crucial for ensuring product quality and safety.
            The data files available are related to MS analysis of samples taken during the manufacturing process. You are only allowed to consider MS data in this focus area. HPLC data is not to be considered.
            Your focus area statement must consider the importance of data quality, reliability, and relevance to the manufacturing process.

            --------- Task ------------
            - You must call the Focus_Area_Report function to provde your focus area statement.
            - You must ensure that your focus area statement is relevant to the data files available. You have access to the metadata and descriptions of the data files.
            You have access to reports generated from Exploratory Data Analysis (EDA) that focused on univariate and multivarate EDA.
            - You must provide some examples of appropriate data analysis as an example of the expectations for the focus area. Although, you must not
            constrain the planner to only use these examples.
            ----------------------------
             ------- Output Format ------------
            Your output must include the following subsections:
            - Statement: A concise statement defining what data analysis means in the context of the MS data availbale, and highlight the key results. For example (illustrative not exhaustive): MS data analysis aims to find fragmentation patterns that can be used to identify compunds present in samples and their abundance. Accurate compound identification is crucial for ensuring product quality in biopharmaceutical manufacturing, and various analysis techniques can be employed to achieve this.
            - Suggested Data Analysis: A list of suitable data analysis techniques that could be performed within the focus area. You should aim for up to 10 suggestions (short names not long explanations of analysis). For example (illustrative only), "Regression Analysis", "Clustering Analysis", etc.
            - Context: Any important context linked to the datasets that is relevant to the focus. For example (illustrative only), "The MS data contains multiple runs and so analysis should be performed on each run to identify compounds of interest."

            The output should be of the following form:
            **Statement**
            <Your Statement>
            **Suggested Data Analysis**
            <List of Analysis Suggestions>
            **Context**
            <Your Context>
            ----------------------------------
            ------ Context -------------
            ** Current Focus Area Statement **
            {Focus_Area_MS}
            *******************
            ***** Information about the dataset (Exploratory Data Analysis reports) *****
            {Univariate_EDA_Report} 
            {Multivariate_EDA_Report}
            ****** Variables Available (Per File) ******
            {metadata}
            ********************
            """
        else:
            raise ValueError(f"Invalid Analysis_Type: {self.Analysis_Type}. Must be one of 'Usability', 'HPLC', 'MS'.")
        
        self.FA_LLM_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=True,LLM_Type="Reasoning").build_config()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.2,enable_thinking=False,LLM_Type="Reasoning").build_config()
        self.Focus_Agent=Focus_Area_Agent(
            name="Focus_Agent",
            llm_config=self.FA_LLM_Config,
            system_message=self.Focus_Agent_System_Message,
            Update_System_Message=self.Focus_Agent_Update_System_Message)
        
        # Handoffs ---------------
        self.Focus_Agent.agent.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(
                    expression=ContextExpression("${FA_Updated} == True")
                )
            )
        )

        self.Focus_Agent.agent.handoffs.set_after_work(AgentTarget(self.Focus_Agent.agent))

        # Functions-------
        register_function(
            Focus_Area_Report,
            caller=self.Focus_Agent.agent,
            executor=self.Focus_Agent.agent,
            name="Focus_Area_Report",
            description="Generates a concise focus area statement for data analysis based on the provided context."
            )
        
    def run_Conversation(self):
        # Reset Context Variables:
        self.context_variables["FA_Updated"]=False
        agents=[self.Focus_Agent.agent]
        pattern=DefaultPattern(
        initial_agent=agents[0],
        agents=agents,
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,)

        result, ctx, _ = initiate_group_chat(
        pattern=pattern,
        messages="Using the information available, please define a clear and concise focus area statement for data analysis.",
        max_rounds=self.Max_Rounds)
    
        return result, ctx

def Focus_Area_Report(FA: Annotated[str,"Suggest the focus area statement, the relevant analysis types possible and the context linked to the datasets available."], context_variables: ContextVariables) -> ReplyResult:
    """
    Generates a concise focus area statement for data analysis based on the provided context.

    Parameters:
    - FA (str): The focus area statement provided by the Focus Area Agent.
    - context_variables (ContextVariables): The context variables available to the agent.

    Returns:
    - ReplyResult: The structured response containing the focus area details.
    """
    context_variables[f"Focus_Area_{Focus_Chat.Type}"]=FA

    context_variables["FA_Updated"]=True
    return ReplyResult(
        message=f"Focus Area Statement received and recorded.",
        context_variables=context_variables,
    )

def main():
    context_variables=ContextVariables({
        "metadata": "'metadata chromatography_combined.csv' \n Variables: ['Unnamed: 0', 'start_well', 'end_well', 'Sample_Code', 'process_part', 'run', 'chromatography_stage', 'COLUMN_IN_USE', 'Resin_type', 'column', 'Column_Volume_(ml)', 'Load_volume_(mL)', 'Load_pH', 'Unicorn_server_result_file', 'Exported_run_data_file', 'Unicorn_Method_Name', 'Chromatography_Plate_ID', 'Plate_start', 'Plate_end', 'Chromatography_Plate_ID_2', 'Plate2_start', 'Plate2_end', 'run_no', 'Run_Name', 'Excel_Filename', 'Shortened_Code_for_Bioaccord_Samples', 'Run_variables_file', 'Operator_name_(Full)', 'Date', 'Upstream_sample', 'Aliquot_code', 'Time_samples_removed_from_fridge', 'Time_sample_returned_to_fridge', 'IDBS_experiment_number', 'Sample_start_volume_(mL)', 'Sample_volume_after_dilution_with_HPW_(mL)', 'Sample_final_volume_after_pH_titration_(mL)', 'Titration_agent', 'Start_pH', 'Final_pH', 'Total_protein_concentration_(bradford)_(mg/mL)', 'Conducivity_(mS)', 'Concentration_(mg/mL)', 'Filter_type_used', 'Load_sample_start_ID', 'Load_sample_start_IDBS_name', 'Load_sample_end_IDBS_name', 'EQ_and_wash_buffer', 'EQ_and_Wash_Buffer_Inlet', 'Elution_buffer', 'Elution_Buffer_Inlet', 'Sanitisation_buffer', 'Other_buffer', 'All_columns_loaded_from_same_material', 'Notes', 'fraction_volume', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise'] \n -----'MS_combined.csv' ------ \n ['Unnamed: 0', 'Unique_Peak_ID', 'Unique_MS_Sample_ID', 'Sample_Code', 'run_code', 'process_part', 'run', 'MS_method', 'column', 'chromatography_stage', 'start_well', 'end_well', 'Replicate', 'Type', 'Molecule ID', 'Component', 'Observed_TIC_RT_(mins)', 'Observed_UV_RT_(mins)', 'Observed RT delta (mins)', 'Response', '%_of_response', 'Observed_neutral_mass_(Da)', 'Observed_m/z', 'Spectrum_type', 'Expected_mass_(Da)', 'Mass_error_(ppm)', 'Alternative_assignments', 'fraction_volume', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise']\n ------'chromatography_combined.csv'------ \n ['Unnamed: 0', 'run_no', 'run', 'Fraction_unique_ID', 'column', 'volume_ml', 'UV_1_280_ml', 'UV_1_280_mAU', 'Cond_ml', 'Cond_mS_cm', 'Conc_B_ml', 'Conc_B_%', 'Injection_ml', 'Injection_Injection', 'Run_Log_ml', 'Run_Log_Logbook', 'Fraction_ml', 'Fraction_Fraction', 'UV_1_280_CUT_TEMP_100_BASEM_ml', 'UV_1_280_CUT_TEMP_100_BASEM_mAU', 'UV_2_260_ml', 'UV_2_260_mAU', 'pH_ml', 'pH_pH', 'DeltaC_pressure_ml', 'DeltaC_pressure_MPa', 'System_flow_ml', 'System_flow_ml_min', 'Sample_flow_ml', 'Sample_flow_ml_min', 'Sample_Code', 'chromatography_stage', 'chromatography_stage_order', 'Fraction_number', 'fraction_volume_ml_min_precise', 'fraction_volume_ml_max_precise', 'fraction_volume']",
        "Univariate_EDA_Report": "### Dataset Interpretation Report\n\nThis report provides a detailed interpretation of the three files in the dataset: `metadata chromatography_combined.csv`, `MS_combined.csv`, and `chromatography_combined.csv`. Each file is analyzed independently to understand its structure, content, and relevance for downstream analysis.\n\n---\n\n### 1. `metadata chromatography_combined.csv`\n\n**File Structure and Type:**\n- This is a CSV file with a data shape of 354 rows and 58 columns.\n- The file contains metadata related to chromatography runs, including experimental parameters, sample details, buffer information, and operational notes.\n\n**Key Observations:**\n- **Data Types:** The file contains a mix of `int64`, `float64`, and `object` (string) data types. Notably, several columns (e.g., `start_well`, `end_well`, `Sample_Code`) are of type `object`, indicating categorical or textual data.\n- **Missing Values:** Many columns (e.g., `Time_samples_removed_from_fridge`, `Time_sample_returned_to_fridge`, `Notes`) have no observed values (all NaNs), suggesting they may be unused or incomplete.\n- **Dense Data:** The dataset is dense, with most numerical columns having meaningful values. The `Unnamed: 0` column appears to be an index, with values ranging from 0 to 49.\n- **Categorical Variables:** Several columns are highly categorical:\n  - `Resin_type` (4 unique values)\n  - `chromatography_stage` (16 unique values)\n  - `Run_Name`, `Excel_Filename`, `Operator_name_(Full)` (11, 11, and 1 unique values respectively), indicating a small number of experimental runs.\n- **Numerical Variables:** Key experimental parameters include:\n  - `Load_volume_(mL)` (mean ~112 mL, range 35–188 mL)\n  - `Sample_start_volume_(mL)` (mean ~166 mL, range 90–270 mL)\n  - `Total_protein_concentration_(bradford)_(mg/mL)` (mean ~0.16 mg/mL, range 0.10–0.44 mg/mL)\n  - `Start_pH` and `Final_pH` (mean ~7.1 and ~6.2, respectively)\n- **Grouping:** The data is grouped by `run`, `run_no`, and `Sample_Code`, suggesting that each row corresponds to a unique experimental run or sample.\n- **Relevance:** This file serves as a comprehensive metadata source for chromatography experiments, linking sample, process, and operational details. It is essential for contextualizing the other two datasets.\n\n**Conclusion:** This file is a dense, structured metadata file critical for understanding the experimental setup. It should be used to enrich and validate the other datasets.\n\n---\n\n### 2. `MS_combined.csv`\n\n**File Structure and Type:**\n- This is a large CSV file (620.35 MB) with 1,912,029 rows and 30 columns.\n- It contains mass spectrometry (MS) data, likely from peptide or protein identification and quantification.\n\n**Key Observations:**\n- **Data Types:** Mixed types, with `int64`, `float64`, and `object` columns. The `Unnamed: 0` column appears to be an index.\n- **Missing Values:** Several columns (e.g., `Type`, `Molecule ID`, `Component`, `Expected_mass_(Da)`, `Mass_error_(ppm)`, `Alternative_assignments`) have no observed values (all NaNs), indicating they may be unused or incomplete.\n- **Sparse Data:** Despite its large size, the dataset is sparse in certain key columns, particularly those related to molecular identification.\n- **Categorical Variables:**\n  - `Unique_MS_Sample_ID` (283 unique values)\n  - `Sample_Code` (274 unique values)\n  - `chromatography_stage` (19 unique values)\n  - `Spectrum_type` (3 unique values)\n- **Numerical Variables:** Key MS features include:\n  - `Observed_TIC_RT_(mins)` (mean ~2.33 mins, range ~0.62–3.03 mins)\n  - `Observed_UV_RT_(mins)` (mean ~1.98 mins, range ~0.74–2.72 mins)\n  - `Response` (mean ~9,920, range ~4.86–15.25M)\n  - `Observed_neutral_mass_(Da)` (mean ~162.5 kDa, range ~400–314 kDa)\n  - `Observed_m/z` (mean ~733 Da, range ~505–1,263 Da)\n- **Grouping:** Data is grouped by `Unique_MS_Sample_ID`, `Sample_Code`, and `run_code`, suggesting a hierarchical structure where multiple MS spectra are associated with a single sample.\n- **Relevance:** This file contains high-resolution MS data, likely from LC-MS/MS experiments. It is essential for downstream analysis such as peptide identification, quantification, and post-translational modification (PTM) analysis.\n\n**Conclusion:** This file is a large, sparse dataset containing detailed MS data. It is highly relevant for proteomics analysis but requires careful handling due to missing values in key identification columns.\n\n---\n\n### 3. `chromatography_combined.csv`\n\n**File Structure and Type:**\n- This is a large CSV file (257.96 MB) with 1,084,069 rows and 37 columns.\n- It contains chromatographic data, including UV, conductivity, pH, flow, and volume measurements across fractions.\n\n**Key Observations:**\n- **Data Types:** Mixed types, with `int64`, `float64`, and `object` columns. The `Unnamed: 0` column is likely an index.\n- **Missing Values:** Some columns (e.g., `Injection_Injection`, `Fraction_Fraction`) have no observed values (all NaNs), suggesting they may be unused.\n- **Dense Data:** The dataset is dense, with most numerical columns having meaningful values. The `Fraction_number` and `Sample_Code` columns suggest a hierarchical structure where multiple fractions are generated per sample.\n- **Categorical Variables:**\n  - `column` (4 unique values)\n  - `chromatography_stage` (16 unique values)\n  - `Run_Log_Logbook` (17 unique values)\n- **Numerical Variables:** Key chromatographic parameters include:\n  - `volume_ml` (mean ~126 mL, range ~-33.8–335.8 mL)\n  - `UV_1_280_ml` and `UV_1_280_mAU` (mean ~126 mL and ~250 mAU, respectively)\n  - `Cond_ml` and `Cond_mS_cm` (mean ~126 mL and ~15.8 mS/cm)\n  - `pH_ml` and `pH_pH` (mean ~132.5 mL and ~6.28 pH)\n  - `fraction_volume` (mean ~20.9 mL, range ~0.008–66.0 mL)\n- **Grouping:** Data is grouped by `Sample_Code`, `run_no`, and `Fraction_number`, indicating that each row corresponds to a fraction collected during a chromatography run.\n- **Relevance:** This file contains high-resolution chromatographic profiles, essential for understanding elution patterns, peak detection, and fraction collection. It is critical for integrating with MS data to link protein identity to chromatographic behavior.\n\n**Conclusion:** This file is a dense, structured chromatographic dataset that provides detailed fraction-level data. It is indispensable for correlating MS results with chromatographic elution profiles.\n\n---\n\n### Overall Summary\n\n- The dataset consists of three interrelated files:\n  1. `metadata chromatography_combined.csv`: Metadata for experimental runs.\n  2. `chromatography_combined.csv`: High-resolution chromatographic profiles (fraction-level).\n  3. `MS_combined.csv`: High-resolution MS data (peptide/protein-level).\n\n- **Integration Potential:** The three files can be integrated using common keys such as `Sample_Code`, `run_no`, and `chromatography_stage`. This integration enables a comprehensive analysis of protein behavior across chromatography and MS platforms.\n\n- **Gaps and Limitations:**\n  - Several columns in `MS_combined.csv` are entirely missing (all NaNs), which may limit downstream identification.\n  - Some columns in all files (e.g., `Notes`, `Other_buffer`) are unused or incomplete.\n  - Further validation is required to confirm the consistency of `Sample_Code` and `run_no` across files.\n\n- **Recommendations for Further Analysis:**\n  - Perform data merging using `Sample_Code` and `run_no` to align chromatography and MS data.\n  - Investigate the reasons for missing values in `MS_combined.csv`.\n  - Validate the integrity of `fraction_volume` and related columns across files.\n  - Conduct a time-series analysis of chromatographic profiles to identify elution patterns.\n\nThis dataset is well-structured for integrative proteomics analysis, provided that missing data issues are addressed.",
        "Multivariate_EDA_Report": "",
        "Focus_Area_Usability": "",
        "Focus_Area_HPLC": "",
        "Focus_Area_MS": "",
        "FA_Updated": False,
    })

    Focus_Area_Chat1=Focus_Chat(context_variables=context_variables, Analysis_Type="Usability", Max_Rounds=5)
    Focus_Area_Chat1.run_Conversation()
    Focus_Area_Chat2=Focus_Chat(context_variables=context_variables, Analysis_Type="HPLC", Max_Rounds=5)
    Focus_Area_Chat2.run_Conversation()
    Focus_Area_Chat3=Focus_Chat(context_variables=context_variables, Analysis_Type="MS", Max_Rounds=5)
    Focus_Area_Chat3.run_Conversation()

if __name__=="__main__":
    main()