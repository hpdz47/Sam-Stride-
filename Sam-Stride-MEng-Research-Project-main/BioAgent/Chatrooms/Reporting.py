# ---------------------Imports ------------------------------------------------------------------------
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
import numpy as np
import pandas as pd
import json
import csv
import os
from Config.vLLM_Configuration import VLLM_Config
from Config.vLLM_Manager import LLM_Manager
from dotenv import load_dotenv
from Utils.Secure_LocalCommandLineExecutor import SecureLocalCommandLineExecutor
from pathlib import Path
from autogen import Agent
import copy
import pprint
import re
import shutil
import subprocess
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from Agents.AgentFactory import Agent_Factory
load_dotenv()

from Utils.Git_Manager import GitManager
from Utils.Git_Memory_Handling import GitMemory
from Utils.Code_Repo_Management import create_code_git_managers

#Import Functions/Hooks:
from Tools.Reporting_Phase import VL_Hook, Markdown_Hook, Report_Hook

#------------------------------------------------------------------------------------------------

class Visual_Reviewer:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int):
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        self.VL_Review = Agent_Factory(agent_name="VL_Review", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        #------ Functions:
        self.VL_Review.register_hook("process_message_before_send", VL_Hook)

        #----- Handoffs:
        self.VL_Review.handoffs.set_after_work(TerminateTarget())
    
    def run_Conversation(self):
        pattern=DefaultPattern(
        initial_agent=self.VL_Review,
        agents=[self.VL_Review],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        messages= f"""<img {self.context_variables["Current_Image_Name"]}>"""
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages=messages, #This code block format is requred for the SingularityCommandLineExecutor to work properly.
            max_rounds=self.Max_Rounds,
        )
        return result, ctx

class MD_Reviewer:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int):
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        self.Markdown_Review = Agent_Factory(agent_name="Markdown_Review", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()

        #------ Functions:
        self.Markdown_Review.register_hook("process_message_before_send", Markdown_Hook)

        #----- Handoffs:
        self.Markdown_Review.handoffs.set_after_work(TerminateTarget())
    
    def run_Conversation(self):
        pattern=DefaultPattern(
        initial_agent=self.Markdown_Review,
        agents=[self.Markdown_Review],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        messages= f"""Analyse the markdown file and interpret the data analysis."""
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages=messages, #This code block format is requred for the SingularityCommandLineExecutor to work properly.
            max_rounds=self.Max_Rounds,
        )
        return result, ctx

class Report_Writing:
    def __init__(self, context_variables: ContextVariables, Max_Rounds: int):
        self.context_variables = context_variables
        self.Max_Rounds = Max_Rounds

        self.Report_Writer = Agent_Factory(agent_name="Report_Writer", context_variables=self.context_variables).BuildAgent()
        self.Chat_Config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.3,enable_thinking=False,LLM_Type="Reasoning").build_config()
        #------ Functions:
        self.Report_Writer.register_hook("process_message_before_send", Report_Hook)

        #----- Handoffs:
        self.Report_Writer.handoffs.set_after_work(TerminateTarget())
    
    def run_Conversation(self):
        pattern=DefaultPattern(
        initial_agent=self.Report_Writer,
        agents=[self.Report_Writer],
        group_manager_args={"llm_config": self.Chat_Config},
        context_variables=self.context_variables,
        )
        messages= f"""Use the image analysis reports and the markdown analysis reports to produce a final report."""
        result, ctx, _ = initiate_group_chat(
            pattern=pattern,
            messages=messages, #This code block format is requred for the SingularityCommandLineExecutor to work properly.
            max_rounds=self.Max_Rounds,
        )
        return result, ctx


class Final_Report:
    def __init__(self, context_variables: ContextVariables, Max_Rounds:int, Max_Images: int, Max_Markdown: int, Max_MD_Lines: int):
        self.context_variables = context_variables
        self.LLM_Name = os.getenv("LLM_Model_Reasoning")
        self.LLM_Name = self.LLM_Name.replace('/','_') # This ensures no / in LLM name as it is confusing for file naming.
        self.Max_Rounds = Max_Rounds

        #--- Enforcing Limitations (Backup)
        self.Max_Images = Max_Images # Maximum number of images that are allowed to be analysed due to memory/time constraints.
        self.Max_Markdown = Max_Markdown # Maximum number of markdown files that are allowed to be analysed due to memory/time constraints.
        self.Max_MD_Lines = Max_MD_Lines # Maximum number of lines per markdown file that can be accessed due to memory constraints.
        #-------------------------------------

        self.CODE_DIR = Path("/workspace/Code")
        self.ANALYSIS_DIR = Path("/workspace/Analysis")
        if self.ANALYSIS_DIR.exists():
            shutil.rmtree(self.ANALYSIS_DIR)
            self.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        else:
            self.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

        if self.context_variables["Short_Term_Memory"]:
            self.Planning_State = "M" # Stands for Memory On
        else:
            self.Planning_State = "N" # Stands for No Memory
        if self.context_variables["Code_Short_Term_Memory"]:
            self.Coding_State = "M"
        else:
            self.Coding_State = "N"
            
        #------ Final Results Directory:
        self.LLM_Name = os.getenv("LLM_Model_Reasoning") # This extracts the LLM name from teh .env file to enable scoring assocated with that LLM type.
        self.LLM_Name = self.LLM_Name.replace('/','_') # This ensures no / in LLM name as it is confusing for file naming
        self.REPORT_DIR = Path(f"""/workspace/Results/{self.LLM_Name}/Memory_Ablation_{self.context_variables["Max_Plan_Reviews"]}_{self.context_variables["Max_Review_Loops"]}/{self.Planning_State}{self.Coding_State}""")

        if not self.REPORT_DIR.exists():
            self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
            

        print("\n\n====================\nCreating Final Report \n====================")

        # Clean up any output files from debug loop runs that failed.
        for item in Path("/workspace").iterdir():
            if item.is_file() and item.suffix.lower() in (".png", ".jpg", ".jpeg", ".md"):
                item.unlink()

        # Running the code from the Code Repo to store the results for reviewing:
        for file in sorted(self.CODE_DIR.iterdir()):
            if file.suffix != ".py":
                continue
            try:
                result = subprocess.run(
                    ["python3", str(file)],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode != 0:
                    print(f"\n\nCode {file.name} failed with exit code {result.returncode}")
                    print(f"STDERR: {result.stderr[:500]}")
                else:
                    print(f"\nCode {file.name} executed successfully.")
            except subprocess.TimeoutExpired:
                print(f"\n\nCode {file.name} timed out.")
            except Exception as e:
                print(f"\n\nUnexpected error when running {file.name}: {e}")

        # Move fresh outputs into Analysis directory
        for item in Path("/workspace").iterdir():
            if item.is_file() and item.suffix.lower() in (".png", ".jpg", ".jpeg", ".md"):
                shutil.move(str(item), self.ANALYSIS_DIR / item.name)
        #--------------------------------------------------------------------------

        #--- Resetting Context Variables as required:
        self.context_variables["Image_Analysis"] = []
        self.context_variables["Current_Image_Name"] = ""
        self.context_variables["Markdown_Analysis"] = []
        self.context_variables["Markdown_File"] = ""
    
    def create_pdf(self, report, analysis_dir, output_tex="report.tex", output_pdf="report.pdf", allowed_images=None):
        """
        report: ReportResponse object
        analysis_dir: Path to directory containing images
        allowed_images: set/list of valid image filenames
        """
        
        analysis_dir = Path(analysis_dir)
        allowed_images = set(allowed_images) if allowed_images else set()

        def escape_latex(text):
            replacements = {
                "&": "\\&",
                "%": "\\%",
                "$": "\\$",
                "#": "\\#",
                "_": "\\_",
            }
            for k, v in replacements.items():
                text = text.replace(k, v)
            return text

        latex = []

        # --- Header ---
        latex.append(r"""
        \documentclass{article}
        \usepackage{graphicx}
        \usepackage{float}
        \usepackage{geometry}
        \usepackage{listings}
        \usepackage{xcolor}

        \geometry{margin=1in}

        \lstdefinestyle{mypython}{
            language=Python,
            basicstyle=\ttfamily\small,
            keywordstyle=\color{blue},
            stringstyle=\color{red},
            commentstyle=\color{gray},
            breaklines=true,
            frame=single,
            rulecolor=\color{black},
            captionpos=b
        }

        \begin{document}
        """)

        # --- Main Sections ---
        for section in report.Section:
            latex.append(f"\\section{{{escape_latex(section.Title)}}}")
            latex.append(escape_latex(section.Text))

            # Only allow figures in Results and Discussion
            if section.Title == "Results and Discussion":
                for fig in section.Figures:
                    if fig.Path not in allowed_images:
                        continue

                    img_path = analysis_dir / fig.Path
                    if not img_path.exists():
                        continue

                    latex.append(r"\begin{figure}[H]")
                    latex.append(r"\centering")
                    latex.append(f"\\includegraphics[width=0.8\\linewidth]{{{img_path.as_posix()}}}")
                    latex.append(f"\\caption{{{escape_latex(fig.Caption)}}}")
                    latex.append(r"\end{figure}")

        # --- Appendix (Image Files) ---
        latex.append(r"\appendix")
        latex.append(r"\section{Appendix}")
        latex.append(r"\subsection{A: Data Analysis Output Figures}")

        for fig_name in allowed_images:
            img_path = analysis_dir / fig_name
            if not img_path.exists():
                continue

            latex.append(r"\begin{figure}[H]")
            latex.append(r"\centering")
            latex.append(f"\\includegraphics[width=0.8\\linewidth]{{{img_path.as_posix()}}}")
            latex.append(f"\\caption{{{escape_latex(fig_name)}}}")
            latex.append(r"\end{figure}")
        
        # --- Appendix (Code Files) ---
        latex.append(r"\subsection{B: Code Files}")

        for code_file in sorted(self.CODE_DIR.iterdir()):
            if code_file.suffix != ".py":
                continue

            code_name = code_file.stem  # e.g. Code1.py -> Code1
            code_path = code_file.as_posix()

            latex.append(
                f"\\lstinputlisting[style=mypython, caption={{{escape_latex(code_name)}}}]{{{code_path}}}"
            )

        # --- End Document ---
        latex.append(r"\end{document}")

        # Write .tex
        tex_path = Path(output_tex)
        tex_path.write_text("\n".join(latex), encoding="utf-8")
        
        # Compile PDF
        try:
            for _ in range(2):
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", tex_path.name],
                    cwd=str(tex_path.parent),
                    capture_output=True,
                    text=True,
                    errors="replace"
                )
            if result.returncode == 0:
                print("PDF compiled successfully.")
            else:
                print(f"pdflatex exited with code {result.returncode}")
                print(f"STDERR: {result.stderr[:500]}")
        except FileNotFoundError:
            print("pdflatex not found. The .tex file has been saved but PDF compilation was skipped.")

    def generate_report(self):
        # VL Review:
        n=1
        for file in self.ANALYSIS_DIR.iterdir():
            if n>= self.Max_Images:
                break
            if file.suffix.lower() == ".png":
                self.context_variables["Current_Image_Name"] = file.name
                VR = Visual_Reviewer(context_variables = self.context_variables, Max_Rounds = self.Max_Rounds)
                VR.run_Conversation()
                n+=1
        
        # Markdown Reviewer:
        k=1
        for file in self.ANALYSIS_DIR.iterdir():
            if k>= self.Max_Markdown:
                break
            if file.suffix.lower() == ".md":
                with open(file, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    MD = "".join(lines[:self.Max_MD_Lines]) 
                self.context_variables["Markdown_File"] = MD
                MR = MD_Reviewer(context_variables = self.context_variables, Max_Rounds = self.Max_Rounds)
                MR.run_Conversation()
                k+=1

        # Final Report Writer:
        image_list = []

        for item in self.context_variables["Image_Analysis"]:
            if isinstance(item, dict):
                x = item.get("Image_File")
                if x:
                    image_list.append(x)
        
        self.context_variables["Image_List"] = image_list

        RW = Report_Writing(context_variables = self.context_variables, Max_Rounds = self.Max_Rounds)
        RW.run_Conversation()
        
        self.create_pdf(
            report=self.context_variables["Final_Report"],
            analysis_dir=self.ANALYSIS_DIR,
            output_tex=str(self.ANALYSIS_DIR / "report.tex"),
            output_pdf=str(self.ANALYSIS_DIR / "report.pdf"),
            allowed_images=image_list
        )

        #--------- Storing Results for Judging:
        idx = 0
        for item in self.REPORT_DIR.iterdir():
            if item.suffix == ".md":
                idx+=1 # Aim is to save a list of pdf files and markdown files using idx+1 to distinguish between runs.

        # Saving Markdown File (Text Only) with User Requirements Section:
        markdown_name = self.REPORT_DIR/f"Report{idx+1}.md"
        REPORT= ""
        FR = self.context_variables["Final_Report"]
        for s in FR.Section:
            t = s.Title
            tx = s.Text
            section_content = f"""#{t} \n\n{tx}\n\n"""
            REPORT = REPORT + section_content
        User_Requirement_Section = f"""# User Requirements \n\n{self.context_variables["User_Requirements"]}"""
        REPORT = REPORT + User_Requirement_Section
        with open(markdown_name, "w", encoding="utf-8", errors="replace") as f:
            f.write(REPORT)
        
        # Saving PDF File (Text AND Images) by moving to the correct directory:
        # Move and rename generated PDF
        pdf_src = self.ANALYSIS_DIR / "report.pdf"
        pdf_dst = self.REPORT_DIR / f"Report{idx+1}.pdf"

        if pdf_src.exists():
            shutil.move(str(pdf_src), str(pdf_dst))
        else:
            print("Warning: report.pdf not found in Analysis directory.")
        


def main():
    context_variables = ContextVariables({
        # VL Model
        "Image_Analysis": [],
        "Current_Image_Name": "",
        "Max_Images": 0,

        # Markdown Model
        "Markdown_Analysis": [],
        "Markdown_File": "",
        "Max_Markdown_Files": 0,
        "Max_Markdown_Lines": 0,

        # Report_Writer
        "Final_Report":"",
        "Image_List":[],



    })
    Report = Final_Report(context_variables = context_variables, LLM_Name = "Test", Max_Rounds = 10)
if __name__ == "__main__":
    main()


            

            



