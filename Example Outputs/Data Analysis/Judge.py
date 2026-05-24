import os
from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path
import json
import shutil
from scipy.stats import pmean
import csv
from typing import Optional

def LLM_Judge(Input_Path: Path, file_name: str):
    CURRENT_DIR = Path(__file__).parent
    USAGE_DIR = CURRENT_DIR / "Usage.txt"
    usage_rate = None
    if not USAGE_DIR.exists():
        with open(USAGE_DIR, "w") as f:
            f.write(str(0))
    
    with open(USAGE_DIR, "r") as f:
        usage_rate = f.read()
        if int(usage_rate) >= 200:
            raise Exception("API Usage limit reached. Please check usage at https://www.anthropic.com/. Refresh the Usage.txt file if you wish to continue using the API.")
    usage_rate = int(usage_rate) + 1
    with open(USAGE_DIR, "w") as f:
        f.write(str(usage_rate))

    """
    Docstring: This function is for using the Anthropic API to review documents and assign
    numerical scores based on the document quality.
    Args:
        Input_Path (Path): The path to the results folder that contains all documents for review.
        system_message (str): The system message that provides instructions to the LLM for reviewing the document.
        file_name (str): The name of the file to be reviewed, including the extension (e.g., "Report1.pdf").
    Returns:
        Clarity_Score (int): A score from 1 to 10 for the clarity of the report.
        Depth_of_Analysis_Score (int): A score from 1 to 10 for the depth of analysis of results and the effectiveness of the data analysis performed in the report.
        Originality_Score (int): A score from 1 to 10 for the originality of the report.
        Writing_Quality_Score (int): A score from 1 to 10 for the writing quality of the report.
        Use_of_Visuals_Score (int): A score from 1 to 10 for the use of visuals in the report.
        Scientific_Rigour_Score (int): A score from 1 to 10 for the scientific rigour of the report.
        Explanation (str): A brief explanation of the scores assigned, no more than 200 words.
    """
    load_dotenv()

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    system_message = """
    You are a judge that is tasked wth revewing a pdf report and assigning numerical scores based on the quality.
    In your judgement, you must use the following criteria:
    1. Clarity: How clear and understandable is the report? Are the ideas and arguments presented in a logical manner?
    2. Depth of Analysis: Does the report provide a thorough analysis of the topic? Are the arguments well-supported with evidence and examples? Is data analysis performed effectively and does it make a contribution by providing insights into the data in a meaningful way?
    3. Originality: Does the report offer unique insights or perspectives on the topic? Does it demonstrate creativity in its approach to the subject matter?
    4. Writing Quality: Is the report well-written, with proper grammar, spelling, and punctuation? Is the writing style engaging and appropriate for the intended audience?
    5. Use of Visuals: If the report includes visuals (e.g., charts, graphs, images), are they effectively used to enhance the understanding of the content? Are they clear and well-designed?
    6. Scientific Rigour: Does the report demonstrate a strong understanding of the scientific method? Are the research methods and data analysis appropriate and well-executed? Are appropriate conclusions supported by the evidence presented?

    Based on these criteria, you will assign a score from 1 to 10 for the report, with 10 being the highest score.
    """

    file_path = Input_Path / file_name
    with open(file_path, "rb") as f:
        uploaded = client.beta.files.upload(
            file=(f"{file_name}", f, "application/pdf"),
        )

    tools = [
        {
            "name": "document_review",
            "description": "Record the summary and score for the document.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "Clarity_Score": {
                        "type": "integer",
                        "description": "A score from 1 to 10.",
                        "minimum": 1,
                        "maximum": 10,
                    },

                    "Depth_of_Analysis_Score": {
                        "type": "integer",
                        "description": "A score from 1 to 10.",
                        "minimum": 1,
                        "maximum": 10,
                    },

                    "Originality_Score": {
                        "type": "integer",
                        "description": "A score from 1 to 10.",
                        "minimum": 1,
                        "maximum": 10,
                    },

                    "Writing_Quality_Score": {
                        "type": "integer",
                        "description": "A score from 1 to 10.",
                        "minimum": 1,
                        "maximum": 10,
                    },

                    "Use_of_Visuals_Score": {
                        "type": "integer",
                        "description": "A score from 1 to 10.",
                        "minimum": 1,
                        "maximum": 10,
                    },

                    "Scientific_Rigour_Score": {
                        "type": "integer",
                        "description": "A score from 1 to 10.",
                        "minimum": 1,
                        "maximum": 10,
                    },

                    "Explanation": {
                        "type": "string",
                        "description": "A brief explanation of the scores assigned. No more than 200 words.",
                    }
                },
                "required": ["Clarity_Score", "Depth_of_Analysis_Score", "Originality_Score", "Writing_Quality_Score", "Use_of_Visuals_Score", "Scientific_Rigour_Score", "Explanation"],
            },
        }
    ]

    response = client.beta.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system_message,
        tools=tools,
        tool_choice={"type": "tool", "name": "document_review"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please review the following document."},
                    {
                        "type": "document",
                        "source": {
                            "type": "file",
                            "file_id": uploaded.id,
                        },
                    },
                ],
            },
        ],
        betas=["files-api-2025-04-14"],
    )

    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    clarity_score = tool_use_block.input["Clarity_Score"]
    depth_of_analysis_score = tool_use_block.input["Depth_of_Analysis_Score"]
    originality_score = tool_use_block.input["Originality_Score"]
    writing_quality_score = tool_use_block.input["Writing_Quality_Score"]
    use_of_visuals_score = tool_use_block.input["Use_of_Visuals_Score"]
    scientific_rigour_score = tool_use_block.input["Scientific_Rigour_Score"]
    explanation = tool_use_block.input["Explanation"]

    return clarity_score, depth_of_analysis_score, originality_score, writing_quality_score, use_of_visuals_score, scientific_rigour_score, explanation

def Average_Score(Folders:Path, p:int=1, Iters: int=1)-> list:
    """ Docstring: This function searches the given folder for all Report folders generated by LLM_Judge function, then
                   it extracts the scores to apply a power mean function.
                   Args:
                       Folders (Path): The path to the outermost results folder that contains all the Report folders generated by LLM_Judge function.
                       p (int): The power parameter for the power mean function. Default is 1, which corresponds to the arithmetic mean.
                       Iters (int): The iteration number to look for in the folder names.
                       Model_Size (Optional[int]): The model size to filter the folders. Default is None, which means no filtering by model size. This is only useful
                       for when this function is called to modify the power mean calculation at a later stage.
    """
    def get_Iter_Dir(root: Path, Iters:int):
        subdirs = [d for d in root.iterdir() if str(Iters) in d.name] # Finds all folders that have the correct iteration number in the name.
        if not subdirs:
            return root
        return get_Iter_Dir(subdirs[0], Iters) # Returns the single subdir with the results for that iteration number.
    
    ablation_dir = get_Iter_Dir(Folders, Iters)
    Study_Folders = [ablation_dir/"NN", ablation_dir/"MN", ablation_dir/"NM", ablation_dir/"MM"]

    def get_report_paths(root: Path)-> list:
        subdirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("Report")]
        return subdirs
    
    for Folder in Study_Folders:
        if Folder.exists():
            report_paths = get_report_paths(Folder)
            score_array = []
            for item in report_paths:
                evaluation_file = item/"Evaluation.json"
                if evaluation_file.exists():
                    with open(evaluation_file, "r") as f:
                        #print(f"evaluation_file {evaluation_file}")
                        data = json.load(f)
                    scores = [
                        data["Clarity Score"],
                        data["Depth of Analysis Score"],
                        data["Originality Score"],
                        data["Writing Quality Score"],
                        data["Use of Visuals Score"],
                        data["Scientific Rigour Score"]
                    ]
                    #print(f"Scores extracted from {evaluation_file}: {scores}")
                    average_score = pmean(scores, p)
                    score_array.append(average_score)
            print(f"Average Scores for {Folder}: {score_array}")
            print(f"\n----\nAverage Scores saved for for {Folder}.\n----\n")
            with open(Folder/"LLM_Judge_Scoring.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Average_Score"])
                for n in score_array:
                    writer.writerow([n]) # Save as a column vector.
                    

def Evaluate(Folders: Path, Iters:int):
    print("\n=============\nRunning LLM-as-a-Judge Evaluation \n============= \n")

    def get_Iter_Dir(root: Path, Iters:int):
        subdirs = [d for d in root.iterdir() if str(Iters) in d.name] # Finds all folders that have the correct iteration number in the name.
        if not subdirs:
            return root
        return get_Iter_Dir(subdirs[0], Iters) # Returns the single subdir with the results for that iteration number.
    
    ablation_dir = get_Iter_Dir(Folders, Iters)
    Study_Folders = [ablation_dir/"NN", ablation_dir/"MN", ablation_dir/"NM", ablation_dir/"MM"]

    for Folder in Study_Folders:
        if Folder.exists():
            for file in Folder.iterdir():
                if file.suffix.lower() == ".pdf":
                    New_Folder = Folder/file.name.replace(".pdf", "")
                    New_Folder.mkdir(exist_ok=True, parents=True)
                    if (New_Folder/".SUCCESS").exists():
                        continue
                    clarity, depth, originality, writing_quality, visuals, rigour, explanation = LLM_Judge(file.parent, file.name)
                    Results = {
                        "Clarity Score": clarity,
                        "Depth of Analysis Score": depth,
                        "Originality Score": originality,
                        "Writing Quality Score": writing_quality,
                        "Use of Visuals Score": visuals,
                        "Scientific Rigour Score": rigour,
                        "Explanation": explanation
                    }
                    with open(New_Folder/"Evaluation.json", "w") as f:
                        json.dump(Results, f, indent=4)
                    dest = New_Folder / file.name
                    if dest.exists():
                        dest.unlink() # Remove the original PDF file if it already exists in the new folder.
                    shutil.move(file, dest)

                    with open(New_Folder/".SUCCESS", "w") as f:
                        pass
                    

    Average_Score(Folders,1, Iters) # Initialises this using the arithmetic mean to start with.


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python your_script.py <folder_path>")
        sys.exit(1)

    folder_path = Path(sys.argv[1])


    Evaluate(folder_path, Iters=5)  # You can change the iteration number as needed.
            