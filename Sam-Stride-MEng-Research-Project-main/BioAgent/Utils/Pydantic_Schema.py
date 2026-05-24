from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Literal, Union

# ------------------ Structured Outputs -------------------

# 1 =======================================
class Step(BaseModel):
    Step_Number: int = Field(..., description="The step number in the plan.")
    Analysis_Type: str = Field(..., description="A clear statement of the analysis type that must be performed by the coder.")
    Data_File: str = Field(..., description="Explictly name the one data file that must be used for the analysis.")
    Variables: List[str] = Field(..., description="A list of variable names from the data file that must be used for the analysis.")
    Context: str = Field(..., description="Any important information that the coder must be aware of when handling the data.")
    Output_Format: str = Field(..., description="The output format that must be used for the analysis step. This must be either text-based output appended (never overwrite) to a markdown file OR visualisations saved as image files. It can be a combination of both")
    Output_Details: str=Field(..., description="Detailed instructions on how the output must be formatted. If the output is a graph, specify the type of graph, labels, title, and any other relevant details. If the output is text-based, specify the structure and content required.")


class PlanResponse(BaseModel):
    Plan_Section: List[Step] = Field(..., description="A list of steps for the Plan.")
    Number_of_Steps: int = Field(..., description="The total number of steps in the plan.")
#==========================================
# 2 =========================================
class P(BaseModel):
    Point: str = Field(..., description="If the point is a positive aspect, outline the specific positive aspect of the plan.")
class N(BaseModel):
    Point: str = Field(..., description="If the point is a negative aspect, outline the specific negative aspect of the plan. If there is no clear negative aspect, this should say 'No clear negative aspect identified'.")
class I(BaseModel):
    Point: str = Field(..., description="A single point outlining a specific improvement aspect of the plan. This should be to address the negatives outlined in the review.If there are no clear improvement aspect, this should say 'No clear improvement aspect identified'.")
    Severity: Literal["Major Issue", "Moderate Issue", "Minor Issue", "Dataset Limitation"] = Field(..., description="The severity of the improvement to be made, rated as Major Issue, Moderate Issue, Minor Issue, or Dataset Limitation.")


class ReviewResponse(BaseModel):
    Positives: List[P] = Field(..., description="A list of positive points outlining specific positive aspects of the plan.")
    Negatives: List[N] = Field(..., description="A list of negative points outlining specific negative aspects of the plan. If no clear negative aspects are identified, this should say 'No clear negative aspect identified'.")
    Improvements: List[I] = Field(..., description="A list of improvement points outlining specific improvement aspects of the plan. This should be to address the negatives outlined in the review. If there are no clear improvement aspects, this should say 'No clear improvement aspect identified'.")
#============================================
# 3 ==========================================
class J(BaseModel):
    Point: str = Field(..., description="A single point outlining a specific improvement that must be made.")
    Severity: Literal["Major Issue", "Moderate Issue", "Minor Issue", "Dataset Limitation"] = Field(..., description="The severity of the improvement to be made, rated as Major Issue, Moderate Issue, Minor Issue, or Dataset Limitation.")
class Compile_Response(BaseModel):
    Variable_Feedback: List[J] = Field(..., description="A list of the most critical improvements that must be made based on variable selection and usage.")
    Focus_Area_Alignment: List[J] = Field(..., description="A list of the most critical improvements that must be made based on the alignment between the focus area and the analysis type.")
    Task_Decomposition: List[J] = Field(..., description="A list of the most critical improvements that must be made based on the breakdown of the main analysis task into clearer sub-tasks.")
    Output_Instructions: List[J] = Field(..., description="A list of the most critical improvements that must be made based on the output instructions provided and the analysis type.")
#==========================================
# 4 =================================================
class R(BaseModel):
    ID: str = Field(..., description="A unique identifier for the resolved issue.")
    Resolved_Point: str = Field(..., description="A single point outlining a specific issue that has been resolved based on the improvement made.")
class U(BaseModel):
    ID: str = Field(..., description="A unique identifier for the unresolved issue.")
    Unresolved_Point: str = Field(..., description="A single point outlining a specific issue that remains unresolved based on the changes made. If there are no clear unresolved points, this should say 'No clear unresolved points identified'.")
    Required_Action: str = Field(..., description="A specific and actionable instruction on what must be done to resolve the issue that remains unresolved. This must be based on learnings from past feedback.")

class LearningResponse(BaseModel):
    Resolved_Issues: List[R] = Field(..., description="A list of the most critical issues that have been resolved based on the improvement made.")
    Unresolved_Issues: List[U] = Field(..., description="A list of the most critical issues that remain unresolved based on the changes made. These should be concise and actionable. If there are no clear unresolved points identified, this should say 'No clear unresolved points identified'.")
# 5 =================================================
class A(BaseModel):
    Approach: str = Field(..., description="A single point outlining the approach taken to update the plan. This should be assessed based on the Git diffs.")
class DiffResponse(BaseModel):
    Variables: List[A] = Field(..., description="A list of the changes made based on the variable selection and pre-processing steps.")
    Global_Analysis: List[A] = Field(..., description="A list of the changes made based on the overall analysis approach and structure of the code.")
    Local_Analysis: List[A] = Field(..., description="A list of the changes made based on the sub-tasks within the context section of any of the steps in the plan.")
    Output: List[A] = Field(..., description="A list of the changes made based on the output instructions provided in any of the steps in the plan.")
#======================================================

# 6 =========================================
class CP(BaseModel):
    Point: str = Field(..., description="If the point is a positive aspect, outline the specific positive aspect of the code.")
class CN(BaseModel):
    Point: str = Field(..., description="If the point is a negative aspect, outline the specific negative aspect of the code. If there is no clear negative aspect, this should say 'No clear negative aspect identified'.")
class CI(BaseModel):
    Point: str = Field(..., description="A single point outlining a specific improvement aspect of the code. This should be to address the negatives outlined in the review. If there are no clear improvement aspect, this should say 'No clear improvement aspect identified'.")
    Severity: Literal["Major Issue", "Moderate Issue", "Minor Issue"] = Field(..., description="The severity of the improvement to be made, rated as Major Issue, Moderate Issue or Minor Issue.")

class CodeReviewResponse(BaseModel):
    Positives: List[CP] = Field(..., description="A list of positive points outlining specific positive aspects of the code.")
    Negatives: List[CN] = Field(..., description="A list of negative points outlining specific negative aspects of the code. If no clear negative aspects are identified, this should say 'No clear negative aspect identified'.")
    Improvements: List[CI] = Field(..., description="A list of improvement points outlining specific improvement aspects of the code. This should be to address the negatives outlined in the review. If there are no clear improvement aspects, this should say 'No clear improvement aspect identified'.")
#============================================

#7 ========================================== 
class RAGQuestions(BaseModel):
    """Simple schema for RAG questions - just a list."""
    questions: List[str] = Field(
        ..., 
        description="List of 3-6 questions to ask the research database.",
        min_length=3,
        max_length=6
    )

#8 ============================================
class SummaryResponse(BaseModel):
    Outcome: bool = Field(..., description = "True if code execution is successful. False if code execution is not successful.")
    Summary: str = Field(..., description = "A concise summary of the code error message (if applicable).")
    Pip: bool = Field(..., description = "True if python packages need to be installed. False if error is not linked to python package installations")

#9 ==============================================
class GitControlResponse(BaseModel):
    Rollback: bool = Field(..., description = "Set to true to roll back or false to keep all changes.")
    Suggestions: str = Field(..., description = "Provide advice to the debug agent to help it learn from past mistakes.")

#10 ==============================================
class DebugResponse(BaseModel):
    Code: str = Field(..., description = "You must provide your updated code here.")
    Approach: str = Field(..., description = "You must detail the approach to fixing the code in a concise sentence.")

#11 =============================================
class VLResponse(BaseModel):
    Interpretation: str = Field(..., description = "You must provide a concise paragraph describing the data analysis performed and explaining the image produced from the data analysis.")

#12 ===============================================
class MarkdownResponse(BaseModel):
    Interpretation: str = Field(..., description = "You must provide a concise paragraph describing the data analysis performed and explaining the markdown file produced from the data analysis.")

#13 ===============================================
class Fig(BaseModel):
    Path: str = Field(..., description = "You MUST choose the name of an image from the list of images provided in the image analysis summaries.")
    Caption: str = Field(..., description = "This must be Figure <number>: <1-sentence description of figure>")
class Z(BaseModel):
    Title: Literal["Abstract", "Introduction", "Methodology", "Results and Discussion", "Conclusion"]
    Figures: List[Fig] = Field(..., description = "This is a list of figures that are required to accompany the explanation in the report. Only files ending in .png .jpg .jpeg are allowed.")
    Text: str = Field(..., description = "This is the text that will be placed in the final report under the title heading. The text must be based on the title chosen and the relevant information provided.")
    
class ReportResponse(BaseModel):
    Section: List[Z] = Field(..., description = "The template must be flled out for each section in the report. It must be based on the information provided.")
