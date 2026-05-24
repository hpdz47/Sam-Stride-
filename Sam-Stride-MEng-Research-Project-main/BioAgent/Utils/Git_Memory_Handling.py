from pathlib import Path
import json
from typing import List, Dict, Optional, Union, Any

class GitMemory:
    _instance = None # Using the Singleton pattern to ensure that only 1 GitMemory is allowed.
    _initialised = False # This is to ensure that the __init__ method only runs once, even if multiple instances are attempted to be created.
    """ DocString:
        GitMemory has 3 tiers of memory:

        Tier 1: SHORT-TERM MEMORY: This is step-level memory that the GitManager uses to access the change history
                to pass to LLMs. (Note: On its own this could be managed only by using the GitManager, but future
                iterations of the system can use other forms of memory for different purposes).
        Tier 2: Medium-Term Memory: After each step, the GitMemory gets an LLM to help summarise learnngs for each file.
                                    This is useful for each iteration learning from previous experimentation and converging 
                                    to quicker solutions.
        Tier 3: Long-Term Memory: After entire run is complete, the Git Memory summarises learnings by file and data type.
                                  This uses an LLM to help summarise learnings not based on the file name, but based on the 
                                  data and domain specfic details for how future files using that type could be processed better.
                                  This could lead to a large knowledge base for future runs to access and potentially allow for
                                  fine-tuning LLMs on ths knowledge base.
        """
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self, Root_Dir: Path):
        if self._initialised:
            return
        self._initialised = True
        self.Root_Dir = Root_Dir
        self.Repo_Dir = self.Root_Dir / "Repos"
        # Tier 1 Instance Variables ----------------------------
        self.Step_Number = None
        self.Step_Folder = None
        self.Step_File = None
        self.Step_Data = None
        #--------------------------------------------------------
    #=======================================================================================================
    # Tier 1 Methods: Short-Term Memory
    #=======================================================================================================
    def set_step(self, Step_Number: int, Step_Folder: Path)->None:
        """
        Set which step we're working on and load its memory.
        
        Call this before using record() or history().
        """
        self.Step_Number = Step_Number
        self.Step_Folder = Step_Folder
        self.Step_File = self.Step_Folder / f"Step_{self.Step_Number}.json"
        self.Step_Data = self._load()

    def _load(self) -> Dict: # Underscore indicates that this s a private method for internal usage only.
        if self.Step_File.exists():
            with open(self.Step_File, 'r') as f:
                return json.load(f)
        else:
            # The outer folder will be created by the GitManager, so it can be assumed that it is already
            # there and we are just checking for the presence of the file in that path.
            return {
                "Step_Number": self.Step_Number,
                "Cycles": []
            }
    def _save(self)-> None:
        # If memory file already exists, it will be opened. If not, it will be created.
        with open(self.Step_File, 'w') as f:
            json.dump(self.Step_Data, f, indent=4)
    def record(self,Cycle_Number:Optional[int]=None, Approach: Optional[str]=None, Outcome: Optional[Union[str, dict]]=None, Error_Message: Optional[str]=None, Rollback: Optional[bool]=False)-> None:
        """ DocString:
            This method is used to record the information about each cycle in the analysis plan. 

            - Cycle_Number: The number of the cycle of iteration in trying to fix the code. This will give an LLM
                            an idea of how many times the code has been attempted to be fixed without success.
            - Approach: The description from the Debug Agent about the attempted code patch.
            - Outcome: Success or Fail.
            - Error_Message: If outcome was a fail, the error message will be summarised into 1 sentence by the Error Agent
                             to help the LLM understand what went wrong and without overwhelming it with unnecessary details.
        """
        self.Rollback = Rollback
        # If Rollback is True, the Step Data must be modifed to keep up to date with the actual state of the git repo.
        if self.Rollback:
            self.Step_Data["Cycles"].pop() # This will just remove the last entry in the commit history to ensure the history is a clean progression of the attempts without including rollbacks.
        else:
            self.Step_Data["Cycles"].append({
                "Cycle_Number": Cycle_Number,
                "Approach": Approach,
                "Outcome": Outcome,
                "Error_Message": Error_Message
            })
        self._save()
    def history(self)-> str:
        """ DocString:
        This method is used to return a string detailing the previous commit history for the current plan
        step. This will be used to gve an LLM an easy-to-read format so that it can learn from previous
        attempts and converge to a solution more quickly.
        """
        history_str = f"Step {self.Step_Number} Commit History:\n"
        for cycle in self.Step_Data["Cycles"]:
            history_str += f"Cycle {cycle['Cycle_Number']}:\n"
            history_str += f"Approach: {cycle['Approach']}\n"
            history_str += f"Outcome: {cycle['Outcome']}\n"
            if cycle['Error_Message']:
                history_str += f"Error Message: {cycle['Error_Message']}\n"
            history_str +="---------------------- \n"
        return history_str
    #=======================================================================================================
    # Tier 2 Methods: Medium-Term Memory
    #=======================================================================================================
    # TODO: Tier 2 methods will need to manage run-level memory. Tier 1 memeory does not know anything about run level.


    #=======================================================================================================
    # Tier 3 Methods: Long-Term Memory
    #=======================================================================================================

