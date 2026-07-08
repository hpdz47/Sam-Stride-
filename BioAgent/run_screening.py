"""
run_screening.py
================

Standalone entry point to run the host-guest screening decision phase on the
GPU cluster — the first end-to-end check of the M2/M3 work with the real
vLLM/Qwen backend.

It runs JUST the screening (decision) phase, not the full plan/code/report
pipeline, so it produces a narrated screening result directly:

  1. deterministic tools compute the verdicts (validated: 18/18 vs ground truth),
  2. the Screening_Interpreter agent (served by the Reasoning vLLM model on
     port 8002) narrates the plate-level outcome,
  3. verdicts + interpretation are printed and saved to /workspace/Results.

Data: mount the ground-truth bundle so that /inputs/DATA/... and
/inputs/RAW-NMR/... exist (i.e. copy the cooper-group `data/` CONTENTS into
$PROJECT/Inputs). Configure model ids in /app/.env (LLM_Model_Reasoning, API_KEY).

Env overrides (optional):
  SCREEN_WORKFLOW   workflow prefix (default SUPRAMOL-SCREENING)
  SCREEN_USE_PROCESSED  "1" reads TopSpin 1r (default), "0" processes raw FID
"""

import json
import os
from pathlib import Path

from autogen.agentchat.group import ContextVariables

from Chatrooms.Decision_Making import Decision_Making
from Config.vLLM_Manager import LLM_Manager

# A short focus-area statement so the interpreter has framing without running the
# full Focus Area phase first (standalone mode).
DEFAULT_FOCUS = (
    "Reproduce the Dai et al. (Nature 2024) autonomous host-guest discovery "
    "workflow: decide which amine/carbonyl/metal combinations self-assemble into "
    "the target metal-organic cage, using orthogonal 1H NMR (different-from-"
    "reagents) and UPLC-MS (expected-mass with multi-metal quorum) screening."
)


def main():
    workflow = os.getenv("SCREEN_WORKFLOW", "SUPRAMOL-SCREENING")
    use_processed = os.getenv("SCREEN_USE_PROCESSED", "1") == "1"

    context_variables = ContextVariables(
        {
            "Screening_Verdicts": {},
            "Screening_Summary": [],
            "Screening_Available": False,
            "Screening_Interpretation": "",
            "Screening_Interpreted": False,
            "Current_Sample_Detail": {},
            "Focus_Area_Statement": DEFAULT_FOCUS,
        }
    )

    try:
        dm = Decision_Making(
            context_variables=context_variables,
            Max_Rounds=10,
            Workflow_Name=workflow,
            Use_Processed=use_processed,
        )
        dm.run_Conversation()
    finally:
        LLM_Manager(LLM_Type="Reasoning").stop_server()

    # --- Report ---
    verdicts = context_variables["Screening_Verdicts"]
    interpretation = context_variables["Screening_Interpretation"]

    print("\n" + "=" * 78)
    print(f"SCREENING VERDICTS ({workflow}) — deterministic tools")
    print("=" * 78)
    n_pass = 0
    for pos, v in verdicts.items():
        n_pass += bool(v["REPLICATION"])
        print(
            f"  {pos:>2} {v['amine']:<5} {v['carbonyl']:<5} {v['metal']:<7} "
            f"NMR={str(v['NMR_PASS']):<5} MS={str(v['MS_PASS']):<5} "
            f"-> {'PASS' if v['REPLICATION'] else 'fail'}"
        )
    print(f"\n  {n_pass}/{len(verdicts)} samples passed both NMR and MS.")

    print("\n" + "=" * 78)
    print("INTERPRETER NARRATION (Qwen Reasoning model)")
    print("=" * 78)
    print(interpretation)

    # --- Save ---
    out_dir = Path("/workspace/Results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{workflow}-SCREENING-VERDICTS.json").write_text(
        json.dumps(verdicts, indent=2)
    )
    (out_dir / f"{workflow}-SCREENING-INTERPRETATION.txt").write_text(
        str(interpretation)
    )
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
