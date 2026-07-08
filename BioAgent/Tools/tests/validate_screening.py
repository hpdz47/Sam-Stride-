"""
validate_screening.py
=====================

M1 gate: prove the ported decision heuristics in ``Tools/Spectral_Analysis.py``
reproduce the paper's published ground-truth verdicts *exactly*.

Ground-truth bundle (cooper-group-uol-robotics/data), SUPRAMOL-SCREENING:

  INPUT
    SUPRAMOL-SCREENING-SM-NMR.json    starting-material peak lists (per block)
    SUPRAMOL-SCREENING-EXPECTED-MS.json  {position: {formula: {charge: mz}}}
  EXPECTED OUTPUT (what we must match)
    SUPRAMOL-SCREENING/DATA/SUMMARY_NMR.json  per-sample peaks_ppm + verdicts
    SUPRAMOL-SCREENING/DATA/SUMMARY_MS.json   per-sample mz_peaks + MS_PASS

For each of the 18 samples we:
  * NMR: build reagents = [amine_peaks, carbonyl_peaks] from SM-NMR.json, feed
    the published reaction peaks_ppm to ``nmr_screening_rule`` and compare to
    the published ``NMR_PASS``.
  * MS : feed the published ``mz_peaks`` hits to ``ms_quorum_rule`` and compare
    to the published ``MS_PASS``.
  * Screening verdict: ``combine_screening(NMR_PASS, MS_PASS)`` is reported for
    completeness.

Run::

    python Tools/tests/validate_screening.py [/path/to/data]

Exit code 0 iff every sample matches on both NMR and MS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make Tools importable whether run from repo root or from Tools/tests.
_HERE = Path(__file__).resolve()
_TOOLS = _HERE.parent.parent
sys.path.insert(0, str(_TOOLS.parent))

from Tools.Spectral_Analysis import (  # noqa: E402
    combine_screening,
    ms_quorum_rule,
    nmr_screening_details,
    nmr_screening_rule,
)

# Default location of the ground-truth data (repo cloned alongside this one).
DEFAULT_DATA = Path("/workspace/cooper-group-uol-robotics/data")


def load_ground_truth(data_dir: Path):
    base = data_dir / "DATA"
    sm_nmr = json.loads(
        (base / "INPUT" / "SUPRAMOL-SCREENING-SM-NMR.json").read_text()
    )
    summary_nmr = json.loads(
        (base / "SUPRAMOL-SCREENING" / "DATA" / "SUMMARY_NMR.json").read_text()
    )
    summary_ms = json.loads(
        (base / "SUPRAMOL-SCREENING" / "DATA" / "SUMMARY_MS.json").read_text()
    )
    return sm_nmr, summary_nmr, summary_ms


def sm_peaks(sm_nmr: dict, block: str) -> list[float]:
    """Peak list for a building block; empty if absent (e.g. aliphatic Tren)."""
    entry = sm_nmr.get(block)
    if entry is None:
        return []
    return list(entry.get("peaks_ppm", []))


def main(argv: list[str]) -> int:
    data_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_DATA
    if not (data_dir / "DATA").exists():
        print(f"ERROR: ground-truth data not found under {data_dir}")
        return 2

    sm_nmr, summary_nmr, summary_ms = load_ground_truth(data_dir)

    positions = sorted(summary_nmr.keys(), key=int)
    nmr_ok = ms_ok = 0
    nmr_fail: list[str] = []
    ms_fail: list[str] = []

    print("=" * 96)
    print(
        f"{'Pos':>3} {'amine':<6} {'carbonyl':<6} {'metal':<7} "
        f"| {'NMR pred/gt':<13} {'':<3} | {'MS pred/gt':<11} {'':<3} "
        f"| screen(pred)"
    )
    print("-" * 96)

    for pos in positions:
        rec = summary_nmr[pos]
        amine = rec["amine"]
        carbonyl = rec["carbonyl"]
        metal = rec["metal"]

        # ---- NMR ----
        reagents = [sm_peaks(sm_nmr, amine), sm_peaks(sm_nmr, carbonyl)]
        reaction = rec["peaks_ppm"]
        nmr_pred = nmr_screening_rule(reaction, reagents)
        nmr_gt = bool(rec["NMR_PASS"])
        nmr_match = nmr_pred == nmr_gt
        nmr_ok += nmr_match
        if not nmr_match:
            nmr_fail.append(pos)

        # ---- MS ----
        hits = summary_ms[pos]["mz_peaks"]
        ms_pred, _ = ms_quorum_rule(hits)
        ms_gt = bool(summary_ms[pos]["MS_PASS"])
        ms_match = ms_pred == ms_gt
        ms_ok += ms_match
        if not ms_match:
            ms_fail.append(pos)

        screen_pred = combine_screening(nmr_pred, ms_pred)

        nmr_flag = "OK " if nmr_match else "XX "
        ms_flag = "OK " if ms_match else "XX "
        print(
            f"{pos:>3} {amine:<6} {carbonyl:<6} {metal:<7} "
            f"| {str(nmr_pred):<5}/{str(nmr_gt):<6} {nmr_flag} "
            f"| {str(ms_pred):<4}/{str(ms_gt):<5} {ms_flag} "
            f"| {screen_pred}"
        )

    n = len(positions)
    print("-" * 96)
    print(f"NMR screening: {nmr_ok}/{n} match" + (
        "" if not nmr_fail else f"   FAILURES: {nmr_fail}"))
    print(f"MS  screening: {ms_ok}/{n} match" + (
        "" if not ms_fail else f"   FAILURES: {ms_fail}"))
    print("=" * 96)

    # Show NMR diagnostics for any mismatch to aid debugging.
    for pos in nmr_fail:
        rec = summary_nmr[pos]
        reagents = [
            sm_peaks(sm_nmr, rec["amine"]),
            sm_peaks(sm_nmr, rec["carbonyl"]),
        ]
        det = nmr_screening_details(rec["peaks_ppm"], reagents)
        print(f"\n[NMR mismatch @ {pos}] {rec['amine']}+{rec['carbonyl']}:")
        print(f"   {det}")

    return 0 if (nmr_ok == n and ms_ok == n) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
