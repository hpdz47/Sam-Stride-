"""
validate_m3.py
=============

M3 gate: prove the replication (DTW ``same_as_reference``) and host-guest
binding (``guest_binding``) decision phases reproduce the paper's published
verdicts.

Ground truth (cooper-group-uol-robotics/data):
  SUPRAMOL-REPLICATION/DATA/SUMMARY_NMR.json   NMR_PASS + REPLICATED (12 samples)
  SUPRAMOL-HOST-GUEST/DATA/SUMMARY_NMR.json    HG_BOUND (12 samples)

Run::

    python Tools/tests/validate_m3.py [/path/to/data]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TOOLS = _HERE.parent.parent
sys.path.insert(0, str(_TOOLS.parent))

from Tools.Decision_Phase import (  # noqa: E402
    host_guest_experiment_set,
    replicate_experiment_set,
)

DEFAULT_DATA = Path("/workspace/cooper-group-uol-robotics/data")


def main(argv):
    data_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_DATA
    base = data_dir / "DATA"

    # ---- Replication ----
    rep_gt = json.loads(
        (base / "SUPRAMOL-REPLICATION" / "DATA" / "SUMMARY_NMR.json").read_text()
    )
    rep = replicate_experiment_set(data_dir)
    nmr_ok = repl_ok = 0
    print("=" * 78)
    print("REPLICATION (DTW same_as_reference, threshold 20.0, prune 0.1)")
    print(f"{'Pos':>3} {'sid':>3} {'dist':>7} | NMR pred/gt | REPLICATED pred/gt")
    print("-" * 78)
    for pos, v in rep.items():
        ng = bool(rep_gt[pos]["NMR_PASS"])
        rg = bool(rep_gt[pos]["REPLICATED"])
        nmr_ok += v["NMR_PASS"] == ng
        repl_ok += v["REPLICATED"] == rg
        print(
            f"{pos:>3} {v['screening_id']:>3} {v['dtw_distance']:>7} "
            f"| {str(v['NMR_PASS']):<5}/{str(ng):<5} "
            f"| {str(v['REPLICATED']):<5}/{str(rg):<5} "
            f"{'OK' if v['REPLICATED']==rg else 'XX'}"
        )
    nrep = len(rep)
    print("-" * 78)
    print(f"REPLICATION: NMR {nmr_ok}/{nrep}   REPLICATED {repl_ok}/{nrep}")
    print("=" * 78)

    # ---- Host-guest binding ----
    hg_gt = json.loads(
        (base / "SUPRAMOL-HOST-GUEST" / "DATA" / "SUMMARY_NMR.json").read_text()
    )
    hg = host_guest_experiment_set(data_dir)
    hg_ok = 0
    print("\nHOST-GUEST BINDING (guest_binding, shift 0.02 ppm)")
    print(f"{'Pos':>3} | HG_BOUND pred/gt | n_triggers")
    print("-" * 78)
    for pos, v in hg.items():
        g = bool(hg_gt[pos]["HG_BOUND"])
        hg_ok += v["HG_BOUND"] == g
        print(
            f"{pos:>3} | {str(v['HG_BOUND']):<5}/{str(g):<5} "
            f"{'OK' if v['HG_BOUND']==g else 'XX'} | {len(v['trigger_peaks'])}"
        )
    nhg = len(hg)
    print("-" * 78)
    print(f"HOST-GUEST: HG_BOUND {hg_ok}/{nhg}")
    print("=" * 78)

    passed = (nmr_ok == nrep and repl_ok == nrep and hg_ok == nhg)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
