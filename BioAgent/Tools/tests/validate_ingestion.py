"""
validate_ingestion.py
====================

M1 ingestion gate: prove the portable nmrglue peak-picker
(``Tools/Spectral_Ingestion.py``) reproduces the paper's TopSpin peak lists
well enough that the ported decision rule still returns the correct verdicts.

Two levels are reported for each of the 18 SUPRAMOL-SCREENING samples:

  A. PEAK AGREEMENT - peaks picked from TopSpin's processed ``1r`` vs the
     published ``peaks_ppm`` (matched within 0.02 ppm): matched / extra /
     missing counts.
  B. VERDICT AGREEMENT (end-to-end) - feed the *picked* peaks through
     ``nmr_screening_rule`` and compare to the published ``NMR_PASS``. This is
     the decision that actually matters.

Both the processed-read path (``use_processed=True``) and, optionally, the raw
FID path can be exercised; the processed path is the faithful reproduction of
the paper, the raw-FID path is the portable replacement that transfers to the
Varian data.

Run::

    python Tools/tests/validate_ingestion.py [/path/to/data]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TOOLS = _HERE.parent.parent
sys.path.insert(0, str(_TOOLS.parent))

from Tools.Spectral_Analysis import nmr_screening_rule  # noqa: E402
from Tools.Spectral_Ingestion import bruker_experiment_peaks  # noqa: E402

DEFAULT_DATA = Path("/workspace/cooper-group-uol-robotics/data")
NAME = "SUPRAMOL-SCREENING"
MATCH_TOL = 0.02  # ppm


def match_counts(picked, gt, tol=MATCH_TOL):
    gt_remaining = list(gt)
    matched = 0
    for x in picked:
        for y in list(gt_remaining):
            if abs(x - y) <= tol:
                gt_remaining.remove(y)
                matched += 1
                break
    extra = len(picked) - matched
    missing = len(gt_remaining)
    return matched, extra, missing


def sm_peaks(sm_nmr, block):
    entry = sm_nmr.get(block)
    return list(entry.get("peaks_ppm", [])) if entry else []


def run_path(raw_root, sm_nmr, summary, use_processed):
    """Run one ingestion path over all samples; return (verdict_ok, n, stats)."""
    positions = sorted(summary.keys(), key=int)
    verdict_ok = 0
    tot_matched = tot_extra = tot_missing = tot_gt = 0
    label = "processed 1r (TopSpin output)" if use_processed else \
        "raw FID (portable, nmrglue only)"

    print("=" * 96)
    print(f"INGESTION PATH: {label}")
    print(
        f"{'Pos':>3} {'amine':<5} {'carbonyl':<5} "
        f"| {'picked':>6} {'gt':>3} {'match':>5} {'xtra':>4} {'miss':>4} "
        f"| NMR pred/gt"
    )
    print("-" * 96)

    for pos in positions:
        rec = summary[pos]
        sample_dir = raw_root / f"{NAME}-{int(pos):02d}"
        picked, _ = bruker_experiment_peaks(
            sample_dir, expno="10", use_processed=use_processed
        )
        gt_peaks = rec["peaks_ppm"]
        m, x, mi = match_counts(picked, gt_peaks)
        tot_matched += m
        tot_extra += x
        tot_missing += mi
        tot_gt += len(gt_peaks)

        reagents = [
            sm_peaks(sm_nmr, rec["amine"]),
            sm_peaks(sm_nmr, rec["carbonyl"]),
        ]
        pred = nmr_screening_rule(picked, reagents)
        gt = bool(rec["NMR_PASS"])
        ok = pred == gt
        verdict_ok += ok
        print(
            f"{pos:>3} {rec['amine']:<5} {rec['carbonyl']:<5} "
            f"| {len(picked):>6} {len(gt_peaks):>3} {m:>5} {x:>4} {mi:>4} "
            f"| {str(pred):<5}/{str(gt):<5} {'OK' if ok else 'XX'}"
        )

    n = len(positions)
    recall = tot_matched / tot_gt if tot_gt else 0.0
    print("-" * 96)
    print(
        f"PEAK AGREEMENT: matched {tot_matched}/{tot_gt} "
        f"({recall:.1%} recovered), extra {tot_extra}, missing {tot_missing}"
    )
    print(f"VERDICT AGREEMENT (end-to-end): {verdict_ok}/{n} NMR verdicts match")
    print("=" * 96)
    print()
    return verdict_ok, n


def main(argv):
    data_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_DATA
    base = data_dir / "DATA"
    raw_root = data_dir / "RAW-NMR" / "NMR" / NAME / "DATA" / "NMR"
    if not raw_root.exists():
        print(f"ERROR: raw NMR not found at {raw_root}")
        return 2

    sm_nmr = json.loads((base / "INPUT" / f"{NAME}-SM-NMR.json").read_text())
    summary = json.loads((base / NAME / "DATA" / "SUMMARY_NMR.json").read_text())

    # Path 1: reproduce the paper exactly using TopSpin's processed spectrum.
    proc_ok, n = run_path(raw_root, sm_nmr, summary, use_processed=True)
    # Path 2: fully portable raw-FID processing (the path that reaches Varian).
    raw_ok, _ = run_path(raw_root, sm_nmr, summary, use_processed=False)

    print(f"SUMMARY: processed-1r {proc_ok}/{n} | raw-FID {raw_ok}/{n}")
    # Gate on the processed path (exact paper reproduction); raw-FID is a
    # portable best-effort baseline for the Varian transfer.
    return 0 if proc_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
