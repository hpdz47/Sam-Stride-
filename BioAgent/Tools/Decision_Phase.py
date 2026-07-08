"""
Decision_Phase.py
=================

Deterministic decision-making layer for the host-guest screening workflow,
mirroring the reference ``workflows/nmr/supramol/screening.py`` but entirely
off-instrument (nmrglue + scipy, no TopSpin, no Waters SDK).

Design (same principle as the Discovery phase, ``Tools/Discovery_Phase.py``):
the deterministic verdicts are computed by validated tools BEFORE any LLM runs
(see ``Chatrooms/Data_Discovery.py:67-68`` for the pattern). The MAS agents
then *interpret and report* these verdicts; they never regenerate them.

Public API (chatroom-facing, mirrors the Discovery hooks' signatures):

* ``Run_Screening(context_variables, Input_Dir)`` -> ReplyResult
      Walk an experiment set, compute per-sample NMR/MS/overall verdicts with
      the ported heuristics, store them in ``context_variables["Screening_Verdicts"]``.
* ``Screening_Hook(sender, message, recipient, silent)``
      process_message_before_send hook that captures the interpreter agent's
      narration into ``context_variables["Screening_Interpretation"]``.

Lower-level, unit-testable:

* ``screen_experiment_set(data_dir, name)`` -> dict keyed by position.

Input layout (the published Zenodo / cooper-group bundle, and what the MAS
mounts under ``/inputs``)::

    <data_dir>/
      DATA/INPUT/<NAME>-SM-NMR.json     # starting-material peak lists
      DATA/INPUT/<NAME>.json            # manifest: position -> amine/carbonyl/metal
      DATA/<NAME>/DATA/SUMMARY_MS.json  # MS hits (raw .RAW parse is vendor-locked)
      RAW-NMR/NMR/<NAME>/DATA/NMR/<NAME>-<pos:02d>/   # raw Bruker per position

The MS side consumes the provided hit lists (``SUMMARY_MS.json``) because raw
Waters ``.RAW`` parsing needs the Windows-only MassLynx SDK; the *decision*
heuristic (the paper's contribution) is applied by ``ms_quorum_rule``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

try:
    from autogen import Agent, ConversableAgent
    from autogen.agentchat.group import ContextVariables, ReplyResult
except ImportError:  # autogen is only needed for the chatroom-facing wrappers;
    # the deterministic decision functions below run standalone (e.g. in tests).
    Agent = ConversableAgent = ContextVariables = object

    class ReplyResult:  # minimal stand-in so signatures still resolve
        def __init__(self, message="", context_variables=None):
            self.message = message
            self.context_variables = context_variables

from Tools.Spectral_Analysis import (
    binding_shift_detection,
    combine_screening,
    ms_quorum_rule,
    nmr_screening_details,
    nmr_screening_rule,
    replication_check,
)
from Tools.Spectral_Ingestion import (
    HOSTGUEST_LINE_BROADENING,
    bruker_experiment_peaks,
    read_bruker_spectrum,
    pick_peaks,
    spectrum_region,
)

DEFAULT_NAME = "SUPRAMOL-SCREENING"
REPLICATION_NAME = "SUPRAMOL-REPLICATION"
HOSTGUEST_NAME = "SUPRAMOL-HOST-GUEST"

# Replication DTW: threshold from settings.toml. pruning_threshold = 0.1 (the
# value used by same_as_reference2 in the reference package) reproduces the
# published verdicts off-instrument, compensating for nmrglue's baseline vs
# TopSpin's apbk. Host-guest shift threshold = settings hg_shift.
DTW_THRESHOLD = 20.0
DTW_PRUNE = 0.1
HG_SHIFT = 0.02


# ---------------------------------------------------------------------------
# Core deterministic screening (pure enough to unit-test on CPU)
# ---------------------------------------------------------------------------
def _sm_peaks(sm_nmr: dict, block: str) -> list[float]:
    entry = sm_nmr.get(block)
    return list(entry.get("peaks_ppm", [])) if entry else []


def screen_experiment_set(
    data_dir: str | Path,
    name: str = DEFAULT_NAME,
    use_processed: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compute per-position screening verdicts for one experiment set.

    Reproduces ``screening.py``: for each sample, process the NMR, pick peaks,
    apply ``nmr_screening_rule`` against the amine+carbonyl reagents, apply the
    MS quorum to the provided hits, and combine. Returns a dict keyed by
    position string with the verdicts and diagnostics.

    Parameters
    ----------
    data_dir
        Root of the experiment bundle (see module docstring for layout).
    name
        Workflow prefix (default ``SUPRAMOL-SCREENING``).
    use_processed
        ``True`` reads TopSpin's processed ``1r`` (exact paper reproduction);
        ``False`` processes the raw FID (portable path, needed for Varian).
    """
    data_dir = Path(data_dir)
    base = data_dir / "DATA"
    sm_nmr = json.loads((base / "INPUT" / f"{name}-SM-NMR.json").read_text())
    manifest = json.loads((base / "INPUT" / f"{name}.json").read_text())

    ms_summary_path = base / name / "DATA" / "SUMMARY_MS.json"
    ms_summary = (
        json.loads(ms_summary_path.read_text())
        if ms_summary_path.exists()
        else {}
    )

    raw_root = data_dir / "RAW-NMR" / "NMR" / name / "DATA" / "NMR"

    verdicts: dict[str, dict[str, Any]] = {}
    for pos, entry in manifest.items():
        info = entry["sample_info"]
        amine, carbonyl, metal = info["amine"], info["carbonyl"], info["metal"]

        # --- NMR: ingest -> pick peaks -> screening rule ---
        sample_dir = raw_root / f"{name}-{int(pos):02d}"
        peaks, meta = bruker_experiment_peaks(
            sample_dir, expno="10", use_processed=use_processed
        )
        reagents = [_sm_peaks(sm_nmr, amine), _sm_peaks(sm_nmr, carbonyl)]
        nmr_pass = nmr_screening_rule(peaks, reagents)
        nmr_detail = nmr_screening_details(peaks, reagents)

        # --- MS: quorum on provided hits (raw .RAW parse is vendor-locked) ---
        ms_hits = ms_summary.get(pos, {}).get("mz_peaks", [])
        ms_pass, surviving = ms_quorum_rule(ms_hits) if ms_hits else (False, [])

        overall = combine_screening(nmr_pass, ms_pass)

        verdicts[str(pos)] = {
            "sample_info": info,
            "amine": amine,
            "carbonyl": carbonyl,
            "metal": metal,
            "peaks_ppm": peaks,
            "NMR_PASS": nmr_pass,
            "MS_PASS": ms_pass,
            "REPLICATION": overall,  # paper: screening pass -> proceed to replication
            "mz_peaks": surviving,
            "nmr_detail": nmr_detail,
        }
    return verdicts


# ---------------------------------------------------------------------------
# Replication (ports workflows/nmr/supramol/replication.py + same_as_reference)
# ---------------------------------------------------------------------------
def replicate_experiment_set(
    data_dir: str | Path,
    name: str = REPLICATION_NAME,
    screening_name: str = DEFAULT_NAME,
    use_processed: bool = True,
    distance_threshold: float = DTW_THRESHOLD,
    pruning_threshold: float = DTW_PRUNE,
) -> dict[str, dict[str, Any]]:
    """Compute per-replicate verdicts for a replication run.

    Reproduces ``replication.py``: for each replicate, DTW-compare the (6, 11)
    ppm region of its spectrum against its screening reference (``screening_id``)
    via ``same_as_reference`` (``replication_check``), combine with the MS
    verdict, and mark ``REPLICATED = NMR_PASS AND MS_PASS``.
    """
    data_dir = Path(data_dir)
    base = data_dir / "DATA"
    manifest = json.loads((base / "INPUT" / f"{name}.json").read_text())

    ms_path = base / name / "DATA" / "SUMMARY_MS.json"
    ms_summary = json.loads(ms_path.read_text()) if ms_path.exists() else {}

    raw_root = data_dir / "RAW-NMR" / "NMR" / name / "DATA" / "NMR"
    screen_root = (
        data_dir / "RAW-NMR" / "NMR" / screening_name / "DATA" / "NMR"
    )

    verdicts: dict[str, dict[str, Any]] = {}
    for pos, entry in manifest.items():
        info = entry["sample_info"]
        screen_id = int(info["screening_id"])

        test = spectrum_region(
            raw_root / f"{name}-{int(pos):02d}", use_processed=use_processed
        )
        ref = spectrum_region(
            screen_root / f"{screening_name}-{screen_id:02d}",
            use_processed=use_processed,
        )
        distance, nmr_pass = replication_check(
            test, [ref], distance_threshold, pruning_threshold
        )

        ms_hits = ms_summary.get(pos, {}).get("mz_peaks", [])
        ms_pass, surviving = ms_quorum_rule(ms_hits) if ms_hits else (False, [])
        replicated = combine_screening(nmr_pass, ms_pass)

        verdicts[str(pos)] = {
            "sample_info": info,
            "screening_id": str(screen_id),
            "dtw_distance": round(distance, 3),
            "NMR_PASS": nmr_pass,
            "MS_PASS": ms_pass,
            "REPLICATED": replicated,
            "mz_peaks": surviving,
        }
    return verdicts


# ---------------------------------------------------------------------------
# Host-guest binding (ports workflows/nmr/supramol/host_guest.py + guest_binding)
# ---------------------------------------------------------------------------
def host_guest_experiment_set(
    data_dir: str | Path,
    name: str = HOSTGUEST_NAME,
    replication_name: str = REPLICATION_NAME,
    use_processed: bool = True,
    shift_threshold: float = HG_SHIFT,
) -> dict[str, dict[str, Any]]:
    """Compute per-sample host-guest binding verdicts.

    Reproduces ``host_guest.py``: process the host-guest spectrum (line
    broadening ``hg_lb``) and pick sensitive peaks (``mi=3``); process the
    replication reference and pick strong host peaks (``mi=75``); a host peak
    that has no mixture peak within ``shift_threshold`` ppm has moved -> the
    guest is BOUND (``guest_binding`` / ``binding_shift_detection``).
    """
    data_dir = Path(data_dir)
    base = data_dir / "DATA"
    manifest = json.loads((base / "INPUT" / f"{name}.json").read_text())

    hg_root = data_dir / "RAW-NMR" / "NMR" / name / "DATA" / "NMR"
    rep_root = (
        data_dir / "RAW-NMR" / "NMR" / replication_name / "DATA" / "NMR"
    )

    verdicts: dict[str, dict[str, Any]] = {}
    for pos, entry in manifest.items():
        info = entry["sample_info"]

        ppm_t, data_t, _ = read_bruker_spectrum(
            hg_root / f"{name}-{int(pos):02d}", use_processed=use_processed
        )
        ppm_r, data_r, _ = read_bruker_spectrum(
            rep_root / f"{replication_name}-{int(pos):02d}",
            use_processed=use_processed,
        )
        # Sensitive picking of the mixture (mi=3), strong host peaks in the
        # reference (mi=75); noise-aware gate keeps the mixture list clean.
        test_peaks = pick_peaks(ppm_t, data_t, minimum_intensity=3)
        ref_peaks = pick_peaks(ppm_r, data_r, minimum_intensity=75)

        bound, triggers = binding_shift_detection(
            test_peaks, ref_peaks, shift_threshold
        )
        verdicts[str(pos)] = {
            "sample_info": info,
            "peaks_ppm": test_peaks,
            "HG_BOUND": bound,
            "trigger_peaks": triggers,
        }
    return verdicts


# ---------------------------------------------------------------------------
# Chatroom-facing functions (context-variable side effects)
# ---------------------------------------------------------------------------
def Run_Screening(
    context_variables: ContextVariables,
    Input_Dir: str | Path,
    name: str = DEFAULT_NAME,
    use_processed: bool = True,
) -> ReplyResult:
    """Deterministically screen an experiment set and store the verdicts.

    Called BEFORE the interpreter agent (see ``Chatrooms/Decision_Making.py``),
    exactly as ``Profile_Check``/``Deterministic_EDA`` are in the Discovery
    phase. Populates ``Screening_Verdicts`` and a compact ``Screening_Summary``.
    """
    verdicts = screen_experiment_set(
        Input_Dir, name=name, use_processed=use_processed
    )
    context_variables["Screening_Verdicts"] = verdicts

    # Compact, LLM-friendly summary (keeps context small; no raw arrays).
    summary = []
    for pos, v in verdicts.items():
        summary.append(
            {
                "position": pos,
                "amine": v["amine"],
                "carbonyl": v["carbonyl"],
                "metal": v["metal"],
                "n_peaks": len(v["peaks_ppm"]),
                "NMR_PASS": v["NMR_PASS"],
                "MS_PASS": v["MS_PASS"],
                "overall_pass": v["REPLICATION"],
            }
        )
    context_variables["Screening_Summary"] = summary
    context_variables["Screening_Available"] = True

    n_pass = sum(1 for v in verdicts.values() if v["REPLICATION"])
    return ReplyResult(
        message=(
            f"Screening complete: {len(verdicts)} samples, "
            f"{n_pass} passed both NMR and MS."
        ),
        context_variables=context_variables,
    )


def Run_Replication(
    context_variables: ContextVariables,
    Input_Dir: str | Path,
    name: str = REPLICATION_NAME,
    use_processed: bool = True,
) -> ReplyResult:
    """Deterministically run the replication check and store the verdicts.

    Populates ``Replication_Verdicts`` and a compact ``Replication_Summary``.
    """
    verdicts = replicate_experiment_set(
        Input_Dir, name=name, use_processed=use_processed
    )
    context_variables["Replication_Verdicts"] = verdicts
    context_variables["Replication_Summary"] = [
        {
            "position": pos,
            "screening_id": v["screening_id"],
            "dtw_distance": v["dtw_distance"],
            "NMR_PASS": v["NMR_PASS"],
            "MS_PASS": v["MS_PASS"],
            "REPLICATED": v["REPLICATED"],
        }
        for pos, v in verdicts.items()
    ]
    context_variables["Replication_Available"] = True
    n = sum(1 for v in verdicts.values() if v["REPLICATED"])
    return ReplyResult(
        message=f"Replication complete: {n}/{len(verdicts)} replicated.",
        context_variables=context_variables,
    )


def Run_HostGuest(
    context_variables: ContextVariables,
    Input_Dir: str | Path,
    name: str = HOSTGUEST_NAME,
    use_processed: bool = True,
) -> ReplyResult:
    """Deterministically run host-guest binding detection and store verdicts.

    Populates ``Binding_Verdicts`` and a compact ``Binding_Summary``.
    """
    verdicts = host_guest_experiment_set(
        Input_Dir, name=name, use_processed=use_processed
    )
    context_variables["Binding_Verdicts"] = verdicts
    context_variables["Binding_Summary"] = [
        {
            "position": pos,
            "HG_BOUND": v["HG_BOUND"],
            "n_trigger_peaks": len(v["trigger_peaks"]),
            "trigger_peaks": v["trigger_peaks"],
        }
        for pos, v in verdicts.items()
    ]
    context_variables["Binding_Available"] = True
    n = sum(1 for v in verdicts.values() if v["HG_BOUND"])
    return ReplyResult(
        message=f"Host-guest analysis complete: {n}/{len(verdicts)} bound.",
        context_variables=context_variables,
    )


def Screening_Hook(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool,
) -> Union[dict[str, Any], str]:
    """Capture the interpreter agent's narration into context.

    Mirrors ``Tools/Discovery_Phase.py:EDA_Hook`` including the handoff guard.
    """
    content = message.get("content", "") if isinstance(message, dict) else str(message)

    if isinstance(content, str):
        stripped = content.strip()
        if (
            stripped.startswith("[Handing off to")
            or stripped.startswith("Handing off to")
            or stripped.startswith("***** ")
            or "handoff" in stripped.lower()[:50]
        ):
            return message  # handoff marker: pass through unchanged

    try:
        data = json.loads(content)
        interpretation = data.get(
            "interpretation", data.get("Interpretation", content)
        )
    except Exception:
        interpretation = content

    sender.context_variables["Screening_Interpretation"] = interpretation
    sender.context_variables["Screening_Interpreted"] = True
    return message
