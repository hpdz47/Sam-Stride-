"""
Spectral_Analysis.py
====================

Pure, deterministic re-implementation of the decision-maker heuristics from

    Dai et al., "Autonomous mobile robots for exploratory synthetic chemistry,"
    Nature 635 (2024)

as published in the ``synthesis_bots`` package (Szczypinski et al., v0.3.0),
under ``src/synthesis_bots/decisions/``.

The paper's decision-maker is a hand-coded heuristic, *not* an LLM. In this
project the Multi-Agent System *calls, interprets, and reports on* these
verdicts rather than regenerating the numerics each run (small local models are
unreliable at peak-list arithmetic). Every function here is:

* pure (no TopSpin, no Waters SDK, no file I/O, no network),
* unit-testable against the published ground-truth outputs, and
* a faithful port of the reference algorithm with the exact thresholds from
  ``settings.toml`` ``[workflows.decision]`` / ``[defaults]``.

Each function documents the reference file it ports so the fidelity can be
audited line-by-line.

Reference thresholds (settings.toml)::

    [workflows.decision]
    peak_number        = 3       # NMR: max |#SM_peaks - #reaction_peaks|
    shifted_proportion = 0.5     # NMR: fraction of SM peaks that must move
    metals_mz          = [3, 2]  # MS : >=3-metal assembly needs >=2 m/z hits
    dtw_threshold      = 20.0    # replication: DTW distance PASS ceiling
    ppm_range          = [11, 6] # window of interest
    hg_shift           = 0.02    # host-guest: ppm shift trigger
    hg_lb              = 1.8      # host-guest: Hz line broadening

    [defaults.MS]
    peak_match_tolerance = 0.4   # MS: absolute m/z match tolerance (Da)
"""

from __future__ import annotations

from itertools import chain
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Ground-truth thresholds (mirror of settings.toml). Kept as a module-level
# dict so the values live in exactly one place and match the reference package.
# ---------------------------------------------------------------------------
DECISION_SETTINGS: dict[str, Any] = {
    "peak_number": 3,
    "shifted_proportion": 0.5,
    "metals_mz": (3, 2),
    "dtw_threshold": 20.0,
    "ppm_range": (11.0, 6.0),
    "hg_shift": 0.02,
    "hg_lb": 1.8,
}
MS_PEAK_MATCH_TOLERANCE = 0.4  # Da, absolute (defaults.MS.peak_match_tolerance)


# ===========================================================================
# 1. NMR SCREENING  ---  ports decisions/different_from_reagents.py
# ===========================================================================
def nmr_screening_rule(
    reaction_peaks: Sequence[float],
    reagents_list: Sequence[Sequence[float]],
    peak_number: int = DECISION_SETTINGS["peak_number"],
    shifted_proportion: float = DECISION_SETTINGS["shifted_proportion"],
) -> bool:
    """Decide whether a reaction spectrum differs from its reagents.

    Faithful port of ``different_from_reagents.main``.

    The reagents' peak list is the *deduplicated union* of the two
    starting-material peak lists (amine + carbonyl). The reaction PASSES iff:

    1. ``abs(#reagent_peaks - #reaction_peaks) <= peak_number``  (count test),
       **and**
    2. after removing every reaction peak that coincides with a reagent peak
       (both rounded to 2 dp, i.e. ~0.01 ppm bins), the number of remaining
       "new" reaction peaks is ``>= shifted_proportion * #reagent_peaks``.

    Parameters
    ----------
    reaction_peaks
        Peak positions (ppm) picked from the reaction-mixture spectrum.
    reagents_list
        List of per-reagent peak lists, e.g. ``[amine_peaks, carbonyl_peaks]``.
    peak_number, shifted_proportion
        Thresholds; default to the published values.

    Returns
    -------
    bool
        ``True`` if the reaction is deemed different from the reagents.
    """
    # Deduplicated union of all reagent peaks (set() on raw floats, as in ref).
    reagents_peaks = list(set(chain(*reagents_list)))

    # (1) Count criterion.
    diff = abs(len(reagents_peaks) - len(reaction_peaks))
    if diff > peak_number:
        return False

    # (2) Shift criterion: discard reaction peaks matching a reagent peak
    #     (2-dp binning), then count how many "new" peaks remain.
    reaction_set = {round(x, 2) for x in reaction_peaks}
    for peak in reagents_peaks:
        reaction_set.discard(round(peak, 2))

    if len(reaction_set) < shifted_proportion * len(reagents_peaks):
        return False

    return True


def nmr_screening_details(
    reaction_peaks: Sequence[float],
    reagents_list: Sequence[Sequence[float]],
    peak_number: int = DECISION_SETTINGS["peak_number"],
    shifted_proportion: float = DECISION_SETTINGS["shifted_proportion"],
) -> dict[str, Any]:
    """Same logic as :func:`nmr_screening_rule` but returns a diagnostic dict.

    Useful for the MAS narration layer and for debugging: exposes the peak
    counts, the count/shift sub-verdicts and the surviving "new" peaks without
    changing the boolean decision.
    """
    reagents_peaks = list(set(chain(*reagents_list)))
    diff = abs(len(reagents_peaks) - len(reaction_peaks))
    count_pass = diff <= peak_number

    reaction_set = {round(x, 2) for x in reaction_peaks}
    for peak in reagents_peaks:
        reaction_set.discard(round(peak, 2))
    n_new = len(reaction_set)
    shift_pass = n_new >= shifted_proportion * len(reagents_peaks)

    return {
        "n_reagent_peaks": len(reagents_peaks),
        "n_reaction_peaks": len(reaction_peaks),
        "peak_count_diff": diff,
        "count_criterion_pass": count_pass,
        "n_new_peaks": n_new,
        "new_peaks": sorted(reaction_set, reverse=True),
        "shift_criterion_pass": shift_pass,
        "nmr_pass": bool(count_pass and shift_pass),
    }


# ===========================================================================
# 2. MS SCREENING  ---  ports decisions/expected_mass_metals.py (quorum layer)
#    plus the portable matching layer from lcms_parser (identify_hits).
# ===========================================================================
def _metal_count(formula: str) -> int:
    """Number of metal ions in an assembly formula.

    Ports the reference expression ``int(hit.formula.split("_")[1][1])``.
    E.g. ``"ZnNTf_(2)_Ald1_(6)_Tris_(2)"`` -> ``2`` (char at index 1 of
    ``"(2)"``). This assumes a single-digit metal count, exactly as the
    reference does.
    """
    return int(formula.split("_")[1][1])


def ms_quorum_rule(
    hits: Sequence[dict[str, Any]],
    metals_mz: tuple[int, int] = DECISION_SETTINGS["metals_mz"],
) -> tuple[bool, list[dict[str, Any]]]:
    """Apply the multi-metal quorum to a list of m/z hits.

    Faithful port of the decision half of ``expected_mass_metals.main``
    (the part after ``identify_hits``). Rule (``metals_mz = [x, y]``):

    * an assembly with ``>= x`` metals is retained only if it was observed via
      ``>= y`` distinct m/z hits;
    * an assembly with ``< x`` metals is retained on a single hit.

    The overall MS verdict is PASS iff at least one hit survives.

    Parameters
    ----------
    hits
        List of hit dicts; each must contain a ``"formula"`` key (extra keys
        such as ``mz_value``/``charge`` are preserved). This is exactly the
        structure of the ``mz_peaks`` entries in the published
        ``SUMMARY_MS.json`` / ``SUMMARY_NMR.json``.
    metals_mz
        ``(x, y)`` quorum; defaults to the published ``(3, 2)``.

    Returns
    -------
    (bool, list[dict])
        ``(MS_PASS, surviving_hits)``.
    """
    x_metals, y_hits = metals_mz

    # Count hits per >=x-metal formula (ref only increments for >=x metals).
    multiple_metals: dict[str, int] = {hit["formula"]: 0 for hit in hits}
    for hit in hits:
        if _metal_count(hit["formula"]) >= x_metals:
            multiple_metals[hit["formula"]] += 1

    pruned_hits: list[dict[str, Any]] = []
    for hit in hits:
        if _metal_count(hit["formula"]) >= x_metals:
            if multiple_metals[hit["formula"]] >= y_hits:
                pruned_hits.append(hit)
            # else: dropped for failing the quorum.
        else:
            pruned_hits.append(hit)

    return (len(pruned_hits) > 0), pruned_hits


def match_ms_hits(
    observed_mz: Sequence[float],
    expected_results: dict[str, dict[str, float]],
    atol: float = MS_PEAK_MATCH_TOLERANCE,
) -> list[dict[str, Any]]:
    """Match observed m/z values against an expected assembly table.

    Portable re-implementation of ``lcms_parser.WatersRawFile.identify_hits``'s
    *matching* step (the raw ``.RAW`` reading step is vendor-locked to the
    Waters MassLynx SDK and is handled separately in the ingestion layer).

    ``expected_results`` is the ``{formula: {charge: mz}}`` mapping shipped as
    ``{NAME}-EXPECTED-MS.json``. A hit is recorded whenever an observed m/z is
    within ``atol`` (absolute Da) of an expected m/z (ES+).

    Returns a list of hit dicts compatible with :func:`ms_quorum_rule`.
    """
    obs = np.asarray(observed_mz, dtype=float)
    hits: list[dict[str, Any]] = []
    for formula, charges in expected_results.items():
        for charge, mz_expected in charges.items():
            close = np.isclose(obs, float(mz_expected), atol=atol, rtol=0.0)
            for mz_value in obs[close]:
                hits.append(
                    {
                        "mz_value": float(mz_value),
                        "mode": "ES+",
                        "formula": formula,
                        "charge": str(charge),
                        "mz_expected": float(mz_expected),
                    }
                )
    return hits


# ===========================================================================
# 3. SCREENING VERDICT  ---  NMR_PASS AND MS_PASS
# ===========================================================================
def combine_screening(nmr_pass: bool, ms_pass: bool) -> bool:
    """Combine the two orthogonal screening tests (reference: logical AND)."""
    return bool(nmr_pass and ms_pass)


# ===========================================================================
# 4. REPLICATION  ---  ports decisions/same_as_reference2.py
# ===========================================================================
def replication_check(
    test_nmr: NDArray,
    reference_nmr: Sequence[NDArray],
    distance_threshold: float = DECISION_SETTINGS["dtw_threshold"],
    pruning_threshold: float = 0.1,
) -> tuple[float, bool]:
    """Decide whether a test spectrum replicates its reference spectra.

    Faithful port of ``same_as_reference2.main`` (plotting/archive side
    effects removed). Uses dynamic time warping (``dtw-python``) over
    min-max-normalised, noise-pruned full spectra.

    Steps (identical to reference):

    1. Truncate everything to the shortest length.
    2. Min-max normalise each reference, sum them, then min-max the sum.
    3. Min-max normalise the test.
    4. Prune points ``<= pruning_threshold`` to 0 (noise removal).
    5. ``distance = dtw(ref, test).distance``; PASS iff
       ``distance < distance_threshold``.

    Returns ``(distance, is_same)``.
    """
    import dtw  # local import: heavy dep, only needed for replication

    shortest = min([len(test_nmr), *[len(ref) for ref in reference_nmr]])
    test_np = np.asarray(test_nmr[0:shortest], dtype=float)

    refs_np = []
    for ref in reference_nmr:
        ref = np.asarray(ref, dtype=float)
        ref = (ref - ref.min()) / (ref.max() - ref.min())
        refs_np.append(ref[0:shortest])
    ref_np = np.array(refs_np).sum(axis=0)

    # Min-max scaling for thresholding.
    test_np = (test_np - test_np.min()) / (test_np.max() - test_np.min())
    ref_np = (ref_np - ref_np.min()) / (ref_np.max() - ref_np.min())

    # Prune below threshold to remove noise.
    ref_np_pruned = np.where(ref_np > pruning_threshold, ref_np, 0.0)
    test_np_pruned = np.where(test_np > pruning_threshold, test_np, 0.0)

    alignment = dtw.dtw(ref_np_pruned, test_np_pruned, keep_internals=True)
    distance = float(alignment.distance)

    return distance, (distance < distance_threshold)


# ===========================================================================
# 5. HOST-GUEST BINDING  ---  ports decisions/guest_binding.py
# ===========================================================================
def binding_shift_detection(
    test_nmr: Sequence[float],
    reference_nmr: Sequence[float],
    shift_threshold: float = DECISION_SETTINGS["hg_shift"],
) -> tuple[bool, list[float]]:
    """Detect host-guest binding via shifted host peaks (NMR-only).

    Faithful port of ``guest_binding.main``. For each reference (host) peak,
    if NO peak in the mixture spectrum lies within ``shift_threshold`` ppm
    (``np.isclose(atol=...)``), that host peak is deemed to have moved -> the
    guest is BOUND. Returns ``(hg_bound, trigger_peaks)`` where
    ``trigger_peaks`` are the reference peaks that moved.
    """
    test_nmr_peaks = np.asarray(test_nmr, dtype=float)
    ref_nmr_peaks = np.asarray(reference_nmr, dtype=float)
    hg_bound = False
    trigger_peaks: list[float] = []
    for peak in ref_nmr_peaks:
        if not np.isclose(test_nmr_peaks, peak, atol=shift_threshold).any():
            hg_bound = True
            trigger_peaks.append(float(peak))
    return hg_bound, trigger_peaks


# ===========================================================================
# 6. PEAK MATCHING  ---  Hungarian one-to-one by min |delta ppm|
#    (from the "Claude vs ChemDraw" NMR report; used for the predict-then-
#    compare channel and for diagnostics. NOT used by the reference rules.)
# ===========================================================================
def match_peaks(
    peaks_a: Sequence[float],
    peaks_b: Sequence[float],
    max_shift: float | None = None,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """One-to-one match two peak lists minimising total |delta ppm|.

    Uses the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``).
    Optionally drops matches whose ``|delta| > max_shift``.

    Returns ``(matches, unmatched_a, unmatched_b)`` where ``matches`` is a list
    of ``(index_a, index_b, abs_delta)`` and the unmatched lists hold the
    left-over indices.
    """
    from scipy.optimize import linear_sum_assignment

    a = np.asarray(peaks_a, dtype=float)
    b = np.asarray(peaks_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return [], list(range(a.size)), list(range(b.size))

    cost = np.abs(a[:, None] - b[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)

    matches: list[tuple[int, int, float]] = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    for i, j in zip(row_ind, col_ind):
        delta = float(cost[i, j])
        if max_shift is not None and delta > max_shift:
            continue
        matches.append((int(i), int(j), delta))
        matched_a.add(int(i))
        matched_b.add(int(j))

    unmatched_a = [i for i in range(a.size) if i not in matched_a]
    unmatched_b = [j for j in range(b.size) if j not in matched_b]
    return matches, unmatched_a, unmatched_b
