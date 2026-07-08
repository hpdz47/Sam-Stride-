# M1 — Spectral tools vs. published ground truth

This milestone reimplements the *Dai et al., Nature 2024* host–guest decision-maker
as **pure, deterministic, off-instrument tools** and validates them against the
paper's own published data + expected outputs (the `data/` bundle in the
`cooper-group-uol-robotics` repo, workflow **SUPRAMOL-SCREENING**, 18 samples =
3 amines × 3 carbonyls × 2 metals).

Design principle: the paper's decision-maker is *hand-coded heuristics*, not an
LLM. The MAS will **call, interpret, and report on** these validated tools; it
does not regenerate the numerics each run. The verdicts come from the tools; the
LLMs orchestrate and narrate.

## What was built

| File | Role |
|------|------|
| `Tools/Spectral_Analysis.py`   | Pure ports of the four decision heuristics + peak matching. No TopSpin, no Waters SDK, no I/O. |
| `Tools/Spectral_Ingestion.py`  | `nmrglue`+`scipy` replacement for the TopSpin processing/peak-pick layer. Reads Bruker **and** Varian; two paths (processed `1r`, or raw FID). |
| `Tools/tests/validate_screening.py`  | Decision-layer gate vs published `SUMMARY_NMR.json` / `SUMMARY_MS.json`. |
| `Tools/tests/validate_ingestion.py`  | Ingestion gate: picked peaks → decision rule → verdicts, both ingestion paths. |

### Decision ports (`Spectral_Analysis.py`), each a faithful copy of the reference
- `nmr_screening_rule` ← `decisions/different_from_reagents.py` (`peak_number=3`, `0.5×` new-peak test, 2-dp binning)
- `ms_quorum_rule` ← decision half of `decisions/expected_mass_metals.py` (`metals_mz=[3,2]`)
- `match_ms_hits` ← portable half of `lcms_parser.identify_hits` (`atol=0.4` Da, ES+)
- `replication_check` ← `decisions/same_as_reference2.py` (DTW, prune `>0.1`, PASS `< 20.0`)
- `binding_shift_detection` ← `decisions/guest_binding.py` (`np.isclose(atol=0.02)` ppm)
- `combine_screening` = `NMR_PASS AND MS_PASS`
- `match_peaks` = Hungarian min-|Δδ| (diagnostic / predict-then-compare channel)

## Validation results

**Decision layer** (fed the published peak lists / m/z hits):
```
NMR screening: 18/18 verdicts match
MS  screening: 18/18 verdicts match
```

**Ingestion layer** (peaks picked from the raw spectra, then run through the rule):
```
processed-1r path (reproduces the paper using TopSpin's output):
    peak recall 212/216 (98.1%), extra 2, missing 4
    18/18 NMR verdicts match
raw-FID path (fully portable, nmrglue only — the path that reaches Varian):
    16/18 NMR verdicts match
```

The 2-verdict gap on the raw-FID path is phase/baseline difference vs. TopSpin's
proprietary AI phasing (`apbk`); it is the plan's risk #2 made concrete. For the
Bruker paper-reproduction we use the processed `1r` (18/18); for the Varian
dataset (M4) the raw-FID path is the only option and 16/18 on Bruker is the
confidence baseline to improve on.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install numpy scipy nmrglue dtw-python
# ground-truth data = cooper-group-uol-robotics/data
python Tools/tests/validate_screening.py  /path/to/cooper-group-uol-robotics/data
python Tools/tests/validate_ingestion.py  /path/to/cooper-group-uol-robotics/data
```

## Notes / scope
- **MS raw parsing** (`.RAW` → hits) stays vendor-locked to the Waters MassLynx
  SDK (Windows-only). We validate the **quorum heuristic** against the published
  `mz_peaks`; `match_ms_hits` covers the portable matching if observed m/z lists
  (e.g. mzML) become available. MS is validation-only (M2); NMR is the spine that
  transfers to M4.
- The peak picker is noise-aware (`height = max(cy/mi floor, 5σ)`, `prominence
  = 3σ`, robust MAD σ from signal-free windows). Calibrated on this ground truth;
  stable for noise multipliers 3–8 (not a knife-edge fit).
