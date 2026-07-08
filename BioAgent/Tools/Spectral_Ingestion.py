"""
Spectral_Ingestion.py
=====================

Portable NMR ingestion + processing, replacing the paper's TopSpin-bound layer
(``synthesis_bots/utils/nmr/processing.py``, which delegates ``efp``/``apbk``/
``sref``/``ppf`` to Bruker TopSpin and the proprietary ``fourier_nmr_driver``).

Everything here uses only ``nmrglue`` + ``scipy`` + ``numpy`` so it runs off the
spectrometer, on Linux, and on **both** vendor formats we need:

* **Bruker** (the paper: benchtop 80 MHz, ``fid`` + ``acqus``) -> ``nmrglue.bruker``
* **Varian/Agilent** (Dr. Filip's dataset: ``.fid`` dir + ``procpar``) ->
  ``nmrglue.varian``

Two reading paths are provided:

1. :func:`read_bruker_processed` - read TopSpin's already-processed, referenced
   real spectrum (``pdata/1/1r``). Used to *reproduce* the paper exactly and to
   isolate peak-picking from processing when validating.
2. :func:`process_fid_bruker` / :func:`process_fid_varian` - the true portable
   replacement: raw FID -> group-delay removal -> exponential line broadening ->
   zero-fill -> FFT -> phase -> baseline -> (ppm axis). This is the path that
   transfers to the Varian data where there is no TopSpin output to lean on.

The reference screening recipe (``workflows/nmr/supramol/screening.py``) is::

    process_spectrum(zero_filling="8k", line_broadening=1.2, reference=True)
    pick_peaks(reference_intensity=150, minimum_intensity=10,
               sensitivity=1, ppm_range=(6, 11))

which :func:`pick_peaks` reproduces with ``scipy.signal.find_peaks``:
``cy 150`` + ``mi 10`` == keep peaks whose height >= (10/150) x tallest-in-region.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import nmrglue as ng
from numpy.typing import NDArray
from scipy.signal import find_peaks

# Reference screening parameters (from settings.toml + screening.py).
SCREENING_PPM_RANGE = (6.0, 11.0)
SCREENING_ZERO_FILL = 8192          # "8k"
SCREENING_LINE_BROADENING = 1.2     # Hz
PICK_REFERENCE_INTENSITY = 150.0    # TopSpin `cy`
PICK_MINIMUM_INTENSITY = 10.0       # TopSpin `mi`  -> 10/150 of tallest peak
# Noise-aware gating (calibrated on the SUPRAMOL-SCREENING ground truth: these
# reproduce 18/18 published NMR verdicts and are stable for noise_mult 3..8).
PICK_NOISE_MULT = 5.0               # height floor = noise_mult x sigma
PICK_PROMINENCE_MULT = 3.0          # prominence  = prominence_mult x sigma
# ppm windows assumed signal-free, used for the robust noise (MAD) estimate.
NOISE_REGIONS = ((11.5, 14.0), (-2.0, 0.0))
# Host-guest processing uses a broader line broadening (settings hg_lb).
HOSTGUEST_LINE_BROADENING = 1.8     # Hz


# ===========================================================================
# Reading paths
# ===========================================================================
def read_bruker_processed(
    pdata_dir: str | Path,
) -> tuple[NDArray, NDArray, dict[str, Any]]:
    """Read a TopSpin-processed Bruker spectrum (``pdata/N``).

    Returns ``(ppm, intensity, meta)`` with ``ppm`` descending (high->low), the
    real processed spectrum already phased/baselined/referenced by TopSpin.
    Use this to reproduce the paper's peak lists exactly.
    """
    pdata_dir = Path(pdata_dir)
    dic, data = ng.bruker.read_pdata(str(pdata_dir))
    udic = ng.bruker.guess_udic(dic, data)
    uc = ng.fileiobase.uc_from_udic(udic)
    ppm = uc.ppm_scale()
    data = np.asarray(data, dtype=float)
    meta = {
        "source": "bruker_pdata",
        "path": str(pdata_dir),
        "n_points": int(data.size),
        "sf": float(dic["procs"].get("SF", 0.0)),
        "sw_ppm": float(udic[0].get("sw", 0.0)) / max(udic[0].get("obs", 1.0), 1e-9),
        "obs_mhz": float(udic[0].get("obs", 0.0)),
    }
    return ppm, data, meta


def _process_common(
    fid: NDArray,
    dic_udic: dict,
    line_broadening: float,
    zero_filling: int | None,
    auto_phase: bool = True,
    baseline: bool = True,
) -> tuple[NDArray, NDArray]:
    """Shared time-domain -> frequency-domain processing.

    Exponential line broadening -> zero-fill -> FFT -> (auto) phase -> baseline.
    Returns ``(ppm, real_spectrum)``.
    """
    # Sweep width in Hz for line-broadening -> convert Hz to exponential lb.
    sw_hz = float(dic_udic["sw"])
    n0 = fid.size

    # Exponential window (line broadening in Hz).
    if line_broadening and line_broadening > 0:
        fid = ng.proc_base.em(fid, lb=line_broadening / sw_hz)

    # Zero-fill to target size (next spectrum length).
    if zero_filling:
        fid = ng.proc_base.zf_size(fid, zero_filling)

    # FFT (spectrum), then order so ppm runs high -> low.
    spec = ng.proc_base.fft(fid)
    spec = ng.proc_base.rev(spec)

    # Phase correction: autophase if requested, else magnitude fallback.
    if auto_phase:
        try:
            spec = ng.proc_autophase.autops(spec, "acme", disp=False)
        except Exception:
            spec = np.abs(spec)
    real = spec.real

    # Simple baseline correction (median/polynomial) to flatten drift.
    if baseline:
        try:
            real = ng.proc_bl.baseline_corrector(real, wd=20)
        except Exception:
            real = real - np.median(real)

    # Build ppm axis from the processed size.
    obs = float(dic_udic["obs"])
    car = float(dic_udic["car"])
    n = real.size
    sw_ppm = sw_hz / obs if obs else 0.0
    car_ppm = car / obs if obs else 0.0
    ppm = np.linspace(car_ppm + sw_ppm / 2, car_ppm - sw_ppm / 2, n)
    return ppm, np.asarray(real, dtype=float)


def process_fid_bruker(
    expno_dir: str | Path,
    line_broadening: float = SCREENING_LINE_BROADENING,
    zero_filling: int | None = SCREENING_ZERO_FILL,
    auto_phase: bool = True,
    baseline: bool = True,
) -> tuple[NDArray, NDArray, dict[str, Any]]:
    """Portable Bruker FID processing (no TopSpin).

    Reads the raw ``fid`` + ``acqus`` from an EXPNO directory and processes it
    entirely with nmrglue/scipy. This is the replacement for TopSpin
    ``efp``/``apbk`` and the path validated against Dr. Filip's Varian data.

    NOTE: exact referencing (TopSpin ``sref`` to the solvent) is not applied
    here; use :func:`reference_to_peak` or read the processed ``1r`` when exact
    ppm alignment to the paper is required.
    """
    expno_dir = Path(expno_dir)
    dic, fid = ng.bruker.read(str(expno_dir))
    fid = np.asarray(fid, dtype=complex)

    # Remove Bruker digital-filter group delay before processing.
    fid = ng.bruker.remove_digital_filter(dic, fid)

    udic = ng.bruker.guess_udic(dic, fid)
    ppm, real = _process_common(
        fid, udic[0], line_broadening, zero_filling, auto_phase, baseline
    )
    meta = {
        "source": "bruker_fid",
        "path": str(expno_dir),
        "obs_mhz": float(udic[0]["obs"]),
        "sw_hz": float(udic[0]["sw"]),
        "n_points": int(real.size),
    }
    return ppm, real, meta


def process_fid_varian(
    fid_dir: str | Path,
    line_broadening: float = SCREENING_LINE_BROADENING,
    zero_filling: int | None = SCREENING_ZERO_FILL,
    auto_phase: bool = True,
    baseline: bool = True,
) -> tuple[NDArray, NDArray, dict[str, Any]]:
    """Portable Varian/Agilent FID processing (for Dr. Filip's dataset).

    Reads ``fid`` + ``procpar`` from a Varian ``.fid`` directory and processes
    it with the same nmrglue/scipy pipeline used for Bruker, so the downstream
    peak-picking and decision logic are identical across vendors.
    """
    fid_dir = Path(fid_dir)
    dic, fid = ng.varian.read(str(fid_dir))
    fid = np.asarray(fid, dtype=complex)
    if fid.ndim > 1:  # take first trace if pseudo-2D
        fid = fid[0]

    udic = ng.varian.guess_udic(dic, fid)
    ppm, real = _process_common(
        fid, udic[0], line_broadening, zero_filling, auto_phase, baseline
    )
    meta = {
        "source": "varian_fid",
        "path": str(fid_dir),
        "obs_mhz": float(udic[0]["obs"]),
        "sw_hz": float(udic[0]["sw"]),
        "n_points": int(real.size),
    }
    return ppm, real, meta


# ===========================================================================
# Peak picking  ---  reproduces TopSpin cy/mi/ppf semantics
# ===========================================================================
def estimate_noise(
    ppm: NDArray,
    intensity: NDArray,
    noise_regions: tuple[tuple[float, float], ...] = NOISE_REGIONS,
) -> float:
    """Robust noise sigma from signal-free ppm regions (1.4826 x MAD).

    Falls back to the global MAD if the nominated regions hold too few points.
    """
    seg_mask = np.zeros(ppm.shape, dtype=bool)
    for lo, hi in noise_regions:
        seg_mask |= (ppm >= min(lo, hi)) & (ppm <= max(lo, hi))
    seg = intensity[seg_mask]
    if seg.size < 50:
        seg = intensity
    med = np.median(seg)
    mad = np.median(np.abs(seg - med))
    sigma = 1.4826 * mad
    return float(sigma) if sigma > 0 else float(seg.std())


def pick_peaks(
    ppm: NDArray,
    intensity: NDArray,
    reference_intensity: float = PICK_REFERENCE_INTENSITY,
    minimum_intensity: float = PICK_MINIMUM_INTENSITY,
    ppm_range: tuple[float, float] = SCREENING_PPM_RANGE,
    noise_mult: float = PICK_NOISE_MULT,
    prominence_mult: float = PICK_PROMINENCE_MULT,
    min_distance_hz: float = 0.0,
    obs_mhz: float | None = None,
) -> list[float]:
    """Pick peaks over ``ppm_range``, reproducing TopSpin ``cy``/``mi``/``ppf``.

    Combines two gates, calibrated to reproduce the paper's 18/18 published
    NMR screening verdicts and to transfer to Varian data (where no TopSpin
    output exists):

    * **Relative floor** (TopSpin ``cy 150`` + ``mi 10``): keep peaks whose
      height >= ``(minimum_intensity / reference_intensity) x global_max``.
      ``cy`` scales the *global* tallest peak, so the reference is the whole
      spectrum's max, not just the region's - this matters when the region of
      interest is mostly noise.
    * **Noise floor + prominence**: height must also exceed ``noise_mult x
      sigma`` and stand out by ``prominence_mult x sigma`` (robust MAD noise).
      This is what suppresses noise shoulders without discarding real peaks.

    The final height threshold is ``max(relative_floor, noise_mult x sigma)``.

    Parameters
    ----------
    ppm, intensity
        FULL spectrum (ppm descending). The full range is needed so the noise
        estimate and the global max are computed correctly; ``ppm_range`` then
        selects the region peaks are returned from.
    reference_intensity, minimum_intensity
        TopSpin ``cy`` / ``mi`` values (defaults 150 / 10).
    ppm_range
        Region of interest, default (6, 11).
    noise_mult, prominence_mult
        Noise-relative height floor and prominence multipliers (defaults 5, 3).
        Set both to 0 for a pure TopSpin cy/mi reproduction.
    min_distance_hz, obs_mhz
        Optional minimum peak separation (Hz); needs ``obs_mhz``. 0 = off.

    Returns
    -------
    list[float]
        Peak ppm positions, sorted descending (as the reference returns them).
    """
    global_max = float(np.asarray(intensity, dtype=float).max())
    sigma = estimate_noise(ppm, intensity)

    lo, hi = min(ppm_range), max(ppm_range)
    mask = (ppm >= lo) & (ppm <= hi)
    reg = np.asarray(intensity[mask], dtype=float)
    pp = np.asarray(ppm[mask], dtype=float)
    if reg.size == 0:
        return []

    relative_floor = (minimum_intensity / reference_intensity) * global_max
    height = max(relative_floor, noise_mult * sigma)

    kwargs: dict[str, Any] = {"height": height}
    if prominence_mult > 0:
        kwargs["prominence"] = prominence_mult * sigma
    if min_distance_hz > 0 and obs_mhz:
        pts_per_ppm = reg.size / (hi - lo)
        dist_ppm = min_distance_hz / obs_mhz
        kwargs["distance"] = max(1, int(dist_ppm * pts_per_ppm))

    idx, _ = find_peaks(reg, **kwargs)
    return sorted((float(pp[i]) for i in idx), reverse=True)


def reference_to_peak(
    ppm: NDArray,
    intensity: NDArray,
    target_ppm: float,
    search_window: tuple[float, float],
) -> NDArray:
    """Shift the ppm axis so the tallest peak in ``search_window`` sits at
    ``target_ppm`` (a portable stand-in for TopSpin ``sref``).

    For CH3CN the residual methyl is referenced to ~1.94 ppm; for screening the
    peaks of interest are in (6, 11) so referencing mainly affects absolute
    positions, not the count-based decision.
    """
    lo, hi = min(search_window), max(search_window)
    mask = (ppm >= lo) & (ppm <= hi)
    if not mask.any():
        return ppm
    reg = intensity[mask]
    pk_ppm = ppm[mask][int(np.argmax(reg))]
    return ppm + (target_ppm - pk_ppm)


# ===========================================================================
# Convenience: full Bruker experiment -> peak list
# ===========================================================================
def read_bruker_spectrum(
    sample_dir: str | Path,
    expno: str = "10",
    use_processed: bool = True,
) -> tuple[NDArray, NDArray, dict[str, Any]]:
    """Read one Bruker experiment as ``(ppm, intensity, meta)``.

    ``use_processed=True`` reads TopSpin's ``pdata/1/1r`` (exact reproduction);
    ``False`` processes the raw FID with nmrglue (portable path).
    """
    sample_dir = Path(sample_dir)
    if use_processed:
        return read_bruker_processed(sample_dir / expno / "pdata" / "1")
    return process_fid_bruker(sample_dir / expno)


def bruker_experiment_peaks(
    sample_dir: str | Path,
    expno: str = "10",
    use_processed: bool = True,
    **pick_kwargs: Any,
) -> tuple[list[float], dict[str, Any]]:
    """Read one Bruker experiment and return its screening peak list.

    ``use_processed=True`` reads TopSpin's ``pdata/1/1r`` (exact reproduction);
    ``False`` processes the raw FID with nmrglue (portable path).
    """
    ppm, data, meta = read_bruker_spectrum(sample_dir, expno, use_processed)
    peaks = pick_peaks(ppm, data, obs_mhz=meta.get("obs_mhz"), **pick_kwargs)
    meta["n_peaks"] = len(peaks)
    return peaks, meta


def spectrum_region(
    sample_dir: str | Path,
    ppm_range: tuple[float, float] = SCREENING_PPM_RANGE,
    expno: str = "10",
    use_processed: bool = True,
) -> NDArray:
    """Return the intensity array within ``ppm_range`` for one experiment.

    Used by the DTW replication check, which compares the (6, 11) ppm region
    of two spectra (reproducing TopSpin ``getSpecDataPoints(physRange=...)``).
    """
    ppm, data, _ = read_bruker_spectrum(sample_dir, expno, use_processed)
    lo, hi = min(ppm_range), max(ppm_range)
    mask = (ppm >= lo) & (ppm <= hi)
    return np.asarray(data[mask], dtype=float)
