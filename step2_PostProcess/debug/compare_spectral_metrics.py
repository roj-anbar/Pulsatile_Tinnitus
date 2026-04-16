# -----------------------------------------------------------------------------------------------------------------------
# plot_banded_powers_SSS_vs_TS.py
# Standalone debug script to compare banded SPL powers between two anatomical regions (SSS and TS).
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2026-04
#
# PURPOSE:
#   - Reads spectrogram .npz files produced by compute_Spectrogram.py for two regions (SSS and TS).
#   - Filters spectrogram to the analysis window (frequency range + Q_inlet range).
#   - Computes column-wise mean banded SPL power (low / mid / high frequency bands).
#   - Plots all three bands in a 3-panel figure with SSS and TS overlaid for direct comparison.
#
# REQUIREMENTS:
#   - numpy, matplotlib
#
# EXECUTION:
#    - Execution environment: LOCAL (not on HPC)
#    - Run on a local workstation with access to .npz data files via Dropbox.
#
# USAGE:
#    - python plot_banded_powers_SSS_vs_TS.py
#    - Adjust PATH_DATA, FILE_SSS, FILE_TS, frequency bands, and Q window as needed.
#
# INPUT FILE FORMAT:
#    .npz files saved by compute_Spectrogram.py via:
#        np.savez(path, spectrogram_data)  --> loaded as data['arr_0'].item()
#    The dict contains:
#        'power_avg_dB'  : (n_freq, n_frames) — mean spectrogram in dB (SPL re 20 µPa)
#        'bins'          : (n_frames,)         — time axis in seconds
#        'freqs'         : (n_freq,)           — frequency axis in Hz
# -----------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

PATH_DATA = "/Users/BSL/Dropbox/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step2_PostProcess/Spectrogram_WallPressure/run4_specMetrics/window2732_overlap0.9_ROIcylinder_multiROITrue/files"

FILE_SSS = "PTSeg028_base_0p64_win2732_region1_SSS.npz"
FILE_TS  = "PTSeg028_base_0p64_win2732_region2_TS.npz"

# Frequency band boundaries (Hz) — must match values used in compute_Spectrogram.py
f_low = 100    # low / mid boundary
f_mid = 1000   # mid / high boundary
f_max = 5000   # upper cutoff

# Analysis window in Q_inlet (mL/s), where Q_inlet = 2 * t for PT-Ramp protocol
Q_min = 2.0
Q_max = 10.0

# dB floor: values below this are clamped (should match --cutoff_db in compute_Spectrogram.py)
cutoff_db = 0.0

# Plot style
CLR_SSS, CLR_TS = 'darkgreen', 'limegreen'
LW = 2.5


# ── Load ──────────────────────────────────────────────────────────────────────

def load_spectrogram_npz(filepath: Path) -> dict:
    """
    Load a spectrogram .npz file saved by compute_Spectrogram.py.
    Returns the spectrogram dict with keys: power_avg_dB, bins, freqs, ...
    """
    raw = np.load(filepath, allow_pickle=True)
    return raw['arr_0'].item()   # 0-d object array wrapping the dict


spec_SSS = load_spectrogram_npz(Path(PATH_DATA) / FILE_SSS)
spec_TS  = load_spectrogram_npz(Path(PATH_DATA) / FILE_TS)


# ── Filter & compute banded powers ───────────────────────────────────────────

def compute_banded_powers(spec: dict, f_low: float, f_mid: float, f_max: float,
                          Q_min: float, Q_max: float, cutoff_db: float):
    """
    Filter a spectrogram to the analysis window and compute column-wise mean
    SPL power for three frequency bands.

    Parameters
    ----------
    spec       : dict from load_spectrogram_npz
    f_low      : low / mid frequency boundary (Hz)
    f_mid      : mid / high frequency boundary (Hz)
    f_max      : upper frequency cutoff (Hz)
    Q_min/max  : Q_inlet analysis window (mL/s); converted from time via Q = 2*t
    cutoff_db  : dB floor applied before averaging

    Returns
    -------
    Q_axis     : (n_frames,)  — Q_inlet values for the filtered window
    low_power  : (n_frames,)  — mean SPL in the low  band (0 – f_low Hz)
    mid_power  : (n_frames,)  — mean SPL in the mid  band (f_low – f_mid Hz)
    high_power : (n_frames,)  — mean SPL in the high band (f_mid – f_max Hz)
    """
    freqs        = spec['freqs']
    bins         = spec['bins']
    power_avg_dB = spec['power_avg_dB']

    # Convert time axis to Q_inlet (ramp-specific: Q_inlet = 2*t)
    bins_Q = 2.0 * bins

    # Apply frequency and Q masks
    mask_freq = freqs <= f_max
    mask_Q    = (bins_Q >= Q_min) & (bins_Q <= Q_max)

    power_filt = power_avg_dB[np.ix_(mask_freq, mask_Q)].copy()
    power_filt[power_filt < cutoff_db] = cutoff_db

    freqs_filt = freqs[mask_freq]
    Q_axis     = bins_Q[mask_Q]

    # Band masks on the already-trimmed frequency axis
    mask_low  =  freqs_filt < f_low
    mask_mid  = (freqs_filt >= f_low) & (freqs_filt < f_mid)
    mask_high = (freqs_filt >= f_mid) & (freqs_filt <= f_max)

    low_power  = np.mean(power_filt[mask_low,  :], axis=0)
    mid_power  = np.mean(power_filt[mask_mid,  :], axis=0)
    high_power = np.mean(power_filt[mask_high, :], axis=0)

    return Q_axis, low_power, mid_power, high_power


Q_SSS, low_SSS, mid_SSS, high_SSS = compute_banded_powers(
    spec_SSS, f_low, f_mid, f_max, Q_min, Q_max, cutoff_db)

Q_TS, low_TS, mid_TS, high_TS = compute_banded_powers(
    spec_TS, f_low, f_mid, f_max, Q_min, Q_max, cutoff_db)


# ── Plot ──────────────────────────────────────────────────────────────────────

bands = [
    (f"Low-freq  (< {f_low} Hz)",           low_SSS,  low_TS),
    (f"Mid-freq  ({f_low}–{f_mid} Hz)",     mid_SSS,  mid_TS),
    (f"High-freq ({f_mid}–{f_max} Hz)",     high_SSS, high_TS),
]

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Banded SPL Power: SSS vs. TS", fontsize=13, fontweight='bold')

for ax, (band_label, sss_vals, ts_vals) in zip(axes, bands):
    ax.plot(Q_SSS, sss_vals, color=CLR_SSS, linewidth=LW, label='SSS')
    ax.plot(Q_TS,  ts_vals,  color=CLR_TS,  linewidth=LW, label='TS', linestyle='--')
    ax.set_ylabel("Mean SPL (dB)", fontsize=10)
    ax.set_title(band_label, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("$Q_{inlet}$ (mL/s)", fontsize=11, fontweight='bold')
axes[-1].set_xlim([Q_min, Q_max])

plt.tight_layout()
plt.show()
