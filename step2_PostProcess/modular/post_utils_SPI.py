# -----------------------------------------------------------------------------------------------------------------------
# compute-post-metrics_SPI.py 
# Utilities to compute windowed Spectral Power Index (SPI) of pressure on vessel wall from Oasis/BSLSolver CFD outputs.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2025-10
#
# PURPOSE:
#   - This script is part of the BSL post-processing pipeline.
#   - It computes windowed SPI for pressure at each wall point and saves the resultant array 'SPI_p' on the surface to a VTP file.
#
# REQUIREMENTS:
#   - numpy
#
# EXECUTION:
#   - Used in compute-post-metrics.py script.  
#
# INPUTS:
#   - folder       Path to result
#
#
# OUTPUTS:
#   - output_folder/<case>_SPIp_<f-cut>Hz.vtp  PolyData with 'SPI_p' point-data
#
# NOTES:
#   - Time step dt is inferred from: dt = period / (N-1) (one period covered by N files).
#
#
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------


import sys
import gc
import warnings
from pathlib import Path
import numpy as np
from numpy.fft import fftfreq, fft
#from multiprocessing import sharedctypes

from post_utils_parallel import *

warnings.filterwarnings("ignore", category=DeprecationWarning) 




def generate_windows(n_snapshots, window_size, window_overlap_frac):
    """
    Generate (start, end) index pairs for windowed processing.

    window_overlap_frac : float in [0,1]  (fractional overlap)
    Returns list of (start_idx, end_idx)
    """
    if window_size > n_snapshots:
        return [(0, n_snapshots)]

    step = int(window_size * (1 - window_overlap_frac))
    windows = []
    start = 0
    while start + window_size <= n_snapshots:
        end = start + window_size
        windows.append((start, end))
        start += step

    # Add final window if partial remainder
    if windows[-1][1] < n_snapshots:
        windows.append((n_snapshots - window_size, n_snapshots))

    return windows


# ---------------------------------------- Compute SPI -----------------------------------------------------

def filter_SPI(signal, ind_freq_zero, ind_freq_below_cutoff, mean_tag):
    """
    Compute SPI for a single time series using frequency masks in W_low_cut.

    Arguments:
      signal                : 1D numpy array of length n_times (pressure vs time at a wall point)
      ind_freq_zero         : index of where frequency is zero --> np.where(|f| == 0)
      ind_freq_below_cutoff : index of where frequency is below cutoff freq --> np.where(|f| < f_cutoff)
      mean_tag              : choose between "withmean" or "withoutmean" -> use raw signal; otherwise subtract mean

    """

    # Subtract mean unless instructed otherwise
    if mean_tag == "withmean":
        fft_signal = fft(signal)
    else:
        fft_signal = fft(signal - np.mean(signal))

    # Filter any amplitude corresponding to frequency equal to 0Hz
    fft_signal_above_zero = fft_signal.copy()
    fft_signal_above_zero[ind_freq_zero] = 0

    # Filter any amplitude corresponding to frequency lower than cutoff frequecy (f_cutoff=25Hz)
    fft_signal_below_cutoff = fft_signal.copy()
    fft_signal_below_cutoff[ind_freq_below_cutoff] = 0

    # Compute the absolute value (power)
    Power_below_cutoff = np.sum ( np.power( np.absolute(fft_signal_below_cutoff),2))
    Power_above_zero   = np.sum ( np.power( np.absolute(fft_signal_above_zero),2))
    
    if Power_above_zero < 1e-5:
        return 0
    
    return Power_below_cutoff/Power_above_zero


def compute_SPI(pids, shared_pressure_ctype, shared_SPI_ctype,
                ind_freq_zero, ind_freq_below_cutoff, start_idx, end_idx, with_mean=False):
    """Computes SPI for the given points (pids) over time window [start_idx:end_idx] and writes back into shared SPI array."""
    
    pressure = np_shared_array(shared_pressure_ctype) # (n_points, n_times)
    SPI      = np_shared_array(shared_SPI_ctype)      # (n_points,)

    for point in pids:
        pressure_window = pressure[point, start_idx:end_idx]
        SPI[point] = filter_SPI(pressure_window, ind_freq_zero, ind_freq_below_cutoff, "withmean" if with_mean else "withoutmean")

