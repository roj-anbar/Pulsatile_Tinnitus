# -----------------------------------------------------------------------------------------------------------------------
# compute-post-metrics_SPI.py 
# To calculate windowed Spectral Power Index (SPI) of pressure on vessel wall from Oasis/BSLSolver CFD outputs.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2025-10
#
# PURPOSE:
#   - This script is part of the BSL post-processing pipeline.
#   - It computes windowed SPI for pressure at each wall point and saves the resultant array 'SPI_p' on the surface to a VTP file.
#
# REQUIREMENTS:
#   - h5py
#   - pyvista
#   - On Trillium: virtual environment called pyvista36
#
# EXECUTION:
#   - Run using "compute-post-metrics_RUNME.sh" bash script.
#   - Run directly on a login/debug node as below:
#       > module load StdEnv/2023 gcc/12.3 python/3.12.4
#       > source $HOME/virtual_envs/pyvista36/bin/activate
#       > module load  vtk/9.3.0
#       > python compute-post-metrics_SPI.py <path_to_CFD_results_folder> <path_to_case_mesh_data> <case_name> <path_to_output_folder> <period> <ncores> <f_cutoff> <flag_mean>
#     
#
# INPUTS:
#   - folder       Path to results directory with HDF5 snapshots
#   - mesh         Path to the case mesh data files (with Mesh/Wall/{coordinates,topology,pointIds})
#   - case         Case name (used only in output filename)
#   - out          Output directory for the VTP result
#   - period       Period in seconds (e.g., 0.915)
#   - n_process        Number of worker processes (default: #logical CPUs)
#   - f-cut        Frequency cutoff in Hz (default: 25.0)
#   - with-mean    Keep the mean when computing FFT (default: subtract mean)
#
#
# OUTPUTS:
#   - output_folder/<case>_SPIp_<f-cut>Hz.vtp  PolyData with 'SPI_p' point-data
#
# NOTES:
#   - Time step dt is inferred from: dt = period / (N-1) (one period covered by N files).
#
#
# Adapted from hemodynamic_indices_pressure.py originally written by Anna Haley (2024). 
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------

#import os
import sys
import gc
#import glob
import h5py
import warnings
import argparse
from pathlib import Path
import multiprocessing as mp
from multiprocessing import sharedctypes

import vtk
import numpy as np
from numpy.fft import fftfreq, fft
import pyvista as pv

from post_utils_parallel import * #(create_shared_array, np_shared_array)
from post_utils_readerH5 import * #(assemble_wall_mesh, read_h5_files, extract_timestep_from_h5)
from post_utils_SPI import (generate_windows, compute_SPI)

warnings.filterwarnings("ignore", category=DeprecationWarning) 



# ---------------------------------------- Compute Hemodynamics Metrics -------------------------------------

def hemodynamics(surf: pv.PolyData,
                 input_folder: Path,
                 output_folder: Path,
                 case_name: str,
                 n_process: int,
                 period_seconds: float,
                 window_size: int,
                 with_mean: bool,
                 window_overlap_frac: float = 0.25,
                 freq_cut: float=25.0
                 ):
    """
    Main driver: Reads time-series pressures, computes windowed SPI, and writes it to a VTP file.

    Args:
      surf           : PyVista PolyData for the wall surface
      input_folder   : folder containing '*_curcyc_*up.h5' snapshots
      output_folder  : output folder for VTP
      case_name      : prefix for output filename
      n_process      : number of processes for parallel file reading
      period_seconds : physical period of the cycle (seconds)
      freq_cut       : cutoff frequency (Hz)
      window_size
      window_overlap_frac
    """

    # 1) Gather files
    # Find & sort snapshot files by timestep 
    snapshot_h5_files = sorted(Path(input_folder).glob('*_curcyc_*up.h5'), key = extract_timestep_from_h5)
    n_snapshots = len(snapshot_h5_files)

    if n_snapshots==0:
        print('No files found in {}!'.format(input_folder))
        sys.exit()
    
    # Wall point IDs and sizes
    wall_pids = surf.point_data['vtkOriginalPtIds']
    n_points  = len(surf.points)


    # 2) Create shared arrays
    # Array to hold pressures (n_points, n_times)
    shared_pressure_ctype = create_shared_array([n_points, n_snapshots])



    # 3) Parallel reading
    print ('Reading in parallel', n_snapshots, 'files into 1 array of shape', [n_points, n_snapshots],' ...')

    # divide all snapshot files into groups and spread across processes
    time_indices    = list(range(n_snapshots))
    time_chunk_size = max(n_snapshots // n_process, 1)
    time_groups     = [time_indices[i : i + time_chunk_size] for i in range(0, n_snapshots, time_chunk_size)]
    
    processes_list=[]
    for idx, group in enumerate(time_groups):
        proc = mp.Process(target = read_h5_files, name=f"Reader{idx}", args=(group, wall_pids, snapshot_h5_files, shared_pressure_ctype))
        processes_list.append(proc)

    # Start all readers
    for proc in processes_list:
        proc.start()

    # Wait for all readers to finish
    for proc in processes_list:
        proc.join()

    # Free up memory
    gc.collect()



    # 4) Compute windowed SPI 
    print ('Now computing windowed SPI in parallel ...')

    shared_SPI_ctype = create_shared_array([n_points,]) # 1D

    # Generate windows 
    if window_size is None:
        window_size = n_snapshots  # use entire signal

    windows = generate_windows(n_snapshots, window_size, window_overlap_frac)
    print(f"Generated {len(windows)} windows of size {window_size} with {window_overlap_frac*100:.0f}% overlap.\n")


    # Loop over windows
    for window_idx, (start_idx, end_idx) in enumerate(windows, 1):
        print(f"Computing SPI for window {window_idx}/{len(windows)}: frames {start_idx}–{end_idx}")

        # Set up frequencies
        dt_seconds = period_seconds/10000 #float(n_snapshots-1)

        #if window_size: # windowed fft
        freqs = fftfreq(end_idx-start_idx, d = dt_seconds) # [Hz]
        #else: # entire signal
        #    freqs = fftfreq(n_snapshots, d = dt_seconds) # [Hz]
        # Find indices for DC and <25 Hz
        ind_freq_zero = np.where( np.abs(freqs) == 0 )
        ind_freq_below_cutoff = np.where( np.abs(freqs) < freq_cut)

        
        if window_idx == 1:
            print ('period (s)=', period_seconds, '  n_snapshots=', n_snapshots, '  dt (s)=', dt_seconds) # '  frequencies=',freqs
            print (f'min/max frequency= {np.min(freqs):.2f} / {np.max(freqs):.2f}')

  
        # divide all snapshot files into chunks
        point_indices    = list(range(n_points))
        point_chunk_size = max(int(n_points // n_process), 1)
        point_groups     = [point_indices[i : i + point_chunk_size] for i in range(0, n_points, point_chunk_size)]

        for group in point_groups:
            compute_SPI(group, shared_pressure_ctype, shared_SPI_ctype, ind_freq_zero, ind_freq_below_cutoff, start_idx, end_idx, with_mean)

        # Free up memory
        gc.collect()
        
        # 6) Attach SPI to surface and save VTP
        SPI = np_shared_array(shared_SPI_ctype)
        surf.point_data['SPI_p'] = SPI #SPI.astype(np.float32, copy=False)

        output_file = Path(output_folder) / f"{case_name}_SPIp_win{window_idx:03d}.vtp"

        surf.save(str(output_file))
    
    print (f'Finished writing windowed SPI files to {output_folder}.')




# ---------------------------------------- Run the script -----------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_folder",  required=True,       help="Results folder with CFD .h5 files")
    ap.add_argument("--mesh_folder",   required=True,       help="Case mesh folder containing HDF5 mesh file")
    ap.add_argument("--case_name",     required=True,       help="Case name")
    ap.add_argument("--output_folder", required=True,       help="Output directory for SPI VTP file")
    ap.add_argument("--period",        type=float,          help="Period in seconds (default: 0.915)", default=0.915)
    ap.add_argument("--window_size",   type=int,            help="Size of window (number of snapshots for each window)")
    ap.add_argument("--window_overlap",type=float,          help="Fraction of overlap between windows (0 to 1)", default=0)
    ap.add_argument("--n_process",     type=int,            help="Number of parallel processes", default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--freq_cut",      type=float,          help="Cutoff frequency in Hz", default=25.0)
    ap.add_argument("--with_mean",     action="store_true", help="Flag to keep mean in FFT or not")
    
    return ap.parse_args()


def main():
    args          = parse_args()
    input_folder  = Path(args.input_folder)
    mesh_folder   = Path(args.mesh_folder)
    output_folder = Path(args.output_folder)

    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    mesh_file = list(Path(mesh_folder).glob('*.h5'))[0]
    print(f"Loading mesh: {mesh_file}")
    surf = assemble_wall_mesh(mesh_file)

    print(f"Processing on {args.n_process} processes…")

    #print ('Performing hemodynamics computation on %d core%s.'%(ncore,'s' if ncore>1 else '') )
    hemodynamics(surf, input_folder, output_folder, case_name = args.case_name,
                 n_process = args.n_process, period_seconds = args.period, freq_cut = args.freq_cut,
                 window_size = args.window_size, window_overlap_frac = args.window_overlap, with_mean = args.with_mean)



if __name__ == '__main__':
    main()
