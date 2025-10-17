# -----------------------------------------------------------------------------------------------------------------------
# compute-post-metrics_SPI.py 
# To calculate Spectral Power Index (SPI) of pressure on vessel wall from Oasis/BSLSolver CFD outputs.
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
#   - nproc        Number of worker processes (default: #logical CPUs)
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

import os
import sys
import gc
import glob
import h5py
import warnings
import argparse
import multiprocessing as mp
from multiprocessing import sharedctypes
from pathlib import Path

import vtk
import numpy as np
from numpy.fft import fftfreq, fft
import pyvista as pv


warnings.filterwarnings("ignore", category=DeprecationWarning) 


# -------------------------------- Shared-memory Helper Functions ---------------------------------------------


def create_shared_var(size, dtype=np.float64):
    """Create a ctypes-backed shared array and return the raw shared object."""
    a = np.ctypeslib.as_ctypes( np.zeros(size, dtype=dtype) )
    return sharedctypes.Array(a._type_, a,  lock=False)

def get_shared_var(a):
    """Return a shared ctypes array from an existing ctypes array."""
    return sharedctypes.Array(a._type_, a,  lock=False)

def get_array(shared_var):
    """Create a NumPy view on top of the shared ctypes array."""
    return np.ctypeslib.as_array(shared_var)


# ---------------------------------------- Mesh I/O -----------------------------------------------------

def assemble_mesh(mesh_file):
    """
    Create a PolyData surface (UnstructuredGrid) from the wall mesh stored in a BSLSolver-style HDF5.

    Expects datasets:
      Mesh/Wall/coordinates : (Npoints, 3) float
      Mesh/Wall/topology    : (Ncells, 3 or 4) int (triangles expected)
      Mesh/Wall/pointIds    : (Npoints,) int  (mapping back to volume point numbering)
    """
    
    with h5py.File(mesh_file, 'r') as hf:
        coords    = np.array(hf['Mesh/Wall/coordinates'])  # coords of wall points 
        elems     = np.array(hf['Mesh/Wall/topology'])     # connectivity of wall points
        point_ids = np.array(hf['Mesh/Wall/pointIds'])     # mapping to volume point IDs
        
    # Create VTK connectivity ()
    n_elems = elems.shape[0]
    elem_type = np.ones((n_elems, 1), dtype=int) * 3
    elems = np.concatenate([elem_type, elems], axis = 1)
        
    # Build surface and attach point IDs
    surf = pv.PolyData(coords, elems.ravel())
    surf.point_data['vtkOriginalPtIds'] = point_ids

    return surf

def get_ts(h5_file):
        """ Given a simulation h5_file, get ts. """
        return int(h5_file.stem.split('_ts=')[1].split('_')[0])


# --------------------------------- Parallel file reading -----------------------------------------------

# Read all h5 files
def read_h5_files(ids, wallids, h5_files, _press_):
    """
    Read a chunk of time files and place wall pressures into the shared 2D array.

    Arguments:
      ids      : list of snapshot indices to read
      wallids  : wall point indices (to slice pressure field)
      h5_files : list of Path objects to HDF5 snapshots
      _press_  : shared ctypes array; viewed as press_[n_points, n_times]
    """
    press_ = get_array(_press_)
    for i in ids:
        with h5py.File(h5_files[i], 'r') as hw:
            pn = np.array(hw['Solution']['p'])[wallids]
            pn = pn.flatten()
            # Write each wall-point value into column 'i' (time index i)
            for j in range(pn.shape[0]):
                press_[j][i] = pn[j]


# ---------------------------------------- Compute SPI -----------------------------------------------------
################################################################
def filter_SPI(U, W_low_cut, tag):
    """
    Compute SPI for one time series U using frequency masks in W_low_cut.

    Behavior:
      - If tag == "withmean": use raw U
        else: subtract mean before FFT.
      - Zero DC bin:          U_fft[W_low_cut[0]] = 0
      - Zero <25 Hz bins:     U_fft_25Hz[W_low_cut[1]] = 0
      - SPI = Power(>=25Hz) / Power(non-DC)
    """
    #for HI
    if tag=="withmean":
        U_fft = fft(U)
    else:
        U_fft = fft(U-np.mean(U))

    # Filter any amplitude corresponding to frequency equal to 0Hz
    U_fft[W_low_cut[0]] = 0

    # Filter any amplitude corresponding to frequency lower to 25Hz
    U_fft_25Hz = U_fft.copy()
    U_fft_25Hz[W_low_cut[1]] = 0

    # Compute the absolute value (power)
    Power_25Hz = np.sum ( np.power( np.absolute(U_fft_25Hz),2))
    Power_0Hz  = np.sum ( np.power( np.absolute(U_fft     ),2))
    if Power_0Hz < 1e-5:
        return 0
    return Power_25Hz/Power_0Hz


################################################################
def compute_SPI(ids,_press,_SPI,W_low_cut):
    """Compute SPI for a subset of wall points (ids) and write into shared SPI array."""
    
    press_ = get_array(_press)
    SPI = get_array(_SPI)

    print ('    working on', len(ids), 'points:', ids)

    for j in ids:
        SPI[j] = filter_SPI(press_[j], W_low_cut, "withoutmean")




# ---------------------------------------- Compute Hemodynamics Metrics -------------------------------------

def hemodynamics(surf, input_folder, outfolder, case_name, nproc, period):
    """Main driver: read pressures, compute SPI, and write a VTP.

    Args:
      surf         : PyVista PolyData for the wall surface
      input_folder : folder containing '*_curcyc_*up.h5' snapshots
      outfolder    : output folder for VTP
      case_name    : prefix for output filename
      nproc        : number of processes for parallel file reading
      period       : physical period of the cycle (seconds)
    """

    # 1) Find & sort snapshot files by timestep
    h5_files = sorted(Path(input_folder).glob('*_curcyc_*up.h5'), key  get_ts)
    
    id_start = 34000
    id_end   = 35000
    h5_files = h5_files[id_start:id_end-1] # timesteps: 26155-27155

    file_count = len(h5_files)
    if file_count==0:
        print('No files found in {}!'.format(input_folder))
        sys.exit()
    
    # Wall point IDs and sizes
    wall_ids = surf.point_data['vtkOriginalPtIds']
    number_of_points = len(surf.points)

    print ('   found', file_count, 'pressure files for', number_of_points, 'points.')
    #print ('Allocating %5.2f'%(number_of_points*file_count*8/(1024*1024))+'MB of memory ...', end='')

    # 2) Shared array to hold pressures: shape [n_points, n_times]
    press_ = create_shared_var([number_of_points, file_count])

    #print ('done.', flush=True)

    # 3) Parallel read: split file indices into groups and spawn processes
    print ('Reading in parallel', file_count, 'files into 1 array of shape', [number_of_points, file_count],' ...')#, end='')

    # make group and divide the procedure
    step = max(int(file_count / nproc), 1)
    rng = list(range(0,file_count))
    groups = [rng[i:i+step] for i  in range(rng[0], rng[-1]+1, step)]
    
    p_list=[]
    for i,g in enumerate(groups):
        p = mp.Process(target=read_h5_files, name='Process'+str(i), args=(g,wall_ids,h5_files,press_))
        p_list.append(p)

    for p in p_list: p.start()

    # Wait for all the processes to finish
    for p in p_list: p.join()
    gc.collect()

    #print (' done.', flush=True)
    
    # 4) Build frequency axis
    dt = period/10000 #float(file_count-1)
    W = fftfreq(file_count, d=dt)
    
    # Find indices for DC and <25 Hz
    W_low_cut = np.where( np.abs(W) == 0 ) + np.where( np.abs(W) < 25.0 )

    print ('period=',period, '  file_count=',file_count, '  dt=', dt, '  frequencies=',W)
    print ('max frequency=',np.max(W))
    print ('min frequency=',np.min(W))

    np.set_printoptions(threshold=sys.maxsize)
    print ( 'w_low_cut:', W_low_cut )

    # 5) Compute SPI per point
    print ('Now computing  SPI ...')

    SPI = create_shared_var([number_of_points])

    # make group and divide the procedure
    step = max(int(number_of_points / nproc), 1)
    rng = list(range(0,number_of_points))
    groups = [rng[i:i+step] for i  in range(rng[0], rng[-1]+1, step)]

    for i,g in enumerate(groups):
        compute_SPI(g,press_,SPI, W_low_cut)

    gc.collect()

    # 6) Attach SPI to surface and save VTP
    SPI = get_array(SPI)
    #print ('done.', flush=True)
    #print to file
    surf.point_data['SPI_p'] = SPI
    surf.save(f'{outfolder}/{case_name}_SPIp_tstep{id_start}to{id_end}.vtp')
    #print (' done.')




# ---------------------------------------- Run the script -----------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute SPI pressure on wall from Oasis HDF5 snapshots."
    )
    ap.add_argument("--input_folder",  required=True, help="Results folder with CFD .h5 files")
    ap.add_argument("--mesh_folder",   required=True, help="Case mesh folder containing HDF5 mesh file")
    ap.add_argument("--case_name",     required=True, help="Case name")
    ap.add_argument("--output_folder", required=True, help="Output directory for SPI VTP file")
    ap.add_argument("--period",        type=float,    help="Period in seconds (default: 0.915)", default=0.915)
    ap.add_argument("--nproc",         type=int,      help="Number of parallel processes", default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--f-cut",         type=float,    help="Cutoff frequency in Hz", default=25.0, dest="f_cut", )
    ap.add_argument("--with-mean",     action="store_true", help="Keep mean in FFT (default subtract mean)")
    return ap.parse_args()


def main():
    args          = parse_args()
    input_folder  = Path(args.input_folder)
    mesh_folder   = Path(args.mesh_folder)
    output_folder = Path(args.output_folder)

    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    mesh_file = list(Path(mesh_folder).glob('*.h5'))[0]
    print(f"Loading wall mesh: {mesh_file}")
    surf = assemble_mesh(mesh_file)

    print(f"Processing on {args.nproc} processes…")

    #print ('Performing hemodynamics computation on %d core%s.'%(ncore,'s' if ncore>1 else '') )
    hemodynamics(surf, input_folder, output_folder, case_name=args.case_name, nproc=args.nproc, period=args.period)





if __name__ == '__main__':
    main()
