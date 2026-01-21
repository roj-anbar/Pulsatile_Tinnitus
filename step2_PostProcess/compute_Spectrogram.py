# -----------------------------------------------------------------------------------------------------------------------
# compute_Spectrograms.py 
# To compute the average power spectrogram in dB scale of pressure or velocity from Oasis/BSLSolver CFD outputs.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2025-10
#
# PURPOSE:
#   - This script is part of the BSL post-processing pipeline.
#   - Reads CFD HDF5 snapshots for all timesteps, extracts the quantity of interest (wall pressure/velocity) time-series,
#     computes ROI-averaged spectrograms, and saves both .npz data and .png images.
#
# REQUIREMENTS:
#   - h5py, pyvista, vtk, numpy, scipy, matplotlib
#   - On Trillium: virtual environment called "pyvista36"
#
# EXECUTION:
#   - Run using "compute_Spectrograms_job.sh" bash script.
#   - Run directly on a login/debug node as below:
#       > module load StdEnv/2023 gcc/12.3 python/3.12.4
#       > source $HOME/virtual_envs/pyvista36/bin/activate
#       > module load  vtk/9.3.0
#
# EXAMPLE CLI (with required arguments):
#       > python compute_Spectrograms.py \
#           --input_folder       <path_to_CFD_results_folder> \
#           --mesh_folder        <path_to_case_mesh_data>     \
#           --output_folder      <path_to_output_folder>      \
#           --case_name          PTSeg028_base_0p64           \
#           --ROI_center_csv     <path_to_ROI_CSV_file>       \
#           --ROI_radius         4.0                          \
#
#
# INPUTS:
#   - input_folder        Path to results directory with HDF5 snapshots
#   - mesh_folder         Path to the case mesh data files (with Mesh/Wall/{coordinates,topology,pointIds})
#   - case_name           Case name (used only in output filename)
#   - output_folder       Path to output directory to save results (will create subfolders files/, imgs/, ROIs/)
#   --period_seconds      Flow period [s] (if omitted, try to parse from filenames with '_Per<ms>')
#   --timesteps_per_cyc   Timesteps per cycle (if omitted, try to parse from filenames with '_ts<int>')
#   --density             Blood density [kg/m3] (default = 1050 kg/m3)
#   --spec_quantity       Quantity to compute spectrogram from: 'pressure' or 'velocity'
#   --ROI_type            (default = 'cylinder')
#   --ROI_center_coord    X Y Z center of spherical ROI (mesh units).
#   --ROI_center_csv      Path to CSV file containing the coordinates of multiple points for ROI center.
#   --ROI_radius          Sphere radius (mesh units). If 0, treat ROI_center as the **point ID** to sample.
#   --flag_save_ROI       Flag to save the ROI.vtp surface file (If it's included in args then it's True if not it's False)
#   --flag_multi_ROI      Flag to compute spectrogram in a segment based on multiple ROIs (If it's included in args then it's True if not it's False)
#   --window_length       STFT window length (samples, i.e., snapshots)
#   --n_fft               STFT FFT length (bins)
#   --overlap_frac        STFT noverlap = overlap_frac * window_length --> Overlap fraction between consequent windows (0-1)
#   --window              STFT window type
#   --pad_mode            Edge padding ('cycle','constant','odd','even','none')
#   --detrend             STFT detrend ('linear','constant', or False)
#   --cutoff_dB           Minimum threshold for calculated power in dB (e.g., 0) --> anything below that will be set to this value
#   --n_process           Number of worker processes (default: #logical CPUs)
#
# OUTPUTS:
#   1) Processed spectrograms saved as .npz (output_folder/files)
#   2) Spectrogram images saved as .png     (output_folder/imgs)
#   3) ROI surface file saved as .vtp       (output_folder/ROIs)
#
# NOTES:
#   - Sampling rate inferred as: fs = timesteps_per_cyc / period_seconds [Hz].
#   - Filename helpers expect snapshots containing '_curcyc_' and optionally '_ts<int>' and '_Per<ms>'.
#   - Pressure in Oasis is p/rho; multiply by density (default 1050 kg/m^3) to get Pa if desired.
#
# Adapted from BSL-tools repository (Dan Macdonald 2022) and make_wall_pressure_specs.py (Anna Haley 2024). 
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------

import sys
import gc
import h5py
import warnings
import argparse
from pathlib import Path
import multiprocessing as mp
from multiprocessing import sharedctypes


import vtk
import numpy as np
from scipy.signal import stft, find_peaks
import pyvista as pv
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning) 


# ======================================================================================================
# GENERAL UTILITIES
# ======================================================================================================

# -------------------------------- Shared-memory Utilities ---------------------------------------------

def create_shared_array(size, dtype = np.float64):
    """Create a ctypes-backed shared array filled with zeros."""
    ctype_array = np.ctypeslib.as_ctypes( np.zeros(size, dtype=dtype) )
    return sharedctypes.Array(ctype_array._type_, ctype_array,  lock=False)

def view_shared_array(shared_obj):
    """Get a NumPy view (no copy) onto a shared ctypes array created by create_shared_array."""
    return np.ctypeslib.as_array(shared_obj)


# ---------------------------------------- Helper Utilities -----------------------------------------------------

def extract_timestep_from_h5_filename(h5_file):
        """
        Extract integer timestep values of the current file from filename pattern '*_ts=<int>_...'.
        Used to sort snapshot files chronologically.
        """
        stem = h5_file.stem

        if "_ts=" in stem:
            return int(stem.split("_ts=")[1].split("_")[0])
        else:
            raise ValueError(f"Filename '{h5_file}' does not contain expected '_ts=' pattern.")

def extract_period_from_h5_filename(h5_file):
        """
        Extract period_seconds values from filename pattern '_Per<float>'.
        """
        stem = h5_file.stem

        # Extract period (ms) if not provided
        if "_Per" in stem:
            period_ms = int(stem.split("_Per")[1].split("_")[0]) #period in milliseconds
            period_seconds = period_ms/1000 #[s]
        else:
            raise ValueError(f"Filename '{h5_file}' does not contain expected '_Per' pattern.\nDefine --period_seconds in the CLI.")


        return period_seconds

def extract_timesteps_per_cyc_from_h5_filename(h5_file):
        """
        Extract timesteps_per_cyc value from filename pattern '_ts<int>'.
        """
        stem = h5_file.stem

        # Extract timestep per cycle
        if "_ts" in stem:
            timesteps_per_cyc = int(stem.split("_ts")[1].split("_")[0])
        else:
            raise ValueError(f"Filename '{h5_file}' does not contain expected '_ts' pattern.\nDefine --timestep_per_cyc in the CLI.")

        return timesteps_per_cyc


# ---------------------------------------- Mesh Utilities -----------------------------------------------------

def assemble_wall_mesh(mesh_file):
    """
    Create a Pyvista PolyData surface from the wall mesh stored in a BSLSolver-style HDF5.

    Expects datasets:
      Mesh/Wall/coordinates : (Npoints, 3) float
      Mesh/Wall/topology    : (Ncells, 3 or 4) int (triangles expected)
      Mesh/Wall/pointIds    : (Npoints,) int  (mapping back to volume point numbering)
    """
    
    with h5py.File(mesh_file, 'r') as h5:
        wall_coords = np.array(h5['Mesh/Wall/coordinates'])  # coords of wall points (n_points, 3)
        wall_cells  = np.array(h5['Mesh/Wall/topology'])     # connectivity of wall points (n_cells, 3) -> triangles
        wall_pids   = np.array(h5['Mesh/Wall/pointIds'])     # mapping to volume point IDs (n_points,)
        
    # Create connectivity array compatible with VTK --> requires a size prefix per cell (here '3' for triangles)
    n_cells        = wall_cells.shape[0]
    node_per_cell  = 3      # the surface cells are triangles with size of 3 (3 nodes per elem)
    cell_size      = np.full((n_cells, 1), node_per_cell, dtype=np.int64) # array of size (n_cells, 1) filled with 3 
    cells_vtk      = np.hstack([cell_size, wall_cells]).ravel() # horrizontal stacking of arrays / ravel: flattens the array into a 1d array
        
    # Build surface and attach point ID
    surf = pv.PolyData(wall_coords, cells_vtk)
    surf.point_data['vtkOriginalPtIds'] = wall_pids

    return surf

def assemble_volume_mesh(mesh_file):
    """
    Create a PyVista UnstructuredGrid from the volumetric mesh stored in a BSLSolver-style HDF5.

    Expects datasets:
      Mesh/coordinates : (Npoints, 3) float
      Mesh/topology    : (Ncells,  4) int (tetrahedra expected)
    """
    
    with h5py.File(mesh_file, 'r') as h5:
        coords = np.array(h5['Mesh/coordinates'])  # coords of volumetric points (n_points, 3)
        cells  = np.array(h5['Mesh/topology'])     # connectivity of volumetric points (n_cells, 4) -> tetrahedrons

        
    # Create connectivity array compatible with VTK --> requires a size prefix per cell (here '4' for tetrahedrons)
    # VTK cell array layout = [nverts, v0, v1, v2, v3, nverts, ...]
    n_cells        = cells.shape[0]
    node_per_cell  = 4  # the volumetric cells are tets with size of 4 (4 nodes per elem)
    cell_types     = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    cell_size      = np.full((n_cells, 1), node_per_cell, dtype = np.int64)  # array of size (n_cells, 1) filled with 4 
    cells_vtk      = np.hstack([cell_size, cells]).ravel() # horizontal stacking of array / ravel: flattens the array into a 1d array


    # Build grid and attach points
    vol_mesh = pv.UnstructuredGrid(cells_vtk, cell_types, coords)

    return vol_mesh, cells

# --------------------------------- Parallel File Reader -----------------------------------------------


def read_wall_pressure_from_h5_files(file_ids, wall_pids, h5_files, shared_pressure_ctype, density=1050):
    """
    Reads a *chunk* of time-snapshot HDF5 files, extracts wall pressures, and writes into the shared array.

    Arguments:
      file_ids   : list of snapshot indices to read
      wall_pids  : wall point indices (to slice pressure field)
      h5_files : list of Path objects to HDF5 snapshots
      shared_pressure_ctype  : shared ctypes array; viewed as press_[n_points, n_times]

    Note: Multiply all pressures by density (since oasis return p/rho)
    """

    # Create a shared (across processes) array of wall-pressure time-series
    shared_pressure = view_shared_array(shared_pressure_ctype)
    
    for t_index in file_ids:
        with h5py.File(h5_files[t_index], 'r') as h5:
            pressure = np.array(h5['Solution']['p']) * density
            pressure_wall = pressure[wall_pids].flatten() # shape: (n_points,)
            
        shared_pressure[:, t_index] = pressure_wall 

def read_wall_pressure_from_h5_files_parallel(CFD_h5_files, wall_mesh, n_process, density):
        
        # Total number of saved frames
        n_snapshots = len(CFD_h5_files)

        # Wall point IDs and sizes
        wall_pids = wall_mesh.point_data['vtkOriginalPtIds']
        n_points  = len(wall_mesh.points)


        # 2) Create shared arrays
        # Array to hold pressures (n_points, n_times)
        shared_pressure_ctype = create_shared_array([n_points, n_snapshots])


        # 3) Parallel reading
        print (f"Reading in parallel {n_snapshots} CFD results HDF5 files into 1 array of shape [{n_points}, {n_snapshots}] ... \n")
        

        # divide all snapshot files into groups and spread across processes
        time_indices    = list(range(n_snapshots))
        time_chunk_size = max(n_snapshots // n_process, 1)
        time_groups     = [time_indices[i : i + time_chunk_size] for i in range(0, n_snapshots, time_chunk_size)]
        
        processes_list=[]
        for idx, group in enumerate(time_groups):
            proc = mp.Process(target = read_wall_pressure_from_h5_files, name=f"Reader{idx}", args=(group, wall_pids, CFD_h5_files, shared_pressure_ctype, density))
            processes_list.append(proc)

        # Start all readers
        for proc in processes_list:
            proc.start()

        # Wait for all readers to finish
        for proc in processes_list:
            proc.join()

        wall_pressure = view_shared_array(shared_pressure_ctype) # (n_points, n_times)
        # Free up memory
        gc.collect()

        return wall_pressure



# ---------------------------------- Fourier Transform Utilities -------------------------------------------
# Used to determine STFT params if not given
def shift_bit_length(x):
    """ Round up to nearest power of 2.

    Notes: See https://stackoverflow.com/questions/14267555/  
    find-the-smallest-power-of-2-greater-than-n-in-python  
    """
    return 1<<(x-1).bit_length()

def short_time_fourier(data,
                        sampling_rate: float,
                        window_type: str = 'hann',
                        window_length: int | None = None,
                        overlap_frac: float = 0.75,
                        n_fft: int | None = None,
                        pad_mode: str | None = 'cycle',
                        detrend: str | bool = 'linear'):

    """
    Compute the windowed FFT for a timeseries data.
    All scipy.signal STFT parameters are set to defaults if they are None.
    See here for scipy.signal.stft documentation:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html 
   
    
    Arguments: 
        data: Timeseries data for the point of interest -> shape (n_points, n_frames)
        sampling_rate: Number of samples per second [Hz]
        window_type : Window type for stft
        window_length (nperseg): Number of time samples in each window 
        overlap_frca: Fraction (0-1) of overlap between segments (default = 0.75)
        n_fft: Number of FFT bins in each segment (>= window_length) -> if a zero padded FFT is desired, if None, is equal to window_length (nperseg) 
        pad_mode : Optional padding strategy to reduce edge artifacts {'cycle','constant','odd','even',None}
        detrend : {'linear','constant', False}
    
    Returns:
        array: Average spectrogram of data.
        S_db : Average power spectrogram in dB -> shape (n_freqs, n_frames)
        bins : Time vector in seconds -> shape (n_frames,)
        freqs: Frequency vector in [Hz] -> shape (n_freqs,)
        
    """


    n_frames = data.shape[1]

    # Define defaults
    if window_length is None: window_length = shift_bit_length(int(n_frames / 10))
    if n_fft is None: n_fft = window_length
    if overlap_frac is None: overlap_frac = 0.75

    if pad_mode == 'cycle':
        pad_size = window_length // 2
        front_pad = data[:,-pad_size:]
        back_pad = data[:,:pad_size]
        data = np.concatenate([front_pad, data, back_pad], axis=1)
        boundary = None 

    elif pad_mode == 'constant':
        pad_size = window_length // 2
        front_pad = np.zeros((data.shape[0], pad_size)) + data[:,0][:,None]
        back_pad = np.zeros((data.shape[0], pad_size)) + data[:,-1][:,None]
        data = np.concatenate([front_pad, data, back_pad], axis=1)
        boundary = None 

    elif pad_mode in ['odd', 'even', 'none', None]:
        boundary = pad_mode
    
    else:
        print('Warning: Problem with pad_mode!')

    stft_params = {
        'fs' : sampling_rate,
        'window' : window_type,
        'nperseg' : window_length,
        'noverlap' : int(overlap_frac * window_length),   # number of overlapping samples
        'nfft' : n_fft,
        'detrend' : detrend,
        'return_onesided' : True,
        'boundary' : boundary,
        'padded' : True,
        'axis' : -1,
        }


    # All the below S arrays have shape (n_freq, n_frames)
    freqs, bins, Z = stft(x=data, **stft_params) #data[0] will be the first row

    return freqs, bins, Z



# ---------------------------------- ROI Utilities -------------------------------------------
def read_ROI_points_from_csv(csv_path: str, ROI_type) -> np.ndarray:
    """
    Read a CSV of ROI points with columns:
    - Points:0, Points:1, Points:2
    - FrenetTangent:0, FrenetTangent:1, FrenetTangent:2
    Uses header names to locate columns.
    Returns
    - coords  : (n_points, 3) array of XYZ coordinates
    - normals : (n_points, 3) array of normals
    """

    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    # Field names from the header (quotes in CSV are handled by genfromtxt)
    #column_names = data.dtype.names

    # Change the header names for your dataset
    x = data['Points0']; y = data['Points1']; z = data['Points2']
    
    # We need the normals just for cylindrical ROI
    if ROI_type == "cylinder":
        normx = data['FrenetTangent0']; normy = data['FrenetTangent1']; normz = data['FrenetTangent2']

    elif ROI_type == "sphere":
        normx, normy, normz = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z) 

    coords = np.vstack([x,y,z]).T
    normals = np.vstack([normx,normy,normz]).T

    return coords, normals


def assemble_wall_pressure_for_one_ROI(output_folder_ROIs, wall_mesh, wall_pressure, ROI_params, return_indices=False):
    """
    Select wall points inside a ROI with defined shape (ROI_type) and return the wall-pressure time series for those points.
    Note: Units of the radius should be the same as units of the mesh.
    """

    # Unpack input parameters
    ROI_id            = ROI_params.get("ROI_id")
    ROI_type          = ROI_params.get("ROI_type")
    ROI_center_coord  = ROI_params.get("ROI_center_coord")
    ROI_center_normal = ROI_params.get("ROI_center_normal")
    ROI_radius        = ROI_params.get("ROI_radius")
    ROI_height        = ROI_params.get("ROI_height")
    flag_save_ROI     = ROI_params.get("flag_save_ROI")


    # Find coordinate of ROI center
    #ROI_center_coord = np.asarray(ROI_center_coord, dtype=float) 

    # Case 1: Obtain spectrograms at a single point
    if ROI_type == 'point':
        ROI_pids = ROI_center_coord

    # Case 2: Obtain spectrograms in a spherical ROI (using pyvista)
    elif ROI_type == 'sphere':
        
        # Creates a surface sphere centered at desired point
        ROI_sphere = pv.Sphere(radius = ROI_radius, center = ROI_center_coord)
        
        # Selects mesh points inside the surface, with a certain tolerance (using pyvista)
        ROI_mesh = wall_mesh.select_enclosed_points(ROI_sphere, tolerance=0.01)
        
        # Get indices of the points that falls in the ROI
        points_in_ROI = ROI_mesh.point_data['SelectedPoints'].astype(bool)
        ROI_pids = np.where(points_in_ROI)[0]

        # Save the sphere to a .vtp file (for visualization in paraview later)
        if flag_save_ROI:
            ROI_sphere.save(f'{output_folder_ROIs}/{ROI_id}_{ROI_type}_c{ROI_center_coord}_r{ROI_radius}.vtp') 

    
    # Case 3: Obtain spectrograms in a cylindrical ROI (using pyvista)
    elif ROI_type == 'cylinder':

        # Note: needed to add clean() to the surface to make it compatible with vtk 'select_enclosed_points'
        ROI_cylinder = pv.Cylinder(center = ROI_center_coord, direction = ROI_center_normal, radius = ROI_radius, height = ROI_height).clean()

        # Selects mesh points inside the surface, with a certain tolerance (using pyvista)
        ROI_mesh = wall_mesh.select_enclosed_points(ROI_cylinder, tolerance=0.01)
        
        # Get indices of the points that falls in the ROI
        points_in_ROI = ROI_mesh.point_data['SelectedPoints'].astype(bool)
        ROI_pids = np.where(points_in_ROI)[0]

        # Save the cylinder to a .vtp file (for visualization in paraview later)
        if flag_save_ROI:
            ROI_cylinder.save(f'{output_folder_ROIs}/{ROI_id}_{ROI_type}_c{ROI_center_coord}_r{ROI_radius}_h{ROI_height}.vtp') 

    # For any other types    
    else:
        raise ValueError("--ROI_type is not supported, choose from ['point', 'sphere', 'cylinder'].")


    # --- Sanity check: ensure ROI is not empty ---
    if ROI_pids.size == 0:
        raise ValueError("No wall points found in ROI. Try increasing --ROI_radius (check mesh units: mm vs m) "
                        "or choose a different --ROI_center_coord. ")
    else:
        print(f"Found {ROI_pids.size} wall points in {ROI_id} with center coordinate {ROI_center_coord} ...")

    # Assemble wall pressure for ROI points
    wall_pressure_ROI = wall_pressure[ROI_pids,:]

    # Return the point ids if requested
    if return_indices:
        return ROI_pids

    return wall_pressure_ROI


# ======================================================================================================
# HEMODYNAMICS FUNCTIONS
# ======================================================================================================


# ---------------------------------------- Compute Spectrograms -----------------------------------------------------

def calculate_avg_spectrogram_for_quantity_of_interest(spec_quantity, array_quantity_of_interest, STFT_params, cutoff_dB=None):

    """
    Compute an average spectrogram (in dB) based on the given array for the quantity of interest with configurable STFT parameters.
    """
    
    # Unpack input parameters
    sampling_rate = STFT_params.get("sampling_rate")
    window_length = STFT_params.get("window_length")
    overlap_frac  = STFT_params.get("overlap_frac")
    n_fft         = STFT_params.get("n_fft")
    pad_mode      = STFT_params.get("pad_mode")
    window_type   = STFT_params.get("window_type")
    detrend       = STFT_params.get("detrend")

    if spec_quantity == 'pressure':
        # Assembles pressure data for the ROI
        #wall_pressure_ROI = assemble_wall_pressure_for_one_ROI(output_folder_ROIs, wall_mesh, wall_pressure, ROI_params)
        signal_pressure = array_quantity_of_interest

        n_points = signal_pressure.shape[0]
        n_snapshots = signal_pressure.shape[1] # total number of snapshots


        # If window_length is not defined, divide the signal by 10 by default 
        if window_length is None: window_length = shift_bit_length(int(n_snapshots / 10))


        # Note: All the below S arrays have shape (n_freq, n_frames)

        # Compute FFT for first point. # Pass data as row vectors
        freqs, bins, Z0 = short_time_fourier(signal_pressure[0][None,:], sampling_rate, window_type, window_length, overlap_frac, n_fft, pad_mode, detrend)
        S_sum = np.zeros_like(Z0, dtype=np.float64)

        # Case 1: Single point ROI
        if n_points == 1:
            S_point = np.abs(Z0)**2
            S_avg_dB = 10.0 * np.log10(S_point / np.max(S_point))

        
        # Case 2: Multiple points ROI
        else:
            for point in range(n_points):
                # Pass data as row vectors
                _, _, Z_point = short_time_fourier(signal_pressure[point][None,:], sampling_rate, window_type, window_length, overlap_frac, n_fft, pad_mode, detrend)
                S_point_power = np.abs(Z_point)**2
                S_sum += S_point_power 
            
            S_avg_power = S_sum / n_points
            S_ref = (2e-5)**2 #np.mean(S_avg_power) 
            S_avg_dB = 10.0 * np.log10(S_avg_power / S_ref)
            S_avg_dB = np.squeeze(S_avg_dB)

        if pad_mode in ['cycle', 'even', 'odd']:
            bins = bins - bins[0]


        # Remove last frame to keep edges clean    
        S_avg_dB = S_avg_dB[:,:-1]
        bins = bins[:-1]

        # Clamp values below a threshold
        S_avg_dB[S_avg_dB < cutoff_dB] = cutoff_dB

        # Store all values in spectrogram_data
        spectrogram_data = {
            'power_avg_dB': S_avg_dB,
            'bins': bins,
            'freqs': freqs,
            'sampling_rate': sampling_rate,
            'n_fft': n_fft,
            'window_length': window_length,
            'overlap_frac': overlap_frac,
        }

    elif spec_quantity == 'velocity':
        # the variable is 'u' but bsl tools calculate the norm of it -> 'umag'
        print("Spectrogram calculation for velocity is not implemented yet!")

    return spectrogram_data


# ======================================================================================================
# CLASSIFICATION FUNCTIONS
# ======================================================================================================

def extract_metrics_from_spectrogram_column(freqs, spec_col_dB, f_low, f_high, f_max):
    """
    Compute simple metrics for one spectrogram column (one time).
    spec_col_dB: 1D array (n_freq,) in dB.
    f_low:       low frequency threshold in Hz (default = 100 Hz).
    f_high:      high frequency threshold in Hz (default = 1000 Hz).
    f_max:       max frequency threshold in Hz (default = 1500 Hz).
    
    Returns a dictionary of metrics for each column:
    - mean_power_below_f_low: Average acoustic power below low frequency f_low.
    - mean_power_above_f_low: Average acoustic power above low frequency f_low.
    - mean_power_above_f_high: Average acoustic power above high frequency f_high and below max frequency f_max.
    - 
    """
    
    # Filter the low frequencies
    mask_below_f_low = freqs <= f_low
    mask_above_f_low = (freqs > f_low) & (freqs < f_max)

    spec_below_f_low = spec_col_dB[mask_below_f_low]
    spec_above_f_low = spec_col_dB[mask_above_f_low]

    # Filter the high frequencies
    mask_above_f_high = (freqs >= f_high) & (freqs < f_max)
    spec_above_f_high = spec_col_dB[mask_above_f_high]


    # Basic level metrics
    # Perform the averaging in linear space and convert to dB again
    mean_power_below_f_low = np.mean(spec_below_f_low)    #10 * np.log10(np.mean(10**(spec_below_f_low/10)))
    mean_power_above_f_low = np.mean(spec_above_f_low)
    mean_power_above_f_high = np.mean(spec_above_f_high)
    frac_above_80dB  = np.mean(spec_above_f_high > 80)  #fraction of frequencies with power > 80dB

    # Spectral flatness (0 = very peaky, 1 = white noise)
    linear_power = 10.0**(spec_above_f_high/10.0)
    eps = 1e-12
    geometric_mean = np.exp(np.mean(np.log(linear_power + eps)))
    arithmetic_mean = np.mean(linear_power + eps)
    flatness_highFreq = geometric_mean / arithmetic_mean

    # Number of prominent peaks (harmonic-like structure)
    peaks, _ = find_peaks(spec_above_f_low, height= 80.0, distance=10)
    n_peaks = len(peaks)

    spec_col_metrics = dict(mean_power_below_f_low = mean_power_below_f_low,
                            mean_power_above_f_low = mean_power_above_f_low,
                            mean_power_above_f_high = mean_power_above_f_high,
                            frac_above_80dB  = frac_above_80dB,
                            flatness_highFreq = flatness_highFreq)
                            #n_peaks=n_peaks)

    #print(f"above_flow, f_high: {mean_power_above_f_low:.2f}, {mean_power_above_f_high:.2f}\n")  
    return spec_col_metrics



def classify_spectrogram_phase_per_column(metrics_col):
    mean_power_below_f_low   = metrics_col["mean_power_below_f_low"]
    mean_power_above_f_low   = metrics_col["mean_power_above_f_low"]
    mean_power_above_f_high  = metrics_col["mean_power_above_f_high"]
    frac_above_80dB         = metrics_col["frac_above_80dB"]
    flatness_highFreq       = metrics_col["flatness_highFreq"]
    #n_peaks  = metrics_col["n_peaks"]

    # Define phases
    # --- Phase 0: Quiet (nothing above noise) ---
    if (mean_power_below_f_low <= 60): 
        return 0

    # --- Phase 1: Laminar (onset of activity) ---
    elif (mean_power_below_f_low > 60 and mean_power_above_f_low <= 40):
        return 1

    # --- Phase 2: Transitional (harmonics, weak high-frequency) ---
    elif (mean_power_above_f_low > 40 and mean_power_above_f_high <= 60): #and flatness_highFreq < 0.3):
        return 2

    # --- Phase 3: Turbulent (strong broadband) ---
    elif (mean_power_above_f_high > 60 or flatness_highFreq > 0.5): #frac_above_80dB > 0.1
        return 3

    else:
        return 0


def classify_spectrogram_phases(spectrogram_data, f_low=100, f_high=1000, f_max=1500):
    """
    Map metrics dict -> phase {0,1,2,3}.
    
    f_low:       low frequency threshold in Hz (default = 100 Hz).
    f_high:      high frequency threshold in Hz (default = 1000 Hz).
    f_max:       max frequency threshold in Hz (default = 1500 Hz).

    Intended meaning:
      0: quiet / nothing
      1: weak low-frequency activity (laminar)
      2: mid-frequency activity (harmonics)
      3: strong high-frequency activity (turbulent-like)
    """

    freqs   = spectrogram_data['freqs']
    spec_dB = spectrogram_data['power_avg_dB']

    # Total number of columns of the spectrogram (#times)
    n_cols = spec_dB.shape[1] 

    # Initialize arrays
    phases = np.zeros(n_cols, dtype=int)
    metrics_list = []

    # Loop over each frame and calculate the metrics for it
    for col in range(n_cols):
        metrics_column = extract_metrics_from_spectrogram_column(freqs, spec_dB[:, col], f_low, f_high, f_max)
        metrics_list.append(metrics_column)
        phases[col] = classify_spectrogram_phase_per_column(metrics_column)

    # Optional: enforce non-decreasing phases along Q_inlet
    #for col in range(1, n_cols):
    #    if phases[col] < phases[col-1]:
    #        phases[col] = phases[col-1]

    #print(metrics_list[110], '\n')
    #print(metrics_list[-12:-10], '\n')

    return phases



def first_phase_transitions(phases, x_axis, x_min, x_max, phase_values=(1, 2, 3)):
    """
    Find first occurrence of each phase within a restricted x-range.

    Parameters
    ----------
    phases : np.ndarray
        Phase array (shape: n_cols), values in {0,1,2,3}
    x_axis : np.ndarray
        Physical x-axis (e.g. Q_inlet), same length as phases
    x_min, x_max : float
        Search limits (must match plot xlim)
    phase_values : iterable
        Phases to search for (default = (1,2,3))

    Returns
    -------
    transitions : dict
        {phase: index or None}
    """

    # Mask indices within plotting/search window
    window_mask = (x_axis >= x_min) & (x_axis <= x_max)

    transitions = {}

    for p in phase_values:
        # Indices where phase == p AND inside window
        idx = np.where((phases == p) & window_mask)[0]
        transitions[p] = int(idx[0]) if idx.size > 0 else None
        
    return transitions



def plot_and_save_spectrogram_for_ROI(output_folder_files, output_folder_imgs, case_name, spectrogram_data, spectrogram_phases, plot_title):
    """
    Save spectrogram data (.npz) and a PNG image.
    """

    # Set plot parameters
    # x-axis limits (below values are for flowrate, for time it should be [1, 5])
    xmin = 2
    xmax = 10  

    spec_output_npz = Path(output_folder_files) / f"{plot_title}.npz"
    np.savez(spec_output_npz, spectrogram_data)


    # Extract relevant data for plotting
    bins  = spectrogram_data['bins']
    freqs = spectrogram_data['freqs']
    spectrogram_signal = spectrogram_data['power_avg_dB']

    # JUST FOR PT_RAMP:
    # Create bins to show Q_inlet (instead of time) --> specify based on the ramp slope
    bins_Q = 2*bins # Q_in = 2*t

    # Clamp values below or above a certain dB threshold
    # spectrogram_signal[spectrogram_signal < cutoff_dB] = cutoff_dB
    # spectrogram_signal[spectrogram_signal > 100] = 100

    # Setting plot properties
    font_size = 12
    plt.rc('axes', titlesize=16)         # fontsize of the title
    plt.rc('font', size=font_size)       # controls default text size
    plt.rc('xtick', labelsize=font_size) # fontsize of the x tick labels
    plt.rc('ytick', labelsize=font_size) # fontsize of the y tick labels
    plt.rc('legend', fontsize=font_size) # fontsize of the legend
    plt.rc('axes', labelsize=16)         # fontsize of the x and y labels


    fig, ax = plt.subplots(1,1, figsize=(16,8))
    spectrogram = ax.pcolormesh(bins_Q, freqs, spectrogram_signal, shading='gouraud', cmap='inferno')

    #----- Set properties
    ax.set_title(plot_title)
    #ax.set_xlabel('Time (s)', fontweight='bold', labelpad=0)
    ax.set_xlabel('Q_inlet (ml/s)', fontweight='bold', labelpad=0)
    ax.set_ylabel('Frequency (Hz)', fontweight='bold', labelpad=0)
    ax.set_xlim([xmin, xmax]) 

    # Set different limits based on the case
    if 'PTSeg043' in case_name:
        ax.set_ylim([0, 600])
    else:
        ax.set_ylim([0, 1500])
    

    #----- Adding the colorbar
    cbar = fig.colorbar(spectrogram, ax=ax, orientation='vertical') #pad=0.5

    # Set different limits based on the case
    if 'PTSeg043' in case_name:
        spectrogram.set_clim(40, 100)
    else:
        spectrogram.set_clim(60, 120)

    
    #---------------- Adding the phases ------------------
    # Method 1: Add as vertical lines
    # Find first points of transition
    phase_transitions = first_phase_transitions(spectrogram_phases, x_axis=bins_Q, x_min=xmin, x_max=xmax)
    #print(phase_transitions)

    for phase, idx in phase_transitions.items():
        if idx is None:
            continue
        x = bins_Q[idx]
        ax.axvline(x, color="white", linestyle="- -", linewidth=3, alpha=0.7)

    # Method 2: Add as steps
    #ax2 = ax.twinx()
    #ax2.step(bins_Q, spectrogram_phases, where="mid", color="cyan", linewidth=4)
    #ax2.set_ylabel("Phase", fontweight='bold', rotation=270)
    #ax2.set_yticks([0,1,2,3])
    #ax2.set_ylim(-0.15, 3.15)


    #----- For customizing the colorbar and axis for figures ----
    
    #ax.set_xticks([0, 0.9])
    #ax.set_xticklabels(['0.0', '0.9'])
    #ax.set_yticks([0, 600, 800])
    #ax.set_yticklabels(['0', '600', '800'])

    # Define the ticks you want
    #ticks = [40, 60, 80, 100]
    #cbar.set_ticks(ticks)
    #cbar.set_ticklabels([str(t) for t in ticks])   # optional if you want custom text

    #cbar.ax.xaxis.tick_top()
    #cbar.ax.tick_params(labelsize=46)
    #cbar.ax.xaxis.set_label_position('top')
    #cbar.set_label('Power (dB)', rotation=270, labelpad=15, size=16, fontweight='bold')
    

    plt.tight_layout()
    plt.savefig(Path(output_folder_imgs) / f"{plot_title}.png")#, transparent=True)
    plt.close(fig)


def compute_and_save_spectrogram_for_all_ROIs(
                    case_name: str,
                    output_folder_files: Path,
                    output_folder_imgs: Path,
                    output_folder_ROIs: Path,
                    wall_mesh: pv.PolyData,
                    wall_pressure,
                    period_seconds: float, 
                    timesteps_per_cyc: int,
                    spec_quantity: str,
                    cutoff_dB: float,
                    ROI_params: dict,
                    STFT_params: dict):

    # Unpack input parameters
    ROI_type             = ROI_params.get("ROI_type")
    ROI_center_coord     = ROI_params.get("ROI_center_coord")
    ROI_center_csv       = ROI_params.get("ROI_center_csv")
    ROI_radius           = ROI_params.get("ROI_radius")
    ROI_height           = ROI_params.get("ROI_height")
    ROI_start_center_id  = ROI_params.get("ROI_start_center_id")
    ROI_end_center_id    = ROI_params.get("ROI_end_center_id")
    ROI_stride           = ROI_params.get("ROI_stride")

    window_length    = STFT_params.get("window_length")
    overlap_frac     = STFT_params.get("overlap_frac")
    n_fft            = STFT_params.get("n_fft")


    # Cmpute sampling rate and add to STFT_params
    sampling_rate = timesteps_per_cyc/period_seconds # Hz
    STFT_params["sampling_rate"] = sampling_rate 

    print (f"Now computing {spec_quantity} spectrograms for {ROI_type} ROIs with STFT parameters: \n \
    window_length (samples) = {window_length} \n \
    n_fft         (samples) = {n_fft} \n \
    overlap_fraction        = {overlap_frac} \n")


    #----------------- Case 1: CSV mode --------------------
    # Coordinates of multiple points given in a CSV file
    if ROI_center_csv is not None:
        ROI_centers, ROI_normals = read_ROI_points_from_csv(ROI_center_csv, ROI_type)
        
        #print(f"Loaded {ROI_centers.shape[0]} ROI points from {ROI_center_csv}: \n")
        
        # Loop over all center points (or with a stride)

        #-----Case 1A: Sweeping method. Generate one spectrogram for a segment chosen based on multiple ROIs
        # Here we assemble the pressure data for multiple ROIs first and then extract the average spectrogram
        if ROI_params["flag_multi_ROI"]:

            ROI_point_indices = []

            # Loop over center points in the specified region
            for i in range(ROI_start_center_id, ROI_end_center_id+1, ROI_stride): #to be inclusive of ROI_end
                center = ROI_centers[i]
                normal = ROI_normals[i]
                ROI_id = f"ROI{i:02d}"

                # Make a *copy* of ROI_params so we don't overwrite in-place
                ROI_params_multi = ROI_params.copy()
                ROI_params_multi["ROI_id"] = ROI_id
                ROI_params_multi["ROI_center_coord"]  = center
                ROI_params_multi["ROI_center_normal"] = normal

                # Obtain the point ID of all ROIs combined
                ROI_point_indices.extend(assemble_wall_pressure_for_one_ROI(output_folder_ROIs, wall_mesh, wall_pressure, ROI_params_multi, return_indices=True))

            # Keep only the unique indices
            ROI_point_indices = np.unique(ROI_point_indices)
            wall_pressure_ROI_multi = wall_pressure[ROI_point_indices, :]

            print(f"Found {len(ROI_point_indices)} unique wall points in total in the specified region. \n")

            # Calculate average spectrogram for all the ROIs combined
            spectrogram_data = calculate_avg_spectrogram_for_quantity_of_interest(
                                                spec_quantity = spec_quantity,
                                                array_quantity_of_interest = wall_pressure_ROI_multi,
                                                cutoff_dB = cutoff_dB,
                                                STFT_params = STFT_params)

            # Classify spectrogram phases
            spectrogram_phases = classify_spectrogram_phases(spectrogram_data) #, f_low=100, f_high=1000

            # Save spectrogram plot
            if ROI_type == "sphere":
                spectrogram_title = f'{case_name}_specP_win{window_length}_overlap{overlap_frac:.2f}_ROI{ROI_start_center_id}to{ROI_end_center_id}_r{ROI_radius}' 
                
            elif ROI_type == "cylinder":
                spectrogram_title = f'{case_name}_specP_win{window_length}_overlap{overlap_frac:.2f}_ROI{ROI_start_center_id}to{ROI_end_center_id}_r{ROI_radius}_h{ROI_height}' 
                
            plot_and_save_spectrogram_for_ROI(output_folder_files, output_folder_imgs, case_name, spectrogram_data, spectrogram_phases, spectrogram_title)

        
        #-----Case 1B: Generate one spectrogram per ROI
        else:
            for i in range(ROI_start_center_id, ROI_end_center_id, ROI_stride): #range(0, len(ROI_centers), 1):
                
                center = ROI_centers[i]
                normal = ROI_normals[i]
                ROI_id = f"ROI{i:02d}"

                # Add extra fields to ROI_params
                ROI_params["ROI_id"] = ROI_id
                ROI_params["ROI_center_coord"]  = center
                ROI_params["ROI_center_normal"] = normal

                # Assemble pressure data for each ROI
                wall_pressure_ROI = assemble_wall_pressure_for_one_ROI(output_folder_ROIs, wall_mesh, wall_pressure, ROI_params)

                # Calculate average spectrogram for each ROI
                spectrogram_data = calculate_avg_spectrogram_for_quantity_of_interest(
                                                    spec_quantity = spec_quantity,
                                                    array_quantity_of_interest = wall_pressure_ROI,
                                                    cutoff_dB = cutoff_dB,
                                                    STFT_params = STFT_params)

                # Classify spectrogram phases
                spectrogram_phases = classify_spectrogram_phases(spectrogram_data)

                # Save spectrogram plot
                if ROI_type == "sphere":
                    spectrogram_title = f'{case_name}_specP_win{window_length}_overlap{overlap_frac:.2f}_{ROI_id}_{ROI_type}_r{ROI_radius}' 
                
                elif ROI_type == "cylinder":
                    spectrogram_title = f'{case_name}_specP_win{window_length}_overlap{overlap_frac:.2f}_{ROI_id}_{ROI_type}_r{ROI_radius}_h{ROI_height}' 
                
                plot_and_save_spectrogram_for_ROI(output_folder_files, output_folder_imgs, case_name, spectrogram_data, spectrogram_phases, spectrogram_title)


    #------- Case 2: Coords mode
    # Single ROI center coordinates provided
    if ROI_center_coord is not None:
        ROI_center_coord = np.array(ROI_center_coord, dtype=float)

        # Add extra fields to ROI_params
        ROI_params["ROI_id"] = "single"
        ROI_params["ROI_center_coord"] = ROI_center_coord

        # Assemble pressure data for each ROI
        wall_pressure_ROI = assemble_wall_pressure_for_one_ROI(output_folder_ROIs, wall_mesh, wall_pressure, ROI_params)

        spectrogram_data = calculate_avg_spectrogram_for_quantity_of_interest(
                                                spec_quantity = spec_quantity,
                                                array_quantity_of_interest = wall_pressure_ROI,
                                                cutoff_dB = cutoff_dB,
                                                STFT_params = STFT_params)

        # Classify spectrogram phases
        spectrogram_phases = classify_spectrogram_phases(spectrogram_data) #f_low=100

        # Save spectrogram
        spectrogram_title = f'{case_name}_specP_win{window_length}_overlap{overlap_frac}_r{ROI_radius}_h{ROI_height}' 
        
        plot_and_save_spectrogram_for_ROI(output_folder_files, output_folder_imgs, case_name, spectrogram_data, spectrogram_phases, spectrogram_title)
    

    print (f'\nFinished computing and saving spetrograms.')
    
   

# ======================================================================================================
# MAIN
# ======================================================================================================

# ---------------------------------------- Run the script -----------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_folder",   required=True,       help="Results folder with CFD .h5 files")
    ap.add_argument("--mesh_folder",    required=True,       help="Case mesh folder containing HDF5 mesh file")
    ap.add_argument("--output_folder",  required=True,       help="Output directory for SPI VTP file")
    ap.add_argument("--case_name",      required=True,       help="Case name")
    ap.add_argument("--n_process",      type=int,            help="Number of parallel processes", default=max(1, mp.cpu_count() - 1))

    ap.add_argument("--density",           type=float,  default=1050,   help="Blood density [kg/m3] (default: 1050)")
    ap.add_argument("--period_seconds",    type=float,  default=0.915,  help="Period in seconds (default: 0.915)")
    ap.add_argument("--timesteps_per_cyc", type=int,                    help="Number of timesteps per cycle")
    ap.add_argument("--spec_quantity",     type=str,    required=True,  choices=["pressure","velocity"], help="Quantity of interest used for spectrogram")
    

    # ROI parameters: It allows for either a single center OR a CSV of center
    ROI_group = ap.add_mutually_exclusive_group(required=True)
    ROI_group.add_argument("--ROI_center_coord", nargs=3,  type=float, metavar=("X", "Y", "Z"), help="XYZ coordinates for a single ROI center (mesh units)")
    ROI_group.add_argument("--ROI_center_csv",   type=str, help="CSV file with multiple ROI points; coords columns = Points:0/1/2")
    
    ap.add_argument("--ROI_type",            type=str,   default="cylinder", choices=["point","sphere","cylinder"], help="Type of ROI shape")
    ap.add_argument("--ROI_radius",          type=float, required=True,      help="Radius of ROI in mesh units (mm in most cases)")
    ap.add_argument("--ROI_height",          type=float, default=2,          help="Height of cylindrical ROI in mesh units (mm in most cases)")
    ap.add_argument("--ROI_start_center_id", type=int,   default=1,          help="ROI center ID of the start of the region of inerest")
    ap.add_argument("--ROI_end_center_id",   type=int,   default=10,         help="ROI center ID of the end of the region of inerest")
    ap.add_argument("--ROI_stride",          type=int,   default=1,          help="Stride between ROIs to sweep the region of inerest")
    ap.add_argument("--flag_save_ROI",       action="store_true",            help="Flag to save ROI.vtp surface file")
    ap.add_argument("--flag_multi_ROI",      action="store_true",            help="Flag to compute spectrogram in a segment based on multiple ROIs")



    # Short-time Fourier Transform control (all optional)
    ap.add_argument("--window_length",    type=int,   default=None,     help="Length of FFT window in samples (number of snapshots for each window)")
    ap.add_argument("--n_fft",            type=int,   default=None,     help="FFT length (bins)")
    ap.add_argument("--overlap_fraction", type=float, default=0.9,     help="Overlap fraction between consequent windows [0,1] (default: 0.75)")
    ap.add_argument("--window_type",      type=str,   default="hann",   choices=["hann","hamming","boxcar","blackman","bartlett"], help="Window type for STFT")
    ap.add_argument("--pad_mode",         type=str,   default="cycle",  choices=["cycle","constant","odd","even","none"], help="Padding strategy to reduce edge artifacts")
    ap.add_argument("--detrend",          type=str,   default="linear", help="Detrend option for STFT: 'linear', 'constant', or False")
    ap.add_argument("--cutoff_dB",        type=float, default=0.0,      help="Minimum dB floor for visualization")

    
    return ap.parse_args()



def main():
    args          = parse_args()
    input_folder  = Path(args.input_folder)
    mesh_folder   = Path(args.mesh_folder)
    output_folder = Path(args.output_folder)
    
    # Create paths
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    output_folder_files = Path(f"{output_folder}/window{args.window_length}_overlap{args.overlap_fraction}_ROI{args.ROI_type}_multiROI{args.flag_multi_ROI}/files")
    output_folder_imgs = Path(f"{output_folder}/window{args.window_length}_overlap{args.overlap_fraction}_ROI{args.ROI_type}_multiROI{args.flag_multi_ROI}/imgs")
    output_folder_ROIs = Path(f"{output_folder}/window{args.window_length}_overlap{args.overlap_fraction}_ROI{args.ROI_type}_multiROI{args.flag_multi_ROI}/ROIs")
    
    output_folder_files.mkdir(parents=True, exist_ok=True)
    output_folder_imgs.mkdir(parents=True, exist_ok=True)
    if args.flag_save_ROI: output_folder_ROIs.mkdir(parents=True, exist_ok=True)

    # Put input arguments into dictionaries
    ROI_params = {
        "ROI_type": args.ROI_type,
        "ROI_center_coord": args.ROI_center_coord,
        "ROI_center_csv": args.ROI_center_csv,
        "ROI_radius": args.ROI_radius,
        "ROI_height": args.ROI_height,
        "ROI_start_center_id": args.ROI_start_center_id,
        "ROI_end_center_id": args.ROI_end_center_id,
        "ROI_stride": args.ROI_stride,
        "flag_save_ROI": args.flag_save_ROI,
        "flag_multi_ROI": args.flag_multi_ROI}

    short_time_fourier_params = {
        "window_length": args.window_length,
        "n_fft": args.n_fft,
        "overlap_frac": args.overlap_fraction,
        "window_type": args.window_type,
        "pad_mode": args.pad_mode,
        "detrend": args.detrend}

    # Assemble mesh
    mesh_file = list(Path(mesh_folder).glob('*.h5'))[0]
    wall_mesh = assemble_wall_mesh(mesh_file)
    #vol_mesh  = assemble_volume_mesh(mesh_file)

    print(f"\n[info] Mesh file:               {mesh_file}")
    print(f"[info] Read CFD results from:   {input_folder}")
    print(f"[info] Write spectrograms to:   {output_folder}")

    if args.ROI_center_csv is not None:
        print(f"[info] Read ROI centers from:  {args.ROI_center_csv}: \n")
    else:
        print(f"[info] Read ROI center from:   {args.ROI_center_coord}: \n")

    # Gather CFD results h5 files
    # Find & sort snapshot files by timestep 
    CFD_h5_files = sorted(Path(input_folder).glob('*_curcyc_*up.h5'), key = extract_timestep_from_h5_filename)
    n_snapshots = len(CFD_h5_files)

    if n_snapshots==0:
        print('No files found in {}!'.format(input_folder))
        sys.exit()
            
    # Obtain simulation temporal parameters from filename (if not given as input argument)
    timesteps_per_cyc = args.timesteps_per_cyc
    period_seconds = args.period_seconds

    if timesteps_per_cyc is None:
        timesteps_per_cyc = extract_timesteps_per_cyc_from_h5_filename(CFD_h5_files[0])
        print (f"Found timesteps_per_cycle = {timesteps_per_cyc} from CFD results HDF5 file names. \n")

    if period_seconds is None:
        period_seconds = extract_period_from_h5_filename(CFD_h5_files[0])
        print (f"Found period (s) = {period_seconds} from CFD results HDF5 file names. \n")

    # Assemble CFD data
    wall_pressure = read_wall_pressure_from_h5_files_parallel(CFD_h5_files, wall_mesh, args.n_process, args.density)    
    

    # Run post-processing of assembled CFD results
    print (f"Performing post-processing computation on {args.n_process} cores ... \n" )


    # Computing spectrograms 
    compute_and_save_spectrogram_for_all_ROIs(
                        case_name = args.case_name,
                        output_folder_files = output_folder_files,
                        output_folder_imgs = output_folder_imgs,
                        output_folder_ROIs = output_folder_ROIs,
                        wall_mesh = wall_mesh,
                        wall_pressure = wall_pressure,
                        period_seconds = period_seconds,
                        timesteps_per_cyc = timesteps_per_cyc,
                        spec_quantity = args.spec_quantity,
                        cutoff_dB = args.cutoff_dB,
                        ROI_params = ROI_params,
                        STFT_params = short_time_fourier_params)

if __name__ == '__main__':
    main()