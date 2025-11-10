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
#       > python compute_Spectrograms.py INPUTS (see below)
#               --input_folder      <path_to_CFD_results_folder> \
#               --mesh_folder       <path_to_case_mesh_data>     \
#               --output_folder     <path_to_output_folder>      \
#               --case_name         PTSeg028_base_0p64           \
#               --n_process         <ncores>                     \
#               --period_seconds    0.915                        \
#               --timesteps_per_cyc 10000                        \
#               --spec_quantity     "pressure"                   \
#               --ROI_center_coords        12.3 4.5 6.7          \
#               --ROI_radius        2                            \
#               --save_ROI_flag     True                         \
#               --window_length     1000                         \
#               --overlap_fraction  0.8                          \
#               --clamp_threshold_dB -60
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
#   --window_length       STFT window length (samples, i.e., snapshots)
#   --n_fft               STFT FFT length (bins)
#   --overlap_frac        STFT noverlap = overlap_frac * window_length --> Overlap fraction between consequent windows (0-1)
#   --window              STFT window type
#   --pad_mode            Edge padding ('cycle','constant','odd','even','none')
#   --detrend             STFT detrend ('linear','constant', or False)
#   --ROI_center_coords   X Y Z center of spherical ROI (mesh units). If ROI_radius==0, this is a **point ID**.
#   --ROI_center_csv      Path to CSV file containing the coordinates of multiple points for ROI center.
#   --ROI_radius          Sphere radius (mesh units). If 0, treat ROI_center as the **point ID** to sample.
#   --save_ROI_flag       Boolean flag to save the ROI.vtp surface file or not.
#   --clamp_threshold_dB  Floor for dB image (e.g., -60)
#   - n_process         Number of worker processes (default: #logical CPUs)
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
#from numpy.fft import fftfreq, fft
from scipy.signal import stft
import pyvista as pv
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning) 



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
        Extract integer timestep values from filename pattern '*_ts=<int>_...'.
        Used to sort snapshot files chronologically.
        """
        stem = h5_file.stem

        if "_ts=" in stem:
            return int(stem.split("_ts=")[1].split("_")[0])
        else:
            raise ValueError(f"Filename '{h5_file}' does not contain expected '_ts=' pattern.")

def extract_time_params_from_h5_filename(h5_file):
        """
        Extract timesteps_per_cyc and period_seconds values from filename pattern 'Per<float>_*_ts<int>_...'.
        """
        stem = h5_file.stem

        # Extract timestep per cycle
        if "_ts" in stem:
            timestep_per_cyc = int(stem.split("_ts")[1].split("_")[0])
        else:
            raise ValueError(f"Filename '{h5_file}' does not contain expected '_ts' pattern.")

        # Extract period if not provided
        if "_Per" in stem:
            period_ms = int(stem.split("_Per")[1].split("_")[0]) #period in milliseconds
            period_seconds = period_ms/1000 #[s]
        else:
            raise ValueError(f"Filename '{h5_file}' does not contain expected '_Per' pattern.")


        return timestep_per_cyc, period_seconds

def shift_bit_length(x):
    """ Round up to nearest power of 2.

    Notes: See https://stackoverflow.com/questions/14267555/  
    find-the-smallest-power-of-2-greater-than-n-in-python  
    """
    return 1<<(x-1).bit_length()

def read_ROI_points_from_csv(csv_path: str) -> np.ndarray:
    """
    Read a CSV of ROI points with columns:
    Points:0, Points:1, Points:2
    FrenetNormal:0, FrenetNormal:1, FrenetNormal:2
    Uses header names to locate columns.
    Returns
    coords  : (n_points, 3) array of XYZ coordinates
    normals : (n_points, 3) array of normals
    """

    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    # Field names from the header (quotes in CSV are handled by genfromtxt)
    column_names = data.dtype.names

    # Helper to get a column by name and stack into (n_points, 3)
    def stack_columns(col_name):
        # Create Points1, Points2, Points3
        cols = [f"{col_name}{i}" for i in range(3)]
        for c in cols:
            if c not in column_names:
                raise KeyError(f"Expected column '{c}' not found in CSV header.")
        return np.vstack([data[c] for c in cols]).T

    #normals = stack_columns("FrenetNormal:")
    #coords  = stack_columns("Points")
    x = data['Points0']
    y = data['Points1']
    z = data['Points2']

    coords = np.vstack([x,y,z]).T

    return coords


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
    """
    # Multiply all pressures by density (since oasis return p/rho)
    #density = 1050 #[kg/m3]

    # Create a shared (across processes) array of wall-pressure time-series
    shared_pressure = view_shared_array(shared_pressure_ctype)
    
    for t_index in file_ids:
        with h5py.File(h5_files[t_index], 'r') as h5:
            pressure = np.array(h5['Solution']['p']) * density
            pressure_wall = pressure[wall_pids].flatten() # shape: (n_points,)
            
            # For each wall point j, set shared_pressure[j, t_index] = p_wall[j]
            #for j in range(pressure_wall.shape[0]):
            #    shared_pressure[j][t_index] = pressure_wall[j]
        shared_pressure[:, t_index] = pressure_wall 



# ---------------------------------------- Compute Spectrograms -----------------------------------------------------

def assemble_wall_pressure_for_ROI(output_folder_ROIs, wall_mesh, shared_pressure_ctype, ROI_tag, ROI_center_coords, ROI_radius, save_ROI_flag=False):
    """
    Select wall points inside a ROI with defined shape and return the wall-pressure time series for those points.
    """

    # If ROI_radius == 0: interpret ROI_center as a single POINT ID (not coordinates)

    # Case 1: Obtain spectrograms at a single point
    if ROI_radius == 0:
        #ROI_pids = np.atleast_1d(ROI_center).astype(np.intp)
        ROI_pids = np.asarray(ROI_center_coords, dtype=float) 

    # Case 2: Obtain spectrograms in a spherical ROI
    else:
        # Create spherical ROI (using pyvista)
        #ROI_center = vol_mesh.points[ROI_center_pid] # return coordinates of the desired center
        ROI_center_coords = np.asarray(ROI_center_coords, dtype=float) 
        ROI_sphere = pv.Sphere(radius = ROI_radius, center = ROI_center_coords) # creates a 2d sphere around desired point (units of the radius same as units of the mesh)
        
        # Save the sphere to a .vtp file (for visualization in paraview later)
        if save_ROI_flag:
            ROI_sphere.save(f'{output_folder_ROIs}/{ROI_tag}_cx{ROI_center_coords}_r{ROI_radius}.vtp') 

        # Selects mesh points inside the surface, with a certain tolerance (using pyvista)
        ROI_mesh = wall_mesh.select_enclosed_points(ROI_sphere, tolerance=0.01)
        
        # Get indices of the points that falls in the ROI
        points_in_ROI = ROI_mesh.point_data['SelectedPoints'].astype(bool)
        ROI_pids = np.where(points_in_ROI)[0]
        #ROI_pids = np.where(wall_mesh.point_arrays[ROI_mesh])
        #ROI_pids = ROI_mesh.point_data['vtkOriginalPtIds']
    
        # --- Sanity check: ensure ROI is not empty ---
        if ROI_pids.size == 0:
            raise ValueError("No wall points found in ROI. Try increasing --ROI_radius (check mesh units: mm vs m) "
                            "or choose a different --ROI_center_coords. ")
        else:
            print(f"Found {ROI_pids.size} wall points in {ROI_tag}...")

    # Assemble wall pressure for ROI points
    wall_pressure = view_shared_array(shared_pressure_ctype) # (n_points, n_times)
    wall_pressure_ROI = wall_pressure[ROI_pids,:]

    return wall_pressure_ROI



def average_spectrogram(data,
                        sampling_rate: float,
                        window_type: str = 'hann',
                        window_length: int | None = None,
                        overlap_frac: float = 0.75,
                        n_fft: int | None = None,
                        pad_mode: str | None = 'cycle',
                        detrend: str | bool = 'linear',
                        print_progress: bool = False):

    """
    Compute the power spectrogram over multiple points and return the average value in dB scale.
    All scipy.signal STFT parameters are set to defaults if they are None.
    See here for scipy.signal.stft documentation:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html 
   
    
    Arguments: 
        data: Timeseries data for all ROI points -> shape (n_points, n_snapshots)
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

    elif pad_mode in ['odd', 'even', None]:
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
    freqs, bins, Z0 = stft(x=data[0], **stft_params) #data[0] will be the first row

    S_sum = np.zeros_like(Z0, dtype=np.float64)

    # Case 1: Single point ROI
    if data.shape[0] == 1:
        S_point = np.abs(Z0)**2
        S_avg_dB = 10.0 * np.log10(S_point / np.max(S_point))

    
    # Case 2: Multiple points ROI
    else:
        for point in range(data.shape[0]):
            _, _, Z_point = stft(x=data[point], **stft_params)
            S_point_power = np.abs(Z_point)**2
            S_sum += S_point_power 
        
        S_avg_power = S_sum / data.shape[0]
        S_ref = np.mean(S_avg_power)
        S_avg_dB = 10.0 * np.log10(S_avg_power / S_ref)

    #print(f"Shape of S_avg_dB is {S_avg_dB.shape}")

    if pad_mode in ['cycle', 'even', 'odd']:
        bins = bins - bins[0]


    # Remove last frame to keep edges clean    
    S_avg_dB = S_avg_dB[:,:-1]
    bins = bins[:-1]

    # Store all values in spectrogram_data
    spectrogram_data = {
        'S_avg_dB': S_avg_dB,
        'bins': bins,
        'freqs': freqs,
        'sampling_rate': sampling_rate,
        'n_fft': n_fft,
        'window_length': window_length,
        'overlap_frac': overlap_frac,
    }

    return spectrogram_data



def compute_spectrogram_wall_pressure(output_folder_files,
                                      output_folder_imgs,
                                      output_folder_ROIs,
                                      wall_mesh,
                                      shared_pressure_ctype,
                                      period_seconds,
                                      timesteps_per_cyc,
                                      spec_quantity,
                                      ROI_center_coords,
                                      ROI_radius,
                                      ROI_tag,
                                      save_ROI_flag,
                                      window_length,
                                      n_fft=None,
                                      overlap_frac=None,
                                      window_type='hann',
                                      pad_mode='cycle',
                                      detrend='linear',
                                      clamp_threshold_dB=None):

    """
    Assemble ROI time series and compute an average spectrogram (in dB) with configurable STFT parameters.
    """
    
    if spec_quantity == 'pressure':
        # Assembles data for the ROI
        wall_pressure_ROI = assemble_wall_pressure_for_ROI(output_folder_ROIs, wall_mesh, shared_pressure_ctype, ROI_tag, ROI_center_coords, ROI_radius, save_ROI_flag)
        #print(f"Wall_pressure_ROI shape: {wall_pressure_ROI.shape}")

        n_snapshots = wall_pressure_ROI.shape[1] # total number of snapshots
        sampling_rate = timesteps_per_cyc/period_seconds # [Hz]

        # If window_length is not defined, divide the signal by 10 by default 
        if window_length is None: window_length = shift_bit_length(int(n_snapshots / 10))
        #n_fft = window_length
        #hop_length = int(n_fft / 4)
        #overlap_frac = 0.75
        
        # Compute average spectrogram
        spectrogram_data = average_spectrogram(
                        data=wall_pressure_ROI,
                        sampling_rate = sampling_rate,
                        n_fft = n_fft,
                        overlap_frac = overlap_frac,
                        window_length = window_length,
                        window_type = window_type,
                        pad_mode = pad_mode,
                        detrend = detrend)


        #wall_mesh.spectrogram_data['p'] = spectrogram_data
        

    elif spec_quantity == 'velocity':
        # the variable is 'u' but bsl tools calculate the norm of it -> 'umag'
        print("Spectrogram calculation for velocity is not implemented yet!")

    return spectrogram_data


def plot_spectrogram(output_folder_files, output_folder_imgs, case_name, spectrogram_data, plot_title, clamp_threshold_dB):
    """
    Save spectrogram data (.npz) and a PNG image.
    """

    spec_output_npz = Path(output_folder_files) / f"{plot_title}.npz"
    np.savez(spec_output_npz, spectrogram_data)


    # Extract relevant data for plotting
    bins = spectrogram_data['bins']
    freqs = spectrogram_data['freqs']
    spectrogram_signal = spectrogram_data['S_avg_dB']

    # Clamp values below a certain dB threshold
    spectrogram_signal[spectrogram_signal < clamp_threshold_dB] = clamp_threshold_dB

    # Setting plot properties
    font_size = 12
    plt.rc('axes', titlesize=18)         # fontsize of the title
    plt.rc('font', size=font_size)       # controls default text size
    plt.rc('xtick', labelsize=font_size) # fontsize of the x tick labels
    plt.rc('ytick', labelsize=font_size) # fontsize of the y tick labels
    plt.rc('legend', fontsize=font_size) # fontsize of the legend
    plt.rc('axes', labelsize=16)         # fontsize of the x and y labels


    fig, ax = plt.subplots(1,1, figsize=(12,8))
    spectrogram = ax.pcolormesh(bins, freqs, spectrogram_signal, shading='gouraud', cmap='inferno')
    
    # Set axis
    ax.set_title(plot_title, fontweight='bold')
    ax.set_xlabel('Time (s)', fontweight='bold', labelpad=0)
    ax.set_ylabel('Frequency (Hz)', fontweight='bold', labelpad=0)
    ax.set_xlim([1, 5])
    ax.set_ylim([0, 1500])
    cbar = plt.colorbar(spectrogram, ax=ax) # Adding the colorbar
    cbar.set_label('Power (dB)', rotation=270, labelpad=15, size=16, fontweight='bold')
    #ax.set_xticks([0, 0.9])
    #ax.set_xticklabels(['0.0', '0.9'])
    #ax.set_yticks([0, 600, 800])
    #ax.set_yticklabels(['0', '600', '800'])
    #spectrogram.set_clim([-20, 0])

    
    plt.tight_layout()
    plt.savefig(Path(output_folder_imgs) / f"{plot_title}.png")#, transparent=True)
    plt.close(fig)




# ---------------------------------------- Compute Hemodynamics Metrics -------------------------------------

def hemodynamics(wall_mesh: pv.PolyData,
                 input_folder: Path,
                 output_folder_files: Path,
                 output_folder_imgs: Path,
                 output_folder_ROIs: Path,
                 case_name: str,
                 n_process: int,
                 density: float,
                 period_seconds: float,
                 timesteps_per_cyc: int,
                 spec_quantity: str,
                 ROI_center_coords: list[float],
                 ROI_center_csv: str,
                 ROI_radius: int,
                 save_ROI_flag: bool,
                 clamp_threshold_dB: float,
                 window_length: int,
                 n_fft: int,
                 overlap_frac: float,
                 window_type: str,
                 pad_mode: str,
                 detrend: str,
                 ):

    """
    Main driver: Reads time-series pressures, computes windowed SPI, and writes it to a VTP file.

    Args:
      wall_mesh      : PyVista PolyData for the wall surface
      input_folder   : folder containing '*_curcyc_*up.h5' snapshots
      output_folder  : output folder for VTP
      case_name      : prefix for output filename
      n_process      : number of processes for parallel file reading
    """

    # 1) Gather files
    # Find & sort snapshot files by timestep 
    snapshot_h5_files = sorted(Path(input_folder).glob('*_curcyc_*up.h5'), key = extract_timestep_from_h5_filename)
    n_snapshots = len(snapshot_h5_files)

    if n_snapshots==0:
        print('No files found in {}!'.format(input_folder))
        sys.exit()

    
    # Obtain simulation temporal parameters from filename (if not given as input argument)
    if timesteps_per_cyc is None or period_seconds is None:
        timesteps_per_cyc, period_seconds = extract_time_params_from_h5_filename(snapshot_h5_files[0])

    
    # Wall point IDs and sizes
    wall_pids = wall_mesh.point_data['vtkOriginalPtIds']
    n_points  = len(wall_mesh.points)


    # 2) Create shared arrays
    # Array to hold pressures (n_points, n_times)
    shared_pressure_ctype = create_shared_array([n_points, n_snapshots])


    # 3) Parallel reading
    print (f"Reading in parallel {n_snapshots} HDF5 files into 1 array of shape [{n_points}, {n_snapshots}] ... \n")

    # divide all snapshot files into groups and spread across processes
    time_indices    = list(range(n_snapshots))
    time_chunk_size = max(n_snapshots // n_process, 1)
    time_groups     = [time_indices[i : i + time_chunk_size] for i in range(0, n_snapshots, time_chunk_size)]
    
    processes_list=[]
    for idx, group in enumerate(time_groups):
        proc = mp.Process(target = read_wall_pressure_from_h5_files, name=f"Reader{idx}", args=(group, wall_pids, snapshot_h5_files, shared_pressure_ctype, density))
        processes_list.append(proc)

    # Start all readers
    for proc in processes_list:
        proc.start()

    # Wait for all readers to finish
    for proc in processes_list:
        proc.join()

    # Free up memory
    gc.collect()


    # 4) Computing spectrogram 
    print (f"Now computing {spec_quantity} spectrograms with STFT parameters: \n \
        window_length (samples) = {window_length} \n \
        n_fft         (samples) = {n_fft} \n \
        overlap_fraction        = {overlap_frac} \n")

    # Case 1: Coords mode
    # Single ROI center coordinates provided
    if ROI_center_coords is not None:
        ROI_center_coords = np.array(ROI_center_coords, dtype=float)

        spectrogram_data = compute_spectrogram_wall_pressure(
            output_folder_files = output_folder_files,
            output_folder_imgs = output_folder_imgs,
            output_folder_ROIs = output_folder_ROIs,
            wall_mesh = wall_mesh,
            shared_pressure_ctype = shared_pressure_ctype,
            period_seconds = period_seconds,
            timesteps_per_cyc = timesteps_per_cyc,
            spec_quantity = spec_quantity,
            ROI_center_coords = ROI_center_coords,
            ROI_radius = ROI_radius,
            ROI_tag = "single",
            save_ROI_flag = save_ROI_flag,
            clamp_threshold_dB = clamp_threshold_dB,
            window_length = window_length,
            n_fft = n_fft,
            overlap_frac = overlap_frac,
            window_type = window_type,
            pad_mode = pad_mode,
            detrend = detrend)

        # 5) Save spectrogram
        # Creates the output filename (npz format: numpy zipped file)
        cx, cy, cz = map(float, ROI_center_coords)
        center_tag = f"cx{cx:.2f}cy{cy:.2f}cz{cz:.2f}"
        spectrogram_title = f'{case_name}_specP_win{window_length}_overlap{overlap_frac:.2f)}_{center_tag}_r{ROI_radius}' 
        
        plot_spectrogram(output_folder_files, output_folder_imgs, case_name, spectrogram_data, spectrogram_title, clamp_threshold_dB)


    # Case 2: CSV mode
    # Coordinates of multiple points given in a CSV file
    if ROI_center_csv is not None:
        ROI_centers = read_ROI_points_from_csv(ROI_center_csv)
        print(f"Loaded {ROI_centers.shape[0]} ROI points from {ROI_center_csv}.")

        # Loop over all center points
        for i, center in enumerate(ROI_centers[20:30]):
            ROI_tag = f"ROI{i:03d}"
            
            spectrogram_data = compute_spectrogram_wall_pressure(
                output_folder_files = output_folder_files,
                output_folder_imgs = output_folder_imgs,
                output_folder_ROIs = output_folder_ROIs,
                wall_mesh = wall_mesh,
                shared_pressure_ctype = shared_pressure_ctype,
                period_seconds = period_seconds,
                timesteps_per_cyc = timesteps_per_cyc,
                spec_quantity = spec_quantity,
                ROI_center_coords = center,
                ROI_radius = ROI_radius,
                ROI_tag = ROI_tag,
                save_ROI_flag = save_ROI_flag,
                clamp_threshold_dB = clamp_threshold_dB,
                window_length = window_length,
                n_fft = n_fft,
                overlap_frac = overlap_frac,
                window_type = window_type,
                pad_mode = pad_mode,
                detrend = detrend)

            # 5) Save spectrogram
            # Creates the output filename (npz format: numpy zipped file)
            cx, cy, cz = map(float, center)
            center_tag = f"cx{cx:.2f}cy{cy:.2f}cz{cz:.2f}"
            spectrogram_title = f'{case_name}_specP_win{window_length}_overlap{overlap_frac:.2f}_{ROI_tag}_{center_tag}_r{ROI_radius}' 
            
            plot_spectrogram(output_folder_files, output_folder_imgs, case_name, spectrogram_data, spectrogram_title, clamp_threshold_dB)

    
    print (f'Finished saving spetrograms.')




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
    ap.add_argument("--spec_quantity",     type=str,    required=True,  choices=["pressure","velocity"], help="Quantity used for spectrogram")
    

    # ROI parameters: It allows for either a single center OR a CSV of center
    ROI_group = ap.add_mutually_exclusive_group(required=True)
    ROI_group.add_argument("--ROI_center_coords", nargs=3, type=float, metavar=("X", "Y", "Z"), help="XYZ coordinates for a single ROI center (mesh units)")
    ROI_group.add_argument("--ROI_center_csv", type=str, help="CSV file with multiple ROI points; coords columns = Points:0/1/2")
    #ap.add_argument("--ROI_center",nargs=3, type=float, metavar=("X","Y","Z"), required=True, help="XYZ coordinates for ROI center to compute spectrogram (mesh units)")
    ap.add_argument("--ROI_type",       type=str, choices=["point","sphere","cylinder"], help="Type of ROI shape")
    ap.add_argument("--ROI_radius",     type=float, required=True, help="Radius of ROI to compute spectrogram in mesh units (mm in most cases)")
    ap.add_argument("--save_ROI_flag",  type=bool,  default=False, help="Flag to save ROI.vtp surface file or not")
  
    # Short-time Fourier Transform control (all optional)
    ap.add_argument("--window_length",    type=int,   default=None,     help="Length of FFT window in samples (number of snapshots for each window)")
    ap.add_argument("--n_fft",            type=int,   default=None,     help="FFT length (bins)")
    ap.add_argument("--overlap_fraction", type=float, default=0.75,     help="Overlap fraction between consequent windows [0,1] (default: 0.75)")
    ap.add_argument("--window_type",      type=str,   default="hann",   choices=["hann","hamming","boxcar","blackman","bartlett"], help="Window type for STFT")
    ap.add_argument("--pad_mode",         type=str,   default="cycle",  choices=["cycle","constant","odd","even","none"], help="Padding strategy to reduce edge artifacts")
    ap.add_argument("--detrend",          type=str,   default="linear", help="Detrend option for STFT: 'linear', 'constant', or False")
    ap.add_argument("--clamp_threshold_dB", type=float, default=-60.0, help="Minimum dB floor for visualization")
    

    
    return ap.parse_args()



def main():
    args          = parse_args()
    input_folder  = Path(args.input_folder)
    mesh_folder   = Path(args.mesh_folder)
    output_folder = Path(args.output_folder)

    # Create paths
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    output_folder_files = Path(f"{output_folder}/window{args.window_length}_overlap{args.overlap_fraction}/files")
    output_folder_imgs = Path(f"{output_folder}/window{args.window_length}_overlap{args.overlap_fraction}/imgs")
    output_folder_ROIs = Path(f"{output_folder}/window{args.window_length}_overlap{args.overlap_fraction}/ROIs")
    
    output_folder_files.mkdir(parents=True, exist_ok=True)
    output_folder_imgs.mkdir(parents=True, exist_ok=True)
    output_folder_ROIs.mkdir(parents=True, exist_ok=True)

    # Assemble mesh
    mesh_file = list(Path(mesh_folder).glob('*.h5'))[0]
    wall_mesh = assemble_wall_mesh(mesh_file)
    #vol_mesh  = assemble_volume_mesh(mesh_file)


    print(f"[info] Mesh file:    {mesh_file}")
    print(f"[info] Reading from: {input_folder}")
    print(f"[info] Writing to:   {output_folder} \n")

    print (f"Performing post-processing computation on {args.n_process} cores ..." )

    # Run post-processing    
    hemodynamics(
        wall_mesh = wall_mesh,
        input_folder = input_folder,
        output_folder_files = output_folder_files,
        output_folder_imgs = output_folder_imgs,
        output_folder_ROIs = output_folder_ROIs,
        case_name = args.case_name,
        n_process = args.n_process,
        density = args.density,
        period_seconds = args.period_seconds,
        timesteps_per_cyc = args.timesteps_per_cyc,
        spec_quantity = args.spec_quantity,
        ROI_center_coords = args.ROI_center_coords,
        ROI_center_csv = args.ROI_center_csv,
        ROI_radius = args.ROI_radius,
        save_ROI_flag = args.save_ROI_flag,
        clamp_threshold_dB = args.clamp_threshold_dB,
        window_length = args.window_length,
        n_fft = args.n_fft,
        overlap_frac = args.overlap_fraction,
        window_type = args.window_type,
        pad_mode = args.pad_mode,
        detrend = args.detrend)



if __name__ == '__main__':
    main()