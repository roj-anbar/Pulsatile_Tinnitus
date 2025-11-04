# -----------------------------------------------------------------------------------------------------------------------
# compute-post-metrics_Spectrograms.py 
# To compute the spectrograms of pressure or velocity on vessel wall from Oasis/BSLSolver CFD outputs.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2025-10
#
# PURPOSE:
#   - This script is part of the BSL post-processing pipeline.
#   - It computes 
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
#       > python compute-post-metrics_Spectrograms.py <path_to_CFD_results_folder> <path_to_case_mesh_data> <case_name> <path_to_output_folder> <period> <ncores> <f_cutoff> <flag_mean>
#     
#
# INPUTS:
#   - input_folder      Path to results directory with HDF5 snapshots
#   - mesh_folder       Path to the case mesh data files (with Mesh/Wall/{coordinates,topology,pointIds})
#   - case_name         Case name (used only in output filename)
#   - output_folder     Output directory for the VTP result
#   - n_process         Number of worker processes (default: #logical CPUs)
#   - flag_pressure_velocity
#
#
# OUTPUTS:
#   - output_folder/<case>_p_<f-cut>Hz.vtp  PolyData with 'SPI_p' point-data
#   1) Processed spectrograms saved as .npz
#         2) Spectrogram images saved as .png

# NOTES:
#   - Time step dt is inferred from: dt = period / (N-1) (one period covered by N files).
#
#
# Adapted from hemodynamic_indices_pressure.py originally written by Anna Haley (2024). 
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

def extract_timestep_from_h5(h5_file):
        """
        Extract integer timestep from filename pattern '*_ts=<int>_...'.
        Used to sort snapshot files chronologically.
        """
        return int(h5_file.stem.split('_ts=')[1].split('_')[0])

def shift_bit_length(x):
    """ Round up to nearest power of 2.

    Notes: See https://stackoverflow.com/questions/14267555/  
    find-the-smallest-power-of-2-greater-than-n-in-python  
    """
    return 1<<(x-1).bit_length()

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

# --------------------------------- Parallel Mesh Reader -----------------------------------------------


def read_wall_pressure_from_h5_files(file_ids, wall_pids, h5_files, shared_pressure_ctype):
    """
    Reads a *chunk* of time-snapshot HDF5 files, extracts wall pressures, and writes into the shared array.

    Arguments:
      file_ids   : list of snapshot indices to read
      wall_pids  : wall point indices (to slice pressure field)
      h5_files : list of Path objects to HDF5 snapshots
      shared_pressure_ctype  : shared ctypes array; viewed as press_[n_points, n_times]
    """
    # Multiply all pressures by density (since oasis return p/rho)
    density = 1050 #[kg/m3]

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

def assemble_wall_pressure_for_ROI(output_folder, wall_mesh, shared_pressure_ctype, ROI_center, ROI_radius):
    if ROI_radius == 0:
        ROI_pids = np.atleast_1d(ROI_center).astype(np.intp)
        #print(f"ROI_pid = {ROI_pids}")

    else:
        # Create the spherical ROI (using pyvista)
        #ROI_center = vol_mesh.points[ROI_center_pid] # return coordinates of the desired center
        ROI_center = np.asarray(ROI_center, dtype=float) 
        ROI_sphere = pv.Sphere(radius = ROI_radius, center = ROI_center) # creates a 2d sphere around desired point (units of the radius same as units of the mesh)
        
        # Save the sphere to a .vtp file (for visualization in paraview later)
        ROI_sphere.save(f'{output_folder}/ROI_sphere_pid{ROI_center}_r{ROI_radius}.vtp') 

        # Selects mesh points inside the surface, with a certain tolerance (using pyvista)
        ROI_mesh = wall_mesh.select_enclosed_points(ROI_sphere, tolerance=0.01)
        
        # Get indices of the points on the ROI
        sel = ROI_mesh.point_data['SelectedPoints'].astype(bool)
        ROI_pids = np.where(sel)[0]
        #ROI_pids = np.where(wall_mesh.point_arrays[ROI_mesh])
        #ROI_pids = ROI_mesh.point_data['vtkOriginalPtIds']
    
        # --- Sanity check: ensure ROI is not empty ---
        if ROI_pids.size == 0:
            raise ValueError(
                "No wall points found in ROI. Try increasing --ROI_radius (check mesh units: mm vs m) "
                "or choose a different --ROI_center. ")
        else:
            print(f"Found {ROI_pids.size} wall points in the ROI")

    wall_pressure = view_shared_array(shared_pressure_ctype) # (n_points, n_times)
    wall_pressure_ROI = wall_pressure[ROI_pids,:]

    return wall_pressure_ROI



def average_spectrogram(data, sampling_rate, n_fft=None, hop_length=None, win_length=None, 
                        window='hann', pad_mode='cycle', detrend='linear', print_progress=False):
    """ Compute the average spectrogram of a dataset.
    
    Args: 
        data (array): N * ts array.
        For other args, see librosa.stft docs.
    
    Returns:
        array: Average spectrogram of data.

    """

    # Define defaults
    n_samples = data.shape[1]
    
    if n_fft is None: n_fft = shift_bit_length(int(n_samples / 10))
    if hop_length is None: hop_length = int(n_fft / 4)
    if win_length is None: win_length = n_fft


    if pad_mode == 'cycle':
        pad_size = win_length // 2
        front_pad = data[:,-pad_size:]
        back_pad = data[:,:pad_size]
        data = np.concatenate([front_pad, data, back_pad], axis=1)
        boundary = None 

    elif pad_mode == 'constant':
        pad_size = win_length // 2
        front_pad = np.zeros((data.shape[0], pad_size)) + data[:,0][:,None]
        back_pad = np.zeros((data.shape[0], pad_size)) + data[:,-1][:,None]
        data = np.concatenate([front_pad, data, back_pad], axis=1)
        boundary = None 

    elif pad_mode in ['odd', 'even', None]:
        boundary = pad_mode
    
    else:
        print('Problem with pad_mode')


    # See here for scipy.signal.stft documentation:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html 
    stft_params = {
        'fs' : sampling_rate,
        'window' : window,
        'nperseg' : win_length,
        'noverlap' : win_length - hop_length,
        'nfft' : n_fft,
        'detrend' : detrend,
        'return_onesided' : True,
        'boundary' : boundary,
        'padded' : True,
        'axis' : -1,
        }

    freqs, bins, S0 = stft(x=data[0], **stft_params) #data[0] will be the first row

    S_avg = np.zeros_like(S0)

    # Case 1: Single point ROI
    if data.shape[0] == 1:
        S_point = np.abs(S0)**2
        S_avg = np.log(S_point)

    
    # Case 2: Multiple points ROI
    else:
        for point in range(data.shape[0]):
            _, _, S_point = stft(x=data[point], **stft_params)
            S_point = np.abs(S_point)**2
            S_avg += S_point 
    
        # This is a difference between using signal.stft and signal.spectrogram.
        S_avg = np.log(S_avg / data.shape[0] + 1e-30) # add a small value to avoid log(0)

    if pad_mode in ['cycle', 'even', 'odd']:
        bins = bins - bins[0]

    return S_avg, bins, freqs



def compute_spectrogram_wall_pressure(output_folder, wall_mesh, shared_pressure_ctype, period_seconds, num_cycles,
                                     window_size, ROI_center, ROI_radius, spec_quantity = 'pressure'):
    
    if spec_quantity == 'pressure':
        # Assembles data for the ROI
        wall_pressure_ROI = assemble_wall_pressure_for_ROI(output_folder, wall_mesh, shared_pressure_ctype, ROI_center, ROI_radius)
        

        n_samples = wall_pressure_ROI.shape[1] # number of snapshots
        sampling_rate = n_samples/(num_cycles*period_seconds)

        n_fft = int(window_size)
        hop_length = int(0.25 * n_fft)
        win_length = n_fft


        S, bins, freqs = average_spectrogram(
                        data=wall_pressure_ROI,
                        sampling_rate=sampling_rate,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window='hann',
                        pad_mode='cycle',
                        detrend='linear')

        # Remove last frame
        S = S[:,:-1]
        bins = bins[:-1]

        spec_data = {}
        spec_data['S'] = S
        spec_data['bins'] = bins
        spec_data['freqs'] = freqs
        spec_data['sapmling_rate'] = sampling_rate
        spec_data['n_fft'] = n_fft

        #wall_mesh.spectrogram_data['p'] = spec_data
        

    elif spec_quantity == 'velocity':
        # the variable is 'u' but bsl tools calculate the norm of it -> 'umag'
        print("Spectrogram calculation for velocity is not implemented yet!")

    return spec_data


def plot_spectrogram(output_folder, case_name, spec_data, plot_title):

    spec_output_file = Path(output_folder) / f"{plot_title}.npz"
    np.savez(spec_output_file, spec_data)


    # Extract relevant data for plotting
    bins = spec_data['bins']
    freqs = spec_data['freqs']
    spec_signal = spec_data['S']

    # Clamp values below -20dB
    spec_signal[spec_signal < -20] = -20 

    # Setting plot properties
    size = 10
    plt.rc('font', size=size) #controls default text size
    plt.rc('axes', titlesize=12) #fontsize of the title
    plt.rc('axes', labelsize=size) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=size) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=size) #fontsize of the y tick labels
    plt.rc('legend', fontsize=size) #fontsize of the legend


    fig, ax = plt.subplots(1,1, figsize=(8,6))
    spectrogram = ax.pcolormesh(bins, freqs, spec_signal, shading='gouraud')
    #spectrogram.set_clim([-20, 0])
    ax.set_xlabel('Time (s)', labelpad=-5)
    ax.set_ylabel('Freq (Hz)', labelpad=-10)
    #ax.set_xticks([0, 0.9])
    #ax.set_xticklabels(['0.0', '0.9'])
    #ax.set_yticks([0, 600, 800])
    #ax.set_yticklabels(['0', '600', '800'])
    ax.set_ylim([0, 2000])

    ax.set_title(plot_title)
    plt.tight_layout()
    plt.colorbar(spectrogram, ax=ax) # Adding the colorbar
    plt.savefig(Path(output_folder) / f"{plot_title}.png")#, transparent=True)




# ---------------------------------------- Compute Hemodynamics Metrics -------------------------------------

def hemodynamics(wall_mesh: pv.PolyData,
                 input_folder: Path,
                 output_folder: Path,
                 case_name: str,
                 n_process: int,
                 period_seconds: float,
                 num_cycles: int,
                 spec_quantity: str,
                 window_size: int,
                 ROI_center: list[float],
                 ROI_radius: int,
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
    snapshot_h5_files = sorted(Path(input_folder).glob('*_curcyc_*up.h5'), key = extract_timestep_from_h5)
    n_snapshots = len(snapshot_h5_files)

    if n_snapshots==0:
        print('No files found in {}!'.format(input_folder))
        sys.exit()
    
    # Wall point IDs and sizes
    wall_pids = wall_mesh.point_data['vtkOriginalPtIds']
    n_points  = len(wall_mesh.points)


    # 2) Create shared arrays
    # Array to hold pressures (n_points, n_times)
    shared_pressure_ctype = create_shared_array([n_points, n_snapshots])



    # 3) Parallel reading
    print (f"Reading in parallel {n_snapshots} files into 1 array of shape [{n_points}, {n_snapshots}] ... \n")

    # divide all snapshot files into groups and spread across processes
    time_indices    = list(range(n_snapshots))
    time_chunk_size = max(n_snapshots // n_process, 1)
    time_groups     = [time_indices[i : i + time_chunk_size] for i in range(0, n_snapshots, time_chunk_size)]
    
    processes_list=[]
    for idx, group in enumerate(time_groups):
        proc = mp.Process(target = read_wall_pressure_from_h5_files, name=f"Reader{idx}", args=(group, wall_pids, snapshot_h5_files, shared_pressure_ctype))
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
    print ('Now computing spectrograms ...')

  
    spec_data = compute_spectrogram_wall_pressure(output_folder,
                wall_mesh, shared_pressure_ctype, period_seconds, num_cycles,
                window_size, ROI_center, ROI_radius, spec_quantity)

    # 5) Save spectrogram
    # Creates the output filename (npz format: numpy zipped file)
    cx, cy, cz = map(float, ROI_center)
    center_tag = f"{cx:.2f}{cy:.2f}{cz:.2f}"
    spec_title = f'{case_name}_specP_window{window_size}_center{center_tag}_r{ROI_radius}' 
    plot_spectrogram(output_folder, case_name, spec_data, spec_title)

    
    print (f'Finished saving spetrograms.')




# ---------------------------------------- Run the script -----------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_folder",   required=True,       help="Results folder with CFD .h5 files")
    ap.add_argument("--mesh_folder",    required=True,       help="Case mesh folder containing HDF5 mesh file")
    ap.add_argument("--output_folder",  required=True,       help="Output directory for SPI VTP file")
    ap.add_argument("--case_name",      required=True,       help="Case name")
    ap.add_argument("--period",         type=float,          help="Period in seconds (default: 0.915)", default=0.915)
    ap.add_argument("--num_cycles",     type=int,            help="Number of cycles")
    ap.add_argument("--spec_quantity",  type=str, choices=["pressure","velocity"], required=True, help="Quantity used for spectrogram generation (choose between <pressure>/<velocity>)")
    ap.add_argument("--window_size",    type=int,            help="Size of FFT window (number of snapshots for each window)")
    ap.add_argument("--ROI_center",     nargs=3, type=float, metavar=("X","Y","Z"), required=True, help="XYZ coordinates for ROI center to compute spectrogram (mesh units)")
    ap.add_argument("--ROI_radius",     type=float, required=True,       help="Radius of ROI to compute spectrogram in mesh units (mm in most cases)")
    ap.add_argument("--n_process",      type=int,            help="Number of parallel processes", default=max(1, mp.cpu_count() - 1))

    return ap.parse_args()


def main():
    args          = parse_args()
    input_folder  = Path(args.input_folder)
    mesh_folder   = Path(args.mesh_folder)
    output_folder = Path(args.output_folder)

    # Create paths
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)


    # Assemble mesh
    mesh_file = list(Path(mesh_folder).glob('*.h5'))[0]
    wall_mesh = assemble_wall_mesh(mesh_file)
    #vol_mesh  = assemble_volume_mesh(mesh_file)


    print(f"[info] Mesh file:    {mesh_file}")
    print(f"[info] Reading from: {input_folder}")
    print(f"[info] Writing to:   {output_folder} \n")


    print (f"Performing hemodynamics computation on {args.n_process} cores..." )
    hemodynamics(wall_mesh, input_folder, output_folder, case_name = args.case_name, n_process = args.n_process,
                period_seconds= args.period, num_cycles= args.num_cycles, spec_quantity= args.spec_quantity,
                window_size= args.window_size, ROI_center = args.ROI_center, ROI_radius = args.ROI_radius)



if __name__ == '__main__':
    main()