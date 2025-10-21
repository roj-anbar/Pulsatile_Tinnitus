# -----------------------------------------------------------------------------------------------------------------------
# compute-post-metrics_Qcriterion.py 
# To calculate Qcriterion from Oasis/BSLSolver CFD outputs.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2025-10
#
# PURPOSE:
#   - This script is part of the BSL post-processing pipeline.
#   - It computes Qcriterion at each volumetric point and saves the resultant array 'Qcriterion' on the mesh to a VTU file.
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
#       > python compute-post-metrics_Qcriterion.py <path_to_CFD_results_folder> <path_to_case_mesh_data> <case_name> <path_to_output_folder> <ncores>
#     
#
# INPUTS:
#   - folder       Path to results directory with HDF5 snapshots
#   - mesh         Path to the case mesh data files (with Mesh/Wall/{coordinates,topology,pointIds})
#   - case         Case name (used only in output filename)
#   - out          Output directory for the VTP result
#   - n_process    Number of worker processes (default: #logical CPUs)
#
# OUTPUTS:
#   - output_folder/<case>_Qcriterion_tstep.vtu 
#
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
import pyvista as pv

warnings.filterwarnings("ignore", category=DeprecationWarning) 


# -------------------------------- Shared-memory Utilities ---------------------------------------------

def create_shared_array(size, dtype = np.float64):
    """Create a ctypes-backed shared array filled with zeros."""
    ctype_array = np.ctypeslib.as_ctypes( np.zeros(size, dtype=dtype) )
    return sharedctypes.Array(ctype_array._type_, ctype_array,  lock=False)

def np_shared_array(shared_obj):
    """Get a NumPy view (no copy) onto a shared ctypes array created by create_shared_array."""
    return np.ctypeslib.as_array(shared_obj)


# ---------------------------------------- Mesh Utilities -----------------------------------------------------

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
    n_cells        = cells.shape[0]
    node_per_cell  = 4  # the volumetric cells are tets with size of 4 (4 nodes per elem)
    cell_types     = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    cell_size      = np.full((n_cells, 1), node_per_cell, dtype = np.int64)  # array of size (n_cells, 1) filled with 4 
    cells_vtk      = np.hstack([cell_size, cells]).ravel() # horizontal stacking of array / ravel: flattens the array into a 1d array


    # Build grid and attach points
    vol_mesh = pv.UnstructuredGrid(cells_vtk, cell_types, coords)

    return vol_mesh


# ---------------------------------------- Helper Utilities -----------------------------------------------------

def extract_timestep_from_h5(h5_file):
        """
        Extract integer timestep from filename pattern '*_ts=<int>_...'.
        Used to sort snapshot files chronologically.
        """
        return int(h5_file.stem.split('_ts=')[1].split('_')[0])



# --------------------------------- Parallel File Reader -----------------------------------------------

#### ADD THE SOLUTION TYPE TO READ AS INPUT ARGUMENT
def read_velocity_from_h5_files(file_ids, h5_files, velocity_ctype):
    """
    Reads a *chunk* of time-snapshot HDF5 files, extracts CFD solution, and writes into the shared array.

    Arguments:
      file_ids : list of snapshot indices to read
      h5_files : list of Path objects to HDF5 snapshots
      velocity_ctype  : shared ctypes array [n_points, 3, n_times]
    """
    
    # Create a shared (across processes) array of velocity time-series
    velocity = np_shared_array(velocity_ctype) # (n_points, 3, n_snapshots)
    
    
    for t_index in file_ids:
        with h5py.File(h5_files[t_index], 'r') as h5:
            U = np.asarray(h5['Solution']['u']) # shape: (n_points, 3, n_times)
            
            # For each wall point j, set shared_pressure[j, t_index] = p_wall[j]
            #for j in range(pressure_wall.shape[0]):
            #    solution_shared[j][:][t_index] = velocity[j,:]
            velocity[:,:,t_index] = U


    return velocity


def read_h5_files_parallel(n_process, snapshot_h5_files, velocity_ctype):

    n_snapshots = len(snapshot_h5_files)

    # divide all snapshot files into groups and spread across processes
    time_indices    = list(range(n_snapshots))
    time_chunk_size = max(n_snapshots // n_process, 1)
    time_groups     = [time_indices[i : i + time_chunk_size] for i in range(0, n_snapshots, time_chunk_size)]
    
    processes_list=[]
    for idx, group in enumerate(time_groups):
        proc = mp.Process(
                target = read_velocity_from_h5_files,
                name = f"Reader{idx}",
                args=(group, snapshot_h5_files, velocity_ctype))
        
        processes_list.append(proc)
        proc.start()

    # Start all readers
    #for proc in processes_list: proc.start()

    # Wait for all readers to finish
    for proc in processes_list: proc.join()

    # Free up memory
    gc.collect()







# -------------------------------- Compute Qcriterion -----------------------------------------------------


def compute_Qcriterion_parallel(vol_mesh, snapshot_h5_files, output_folder, n_process):
    """Computes Q-criterion for the given points (pids) and writes back into shared array."""
    
    n_points = vol_mesh.n_points
    n_snapshots = len(snapshot_h5_files)

    # 2) Create shared arrays
    # Array to hold velocity fiels (n_points, n_times)
    velocity_ctype = create_shared_array([n_points, 3, n_snapshots])

    read_h5_files_parallel(n_process, snapshot_h5_files, velocity_ctype)

    velocity = np_shared_array(velocity_ctype)  # (n_points, 3, n_snapshots)

    # reuse geometry arrays to avoid copying VTK objects between processes
    base_cells  = vol_mesh.cells.copy()
    base_types  = vol_mesh.celltypes.copy()
    base_points = vol_mesh.points.copy()


    for t_index in range(n_snapshots):
        if t_index%10 == 0:
            print (f'Computing Q-criterion for frame {t_index} ...')

        # Build a fresh grid per frame
        U = velocity[:, :, t_index]

        vol_mesh.point_data.clear()
        vol_mesh.point_data["Velocity"] = U
        vol_mesh.set_active_vectors("Velocity")

        qgrid_full = vol_mesh.compute_derivative(qcriterion=True, progress_bar=False)
        q_array = np.asarray(qgrid_full.point_data["qcriterion"])

        # minimal grid with ONLY Q
        minimal = pv.UnstructuredGrid(base_cells, base_types, base_points)
        minimal.point_data.clear()
        minimal.point_data["QCriterion"] = q_array

        # use timestep from filename for naming
        ts = extract_timestep_from_h5(Path(snapshot_h5_files[t_index]))
        output_file = Path(output_folder) / f"Q_{ts:06d}.vtu"
        minimal.save(str(output_file))


        # cleanup
        del qgrid_full, minimal
        gc.collect()




# ---------------------------------------- Compute Hemodynamics Metrics -------------------------------------

def hemodynamics(vol_mesh: pv.UnstructuredGrid,
                 input_folder: Path,
                 output_folder: Path,
                 case_name: str,
                 n_process: int,
                 ):
    """
    Main driver: Reads time-series pressures, computes post-metrics, and writes them to files.

    Args:
      vol_mesh       : PyVista UnstructuredGrid for the volumetric mesh
      input_folder   : folder containing '*_curcyc_*up.h5' snapshots
      output_folder  : output folder for VTP
      case_name      : prefix for output filename
      n_process      : number of processes for parallel file reading
    """


    
    # Mesh point IDs and sizes
    n_points  = len(vol_mesh.points)

    # 1) Gather files
    # Find & sort snapshot files by timestep 
    snapshot_h5_files = sorted(Path(input_folder).glob('*_curcyc_*up.h5'), key = extract_timestep_from_h5)
    n_snapshots = len(snapshot_h5_files)
 

    if n_snapshots==0:
        print('No files found in {}!'.format(input_folder))
        sys.exit()
    
    
    # 3) Reading snapshots (parallel)
    print ('Reading in parallel', n_snapshots, 'velocity files into 1 array of shape', [n_points, 3, n_snapshots],' ...')


    # 4) Compute Q-criterion (serial)
    print ('Now computing Q-criterion ...')

    compute_Qcriterion_parallel(vol_mesh, snapshot_h5_files, output_folder, n_process)

    # 6) Attach the metric to mesh and save VTU file

    
    print ('Finished calculating post metrics and save them to the files.')




# ---------------------------------------- Run the script -----------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_folder",  required=True,       help="Results folder with CFD .h5 files")
    ap.add_argument("--mesh_folder",   required=True,       help="Case mesh folder containing HDF5 mesh file")
    ap.add_argument("--case_name",     required=True,       help="Case name")
    ap.add_argument("--output_folder", required=True,       help="Output directory to save VTU files")
    ap.add_argument("--n_process",     type=int,            help="Number of parallel processes", default=max(1, mp.cpu_count() - 1))

    return ap.parse_args()


def main():
    args          = parse_args()
    input_folder  = Path(args.input_folder)
    mesh_folder   = Path(args.mesh_folder)
    output_folder = Path(args.output_folder)

    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    mesh_file = list(Path(mesh_folder).glob('*.h5'))[0]
    #print(f"Loading mesh: {mesh_file}")
    vol_mesh = assemble_volume_mesh(mesh_file)

    print(f"[info] Mesh file:    {mesh_file}")
    print(f"[info] Reading from: {input_folder}")
    print(f"[info] Writing to:   {output_folder}")
    print(f"[info] Processes:    {args.n_process} \n")



    print(f"Performing hemodynamics computation on {args.n_process} processes…")
    hemodynamics(vol_mesh, input_folder, output_folder, case_name = args.case_name, n_process = args.n_process)



if __name__ == '__main__':
    main()