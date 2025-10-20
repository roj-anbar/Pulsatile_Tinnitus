import sys
import gc
import h5py
import warnings
import numpy as np
import pyvista as pv
from pathlib import Path

from post_utils_parallel import *

# ---------------------------------------- Mesh Utilities -----------------------------------------------------

def assemble_wall_mesh(mesh_file):
    """
    Create a Pyvista PolyData surface from the wall mesh stored in a BSLSolver-style HDF5.

    Expects datasets:
      Mesh/Wall/coordinates : (Npoints, 3) float
      Mesh/Wall/topology    : (Ncells, 3 or 4) int (triangles expected)
      Mesh/Wall/pointIds    : (Npoints,) int  (mapping back to volume point numbering)
    """
    
    with h5py.File(mesh_file, 'r') as hf:
        wall_coords = np.array(hf['Mesh/Wall/coordinates'])  # coords of wall points (n_points, 3)
        wall_elems  = np.array(hf['Mesh/Wall/topology'])     # connectivity of wall points (n_cells, 3) -> triangles
        wall_pids   = np.array(hf['Mesh/Wall/pointIds'])     # mapping to volume point IDs (n_points,)
        
    # Create VTK connectivity --> requires a size prefix per cell (here '3' for triangles)
    n_elems      = wall_elems.shape[0]
    tri_size     = np.ones((n_elems, 1), dtype=int) * 3 # array of 3
    vtk_elems    = np.concatenate([tri_size, wall_elems], axis = 1).ravel() #ravel(): flattens the array into a 1d array
        
    # Build surface and attach point ID
    surf = pv.PolyData(wall_coords, vtk_elems)
    surf.point_data['vtkOriginalPtIds'] = wall_pids

    return surf


# ---------------------------------------- Helper Utilities -----------------------------------------------------

def extract_timestep_from_h5(h5_file):
        """
        Extract integer timestep from filename pattern '*_ts=<int>_...'.
        Used to sort snapshot files chronologically.
        """
        return int(h5_file.stem.split('_ts=')[1].split('_')[0])



# --------------------------------- Parallel File Reader -----------------------------------------------


def read_h5_files(file_ids, wall_pids, h5_files, shared_pressure_ctype):
    """
    Reads a *chunk* of time-snapshot HDF5 files, extracts wall pressures, and writes into the shared array.

    Arguments:
      file_ids   : list of snapshot indices to read
      wall_pids  : wall point indices (to slice pressure field)
      h5_files : list of Path objects to HDF5 snapshots
      shared_pressure_ctype  : shared ctypes array; viewed as press_[n_points, n_times]
    """
    
    # Create a shared (across processes) array of wall-pressure time-series
    shared_pressure = np_shared_array(shared_pressure_ctype)
    
    for t_index in file_ids:
        with h5py.File(h5_files[t_index], 'r') as h5:
            pressure = np.array(h5['Solution']['p'])
            pressure_wall = pressure[wall_pids].flatten() # shape: (n_points,)
            
            # For each wall point j, set shared_pressure[j, t_index] = p_wall[j]
            for j in range(pressure_wall.shape[0]):
                shared_pressure[j][t_index] = pressure_wall[j]

            #shared_pressure[:, t_index] = pressure_wall 

