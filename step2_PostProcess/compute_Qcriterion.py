# -----------------------------------------------------------------------------------------------------------------------
# compute-post-metrics_Qcriterion.py 
# To calculate Qcriterion in parallel from Oasis/BSLSolver CFD outputs.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2025-10
#
# PURPOSE:
#   - This script is part of the BSL post-processing pipeline.
#   - It computes Qcriterion at each timestep and saves the array 'Qcriterion' with the mesh to an HDF5 file.
#
# REQUIREMENTS:
#   - h5py
#   - pyvista
#   - On Trillium: virtual environment called pyvista36
#
# EXECUTION:
#   - Run using bash script:
#       > sbatch compute_Qcriterion_job.sh
#   - Run directly on a login/debug node as below:
#       > module load StdEnv/2023 gcc/12.3 python/3.12.4
#       > source $HOME/virtual_envs/pyvista36/bin/activate
#       > module load  vtk/9.3.0
#       > python compute_Qcriterion.py <path_to_CFD_results_folder> <path_to_case_mesh_data> <case_name> <path_to_output_folder> <ncores>
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
###############################################################################################
def create_shared_array(size, dtype = np.float32):
    """Create a ctypes-backed shared array filled with zeros."""
    ctype_array = np.ctypeslib.as_ctypes( np.zeros(size, dtype=dtype) )
    return sharedctypes.Array(ctype_array._type_, ctype_array,  lock=False)

###############################################################################################
def np_shared_array(shared_obj):
    """Get a NumPy view (no copy) onto a shared ctypes array created by create_shared_array."""
    return np.ctypeslib.as_array(shared_obj)


# ---------------------------------------- I/O Utilities -----------------------------------------------------
###############################################################################################
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

###############################################################################################
def extract_timestep_from_h5(h5_filename):
        """
        Extract (integer_timestep, physical_time) from filename pattern '*t=<float>_ts=<int>_up.h5'.
        Used to sort snapshot files chronologically.
        """
        name = str(h5_filename)

        tstep_str = name.split("_ts=")[1].split("_")[0]
        tstep = int(tstep_str)

        time_str = name.split("_t=")[1].split("_")[0]
        time = float(time_str)

        return (tstep, time)

###############################################################################################
def create_h5_output(vol_mesh, topology, n_points: int, n_snapshots: int, output_h5_path: Path):
    """
    Creates one HDF5 file with the mesh (constant) and pre-allocate Time and Q arrays.

    Groups & datasets:
      /Mesh/coordinates : (n_points,3)            # coordinates of mesh
      /Mesh/topology    : (n_cells,4)             # connectivity of mesh
      /Time/values      : (n_snapshots,)          # physical time per frame
      /Data/Q           : (n_points,n_snapshots)  # Q at points; column t is frame t
    """

    f = h5py.File(output_h5_path, "w")

    # Ensure Mesh group exists, then write exact arrays with proper shapes
    group_mesh = f.create_group("Mesh")
    coords = np.asarray(vol_mesh.points, dtype=np.float64)           # (Npoints, 3)
    topo   = np.asarray(topology, dtype=np.int32)                    # (Ncells, 4)

    group_mesh.create_dataset("coordinates", data=coords, dtype="f8", compression="gzip",shuffle=True) # (n_points, 3)
    group_mesh.create_dataset("topology", data=topo, dtype="i4", compression="gzip",shuffle=True) # (n_points, 3)
   

    # Initialize time values placeholder -> to be filled later
    f.create_dataset("Time/values", shape=(n_snapshots,), dtype="f8")

    # Initialize Q placeholder -> to be filled later -> shape (n_points, n_snapshots)
    q_dataset = f.create_dataset(
            "Data/Q",
            shape=(n_points, n_snapshots),
            dtype="f4",
            chunks=(n_points,1),
            compression="gzip",shuffle=True)

    return f


###############################################################################################
def write_xdmf_for_h5(h5_path: Path, xdmf_path: Path, series_name: str = "TimeSeries", attr_name: str = "QCriterion", topo_type: str = "Tetrahedron"):
    """
    Generate an XDMF v3 file:
      - defines a base 'mesh' Grid reading /Mesh/topology and /Mesh/coordinates from the same HDF5.
      - creates a Temporal collection where each step slices /Data/Q with a HyperSlab Layout assumed: /Data/Q has shape (Npoints, Nt)  [column = one time slice]
      - For step t, HyperSlab selects start=(0, t), stride=(1,1), count=(Npoints,1)
      - Time values are read from /Time/values if present; otherwise use integer t.
    """
    h5_path   = Path(h5_path)
    xdmf_path = Path(xdmf_path)
    h5_name   = h5_path.name

    with h5py.File(h5_path, "r") as f:
        coords = np.asarray(f["Mesh/coordinates"])
        topo   = np.asarray(f["Mesh/topology"])
        npoints = int(coords.shape[0])
        ncells  = int(topo.shape[0])
        nverts  = int(topo.shape[1])

        Q = f["Data/Q"]
        if Q.ndim != 2:
            raise RuntimeError("Expected /Data/Q to be 2D (Npoints, Nt).")
        
        npoints_q, nt = int(Q.shape[0]), int(Q.shape[1])
        
        if npoints_q != npoints:
            raise RuntimeError(f"/Data/Q first dim ({npoints_q}) should be equal to Mesh/coordinates length ({npoints}).")

        # Time values (optional)
        if "Time" in f and "values" in f["Time"]:
            time_vals = np.asarray(f["Time/values"]).astype(float).tolist()
            if len(time_vals) != nt:
                # fallback to indices if mismatch
                time_vals = [float(t) for t in range(nt)]
        else:
            time_vals = [float(t) for t in range(nt)]

    header = f'''<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="{series_name}" GridType="Collection" CollectionType="Temporal">
      <Grid Name="mesh" GridType="Uniform">
        <Topology NumberOfElements="{ncells}" TopologyType="{topo_type}" NodesPerElement="{nverts}">
          <DataItem Dimensions="{ncells} {nverts}" NumberType="Int" Format="HDF">{h5_name}:/Mesh/topology</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{npoints} 3" Format="HDF">{h5_name}:/Mesh/coordinates</DataItem>
        </Geometry>
      </Grid>
'''
    grids = []
    for t in range(nt):
        tval = time_vals[t]
        # HyperSlab parameters for selecting column t from (Npoints, Nt):
        # start=(0,t), stride=(1,1), count=(Npoints,1)
        grids.append(f'''      <Grid Name="T{t:06d}">
        <xi:include xpointer="xpointer(//Grid[@Name=&quot;{series_name}&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />
        <Time Value="{tval:.15g}" />
        <Attribute Name="{attr_name}" AttributeType="Scalar" Center="Node">
          <DataItem ItemType="HyperSlab" Dimensions="{npoints} 1" Type="HyperSlab">
            <DataItem Dimensions="3 2" NumberType="Int"> 0 {t}  1 1  {npoints} 1 </DataItem>
            <DataItem Format="HDF" NumberType="Float" Precision="4" Dimensions="{npoints} {nt}">{h5_name}:/Data/Q</DataItem>
          </DataItem>
        </Attribute>
      </Grid>
''')
    footer = '''    </Grid>
  </Domain>
</Xdmf>
'''
    xdmf_text = header + "".join(grids) + footer
    xdmf_path.write_text(xdmf_text)



# -------------------------------- Compute Qcriterion -----------------------------------------------------

###############################################################################################
def compute_Qcriterion_from_h5_files_parallel(file_ids, h5_files, vol_mesh, q_ctype):
    """
    Computes Q for a subset of timesteps from HDF5 files and write results into a shared matrix.
    For each assigned t_index it reads '/Solution/u' from h5 file, compute Q-criterion, and write it into column t_index of the shared Q matrix.

    Arguments:
      file_ids : list[int] of snapshot indices assigned to this worker to read
      h5_files : list[Path] of path to snapshot h5 files
      vol_mesh : pv.UnstructuredGrid
      q_ctype  : shared ctype array to store Q-criterion
    """
    
    # Create a shared (across processes) array of Q-criterion time-series
    q_array = np_shared_array(q_ctype) # (n_points, n_snapshots)
    
    for t_index in file_ids:

        ts, t_val = extract_timestep_from_h5(h5_files[t_index])

        with h5py.File(h5_files[t_index], 'r') as h5:
            U = np.asarray(h5['Solution']['u']) # shape: (n_points, 3)
            
            vol_mesh.point_data.clear()
            vol_mesh.point_data["Velocity"] = U
            vol_mesh.set_active_vectors("Velocity")

            q_grid = vol_mesh.compute_derivative(qcriterion=True, progress_bar=False)

            # Store Q for this frame into the shared (Npoints,Nt) matrix (column-major per time)
            q_array[:,t_index] = np.asarray(q_grid.point_data["qcriterion"], dtype=np.float32)
            

# ---------------------------------------- Compute Hemodynamics Metrics -------------------------------------
###############################################################################################
def hemodynamics(vol_mesh: pv.UnstructuredGrid,
                 mesh_topology: np.array,
                 input_folder: Path,
                 output_folder: Path,
                 case_name: str,
                 n_process: int,
                 ):
    """
    Main driver: Reads time-series CFD results, computes post-metrics, and writes them to files.

    Args:
      vol_mesh       : PyVista UnstructuredGrid for the volumetric mesh
      input_folder   : folder containing '*_curcyc_*up.h5' snapshots
      output_folder  : output folder for VTP
      case_name      : prefix for output filename
      n_process      : number of processes for parallel file reading
    """

    # 1) Initialize:
    
    # Gather files: Find & sort snapshot files by timestep 
    snapshot_h5_files = sorted(Path(input_folder).glob('*_curcyc_*up.h5'), key = extract_timestep_from_h5)
    
    n_snapshots = len(snapshot_h5_files) # number of time frames stored 
    n_points  = len(vol_mesh.points)     # number of nodes in the mesh
 
    if n_snapshots==0:
        print('No files found in {}!'.format(input_folder))
        sys.exit()


    # collect physical times (or fallback)
    time_values = np.zeros((n_snapshots,), dtype=np.float64)
    for t_index, p in enumerate(snapshot_h5_files):
      _, t_val = extract_timestep_from_h5(p)
      time_values[t_index] = t_val
    

    
    # 2) Compute Qcriterion (parallel):
    print (f'Computing Qcriterion in parallel for {n_snapshots} snapshots ... \n')
    
    # Create a shared (across processes) array of Q-criterion time-series
    q_ctype = create_shared_array([n_points, n_snapshots]) # (n_points, n_snapshots)

    # Split all snapshot files (timesteps) into groups and spread across processes (one group per process)
    time_indices    = list(range(n_snapshots))
    time_chunk_size = max(n_snapshots // n_process, 1)
    time_groups     = [time_indices[i : i + time_chunk_size] for i in range(0, n_snapshots, time_chunk_size)]
    
    processes_list=[]
    for idx, group in enumerate(time_groups):
        proc = mp.Process(
                target = compute_Qcriterion_from_h5_files_parallel,
                name = f"Qcriterion{idx}",
                args=(group, snapshot_h5_files, vol_mesh, q_ctype))
        
        processes_list.append(proc)
        proc.start()


    # Start all readers
    #for proc in processes_list: proc.start()

    # Wait for all readers to finish
    for proc in processes_list: proc.join()

    # Free up memory
    gc.collect()


    # 3) Save the output:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_h5_path   = output_folder / f"{case_name}_Qcriterion.h5"   
    output_xdmf_path = output_folder / f"{case_name}_Qcriterion.xdmf"

    # Create output HDF5 file
    h5_dataset = create_h5_output(vol_mesh, mesh_topology, n_points, n_snapshots, output_h5_path)
    #q_dataset  = h5_dataset["Data/Q"]

    # Write to H5 file
    print(f'\nWriting Q-criterion to HDF5 output file into an array of shape [{n_points}, {n_snapshots}] ... \n')
    q_array = np_shared_array(q_ctype)
    h5_dataset["Data/Q"][:] = q_array.astype(np.float64, copy=False)
    h5_dataset["Time/values"][:] = time_values
    h5_dataset.close()

    # Write XDMF index for ParaView (reuses the mesh from the same H5; one attribute per timestep)
    print('Writing XDMF file ... \n')
    write_xdmf_for_h5(output_h5_path, output_xdmf_path, series_name="TimeSeries", attr_name="QCriterion")

    
    print ('Finished calculating the post metrics and saving to the files.')




# ---------------------------------------- Run the script -----------------------------------------------------
###############################################################################################
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_folder",  required=True,       help="Results folder with CFD .h5 files")
    ap.add_argument("--mesh_folder",   required=True,       help="Case mesh folder containing HDF5 mesh file")
    ap.add_argument("--case_name",     required=True,       help="Case name")
    ap.add_argument("--output_folder", required=True,       help="Output directory to save VTU files")
    ap.add_argument("--n_process",     type=int,            help="Number of parallel processes", default=max(1, mp.cpu_count() - 1))

    return ap.parse_args()

###############################################################################################
def main():
    args          = parse_args()
    input_folder  = Path(args.input_folder)
    mesh_folder   = Path(args.mesh_folder)
    output_folder = Path(args.output_folder)

    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    mesh_file = list(Path(mesh_folder).glob('*.h5'))[0]
    #print(f"Loading mesh: {mesh_file}")
    vol_mesh, mesh_topology = assemble_volume_mesh(mesh_file)

    print(f"[info] Mesh file:    {mesh_file}")
    print(f"[info] Reading from: {input_folder}")
    print(f"[info] Writing to:   {output_folder}")
    print(f"[info] Processes:    {args.n_process} \n")



    #print(f"Performing hemodynamics computation on {args.n_process} processes ...")
    hemodynamics(vol_mesh, mesh_topology, input_folder, output_folder, case_name = args.case_name, n_process = args.n_process)



if __name__ == '__main__':
    main()