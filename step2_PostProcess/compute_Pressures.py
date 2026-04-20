# -----------------------------------------------------------------------------------------------------------------------
# compute_PressureDrop.py
# Computes pressure drop from inlet to outlet and plots it as a function of inlet flowrate.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2026-04
#
# PURPOSE:
#   - Reads CFD HDF5 snapshots, extracts pressure at a single inlet and outlet point,
#     computes ΔP = P_inlet − P_outlet over time, converts time to inlet flowrate (Q = 2t),
#     and saves the data and a plot.
#
# REQUIREMENTS:
#   - h5py, numpy, matplotlib
#   - On Trillium: virtual environment called "pyvista36"
#
# EXECUTION:
#   - Run using "compute_PressureDrop_job.sh" bash script.
#   - Run directly on a login/debug node:
#       > module load StdEnv/2023 gcc/12.3 python/3.12.4
#       > source $HOME/virtual_envs/pyvista36/bin/activate
#
# EXAMPLE CLI:
#       > python compute_PressureDrop.py \
#           --input_folder   <path_to_CFD_results_folder> \
#           --mesh_folder    <path_to_case_mesh_data>     \
#           --output_folder  <path_to_output_folder>      \
#           --case_name      PTSeg028_base_0p64           \
#           --centerline_csv <path_to_centerline_CSV>     \
#           --inlet_point_id  5                           \
#           --outlet_point_id 95                          \
#
# INPUTS:
#   --input_folder      Path to results directory with HDF5 snapshots
#   --mesh_folder       Path to the case mesh data files (with Mesh/coordinates)
#   --output_folder     Path to output directory
#   --case_name         Case name (used in output filenames)
#   --centerline_csv    Path to centerline CSV with columns Points:0, Points:1, Points:2
#   --inlet_point_id    Row index into centerline CSV for the inlet point
#   --outlet_point_id   Row index into centerline CSV for the outlet point
#   --density           Blood density [kg/m³] (default: 1057)
#   --period_seconds    Flow period [s] (default: 0.915)
#   --timesteps_per_cyc Timesteps per cycle (parsed from filename if omitted)
#   --flowrate_min      Lower flowrate limit for plot x-axis [mL/s] (default: 2.0)
#   --flowrate_max      Upper flowrate limit for plot x-axis [mL/s] (default: 10.0)
#
# OUTPUTS:
#   - <case_name>_pressure_drop.npz  — Q_in, dP, P_inlet, P_outlet arrays
#   - <case_name>_pressure_drop.png  — ΔP and individual pressures vs Q_inlet
#
# NOTES:
#   - Pressure in Oasis HDF5 is p/rho; multiplied by density to get Pa.
#   - Inlet flowrate is derived from time as Q_inlet = 2 * t [mL/s] (PT-Ramp ramp slope).
#   - Nearest volume mesh node to each centerline coordinate is found via KDTree.
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------

import sys
import re
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
import multiprocessing as mp
from multiprocessing import sharedctypes
from pathlib import Path


# ======================================================================================================
# UTILITIES
# ======================================================================================================

def extract_timestep_from_h5_filename(h5_file: Path) -> int:
    """Extract integer timestep from filename pattern '*_ts=<int>_...' — used as sort key."""
    match = re.search(r'_ts=(\d+)', h5_file.stem)
    if match is None:
        raise ValueError(f"Filename '{h5_file.name}' does not contain expected '_ts=<int>' pattern.")
    return int(match.group(1))


def extract_sim_params_from_h5_filename(h5_file: Path) -> int:
    """Parse timesteps-per-cycle from filename pattern '_ts<int>'."""
    match = re.search(r'_ts(\d+)', h5_file.stem)
    if match is None:
        raise ValueError(
            f"Filename '{h5_file.name}' has no '_ts<int>' pattern. "
            "Supply --timesteps_per_cyc on the CLI instead."
        )
    return int(match.group(1))


def read_centerline_coords_from_csv(csv_path: str) -> np.ndarray:
    """
    Read a centerline CSV (same format as ROI center CSV in compute_Spectrogram.py).
    Expected columns: Points:0, Points:1, Points:2
    Returns (n_points, 3) XYZ coordinate array.
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    x = data['Points0']
    y = data['Points1']
    z = data['Points2']
    return np.vstack([x, y, z]).T


def load_volume_mesh_coords(mesh_file: Path) -> np.ndarray:
    """Load volume mesh node coordinates from the BSLSolver HDF5 mesh file.

    Expects:
      Mesh/coordinates : (Npoints, 3) float
    """
    with h5py.File(mesh_file, 'r') as h5:
        coords = np.array(h5['Mesh/coordinates'])
    return coords


def find_nearest_volume_node(vol_coords: np.ndarray, query_point: np.ndarray) -> int:
    """Return the global volume mesh node index nearest to query_point."""
    dists = np.sum((vol_coords - query_point) ** 2, axis=1)
    return int(np.argmin(dists))


def find_nearest_volume_nodes(vol_coords: np.ndarray, query_points: np.ndarray) -> list:
    """Return volume mesh node indices nearest to each row in query_points."""
    return [find_nearest_volume_node(vol_coords, pt) for pt in query_points]


def compute_centerline_distances(centerline_coords: np.ndarray, inlet_point_id: int) -> np.ndarray:
    """Cumulative arc-length distance along the centerline, zeroed at inlet_point_id [mm].

    centerline_coords : (n_points, 3) in [mm].
    inlet_point_id    : row index of the inlet in centerline_coords.
    Returns (n_points,) distances in mm; negative = upstream of inlet.
    """
    diffs = np.diff(centerline_coords, axis=0)              # (n-1, 3) [mm]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    return cumulative - cumulative[inlet_point_id]


def _create_shared_array(size):
    """Create a ctypes-backed shared array filled with zeros."""
    ctype_array = np.ctypeslib.as_ctypes(np.zeros(size, dtype=np.float64))
    return sharedctypes.Array(ctype_array._type_, ctype_array, lock=False)

def _view_shared_array(shared_obj):
    """Get a NumPy view (no copy) onto a shared ctypes array."""
    return np.ctypeslib.as_array(shared_obj)


def _read_pressure_worker(file_ids, node_ids_arr, h5_files, shared_pressure_ctype, density):
    """Worker: read a chunk of snapshots and write into the shared (n_nodes, n_times) array."""
    shared_pressure = _view_shared_array(shared_pressure_ctype)
    for t_index in file_ids:
        with h5py.File(h5_files[t_index], 'r') as h5:
            shared_pressure[:, t_index] = np.array(h5['Solution']['p']).flatten()[node_ids_arr] * density


def read_pressure_timeseries_at_nodes(h5_files: list, node_ids: list, density: float, n_process: int) -> np.ndarray:
    """
    Read pressure at the given volume node IDs for every HDF5 snapshot in parallel.

    Spawns n_process workers, each reading a contiguous time chunk into shared memory.
    Oasis stores p/rho; multiplied by density to get Pa.

    Returns
    -------
    pressures : (n_nodes, n_snapshots) float array [Pa]
    """
    n_snapshots  = len(h5_files)
    n_nodes      = len(node_ids)
    node_ids_arr = np.array(node_ids)

    shared_pressure_ctype = _create_shared_array([n_nodes, n_snapshots])

    print(f"\nReading pressure at {n_nodes} nodes from {n_snapshots} HDF5 snapshots in parallel ...\n")

    time_indices    = list(range(n_snapshots))
    chunk_size      = max(n_snapshots // n_process, 1)
    time_groups     = [time_indices[i : i + chunk_size] for i in range(0, n_snapshots, chunk_size)]

    processes = []
    for idx, group in enumerate(time_groups):
        proc = mp.Process(
            target = _read_pressure_worker,
            name   = f"Reader{idx}",
            args   = (group, node_ids_arr, h5_files, shared_pressure_ctype, density))
        processes.append(proc)

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    return _view_shared_array(shared_pressure_ctype).copy()  # (n_nodes, n_snapshots)


# ======================================================================================================
# PLOTTING
# ======================================================================================================


def plot_centerline(output_folder: Path, case_name: str,
                mesh_file: Path, 
                centerline_coords: np.ndarray, cl_dist: np.ndarray,
                inlet_point_id: int):
    """
    Render centerline points colored by distance from inlet using matplotlib 3D scatter.

    Parameters
    ----------
    centerline_coords : (n_cl_points, 3) centerline coordinates [mm]
    cl_dist           : (n_cl_points,) distance from inlet [mm]
    inlet_point_id    : row index of inlet in centerline_coords
    """
    out_png = output_folder / f"{case_name}_geometryCenterline.png"

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


    fig = plt.figure(figsize=(9, 9))
    ax  = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(centerline_coords[:, 0], centerline_coords[:, 1], centerline_coords[:, 2], c=cl_dist, cmap='hot', s=18, zorder=5, linewidths=0)
    inlet_coord = centerline_coords[inlet_point_id]
    ax.scatter(*inlet_coord, color='red', s=80, zorder=10)
    ax.text(*inlet_coord, '  Inlet', color='red', fontsize=10)
    fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label='Distance from Inlet (mm)')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    ax.set_title(f'{case_name} — Centerlines', fontweight='bold')
    ax.view_init(elev=30, azim=120)   # 180° flip from matplotlib default (azim=-60)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved centerline geometry  ->  {out_png}  (matplotlib fallback)")




def plot_pressure_drop(output_folder: Path, case_name: str,
                       Q_in: np.ndarray, dP: np.ndarray,
                       P_inlet: np.ndarray, P_outlet: np.ndarray,
                       plot_params: dict):
    """
    Plot ΔP (inlet − outlet) and individual pressures vs inlet flowrate.
    Saves a PNG to output_folder.
    """
    Q_min = plot_params.get("Q_min", Q_in.min())
    Q_max = plot_params.get("Q_max", Q_in.max())

    font_size = 14
    plt.rc('font',   size=font_size)
    plt.rc('axes',   titlesize=font_size)
    plt.rc('xtick',  labelsize=font_size)
    plt.rc('ytick',  labelsize=font_size)
    plt.rc('legend', fontsize=font_size - 2)
    plt.rc('axes',   labelsize=font_size)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle(f'{case_name}: Pressure Drop', fontweight='bold')

    # -------------- Subplot1: Pressure drop -----------------------
    ax.plot(Q_in, dP/133.3, linewidth = 1, color='blue')
    ax.set_xlabel('Inlet Flow Rate (mL/s)',   fontweight='bold', labelpad=10)
    ax.set_ylabel('dP  (mmHg)',               fontweight='bold', labelpad=10)
    ax.set_xlim([Q_min, Q_max])
    ax.set_ylim([0, 60])
    ax.grid(True, alpha=0.3)

    # -------------- Subplot2: Pressure values -----------------------
    # ax[1].plot(Q_in, P_inlet/133.3,  linewidth = 2, color='black')
    # ax[1].plot(Q_in, P_outlet/133.3, linewidth = 2,  color='red')
    # ax[1].set_xlabel('Inlet Flow Rate (mL/s)',  fontweight='bold', labelpad=10)
    # ax[1].set_ylabel('Pressure (mmHg)',         fontweight='bold', labelpad=10)
    # ax[1].set_xlim([Q_min, Q_max])
    #ax[1].set_ylim([0, 60])


    plt.tight_layout()
    out_png = output_folder / f"{case_name}_pressureInOut.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot  ->  {out_png}")


def animate_centerline_pressure(output_folder: Path, case_name: str,
                                pressures_all: np.ndarray, Q_in: np.ndarray,
                                cl_dist: np.ndarray, frame_stride: int):
    """
    Animate P (mmHg) vs. distance from inlet stepping every frame_stride snapshots.

    Parameters
    ----------
    pressures_all : (n_cl_points, n_snapshots) float [Pa]
    Q_in          : (n_snapshots,) float [mL/s]
    cl_dist       : (n_cl_points,) cumulative arc-length from inlet [mm]
    frame_stride  : int — use every Nth snapshot as an animation frame
    """
    n_cl_points, n_snapshots = pressures_all.shape
    frame_indices = list(range(0, n_snapshots, frame_stride))
    P_mmHg        = pressures_all / 133.3                     # Pa -> mmHg

    P_min = float(P_mmHg.min())
    P_max = float(P_mmHg.max())
    pad   = max((P_max - P_min) * 0.05, 1.0)

    font_size = 13
    plt.rc('font',   size=font_size)
    plt.rc('axes',   labelsize=font_size)
    plt.rc('xtick',  labelsize=font_size)
    plt.rc('ytick',  labelsize=font_size)

    fig, ax = plt.subplots(figsize=(10, 6))
    (line,)  = ax.plot([], [], linewidth=1.5, color='steelblue')
    ax.set_xlim(cl_dist[0], cl_dist[-1])
    ax.set_ylim(P_min - pad, P_max + pad)
    ax.set_xlabel('Distance from Inlet (mm)', fontweight='bold', labelpad=8)
    ax.set_ylabel('Pressure (mmHg)',          fontweight='bold', labelpad=8)
    ax.grid(True, alpha=0.3)
    title_obj = ax.set_title('')
    fig.tight_layout()

    def _init():
        line.set_data([], [])
        title_obj.set_text('')
        return line, title_obj

    def _update(anim_frame):
        t_idx = frame_indices[anim_frame]
        line.set_data(cl_dist, P_mmHg[:, t_idx])
        title_obj.set_text(
            f'{case_name}  |  Q_inlet = {Q_in[t_idx]:.2f} mL/s  '
            f'(snapshot {t_idx} / {n_snapshots - 1})'
        )
        return line, title_obj

    anim = manim.FuncAnimation(
        fig, _update,
        frames   = len(frame_indices),
        init_func= _init,
        blit     = True,
        interval = 100,
    )

    out_mp4 = output_folder / f"{case_name}_pressureCenterline.mp4"
    try:
        writer = manim.FFMpegWriter(fps=10, metadata=dict(title=case_name), bitrate=1800)
        anim.save(str(out_mp4), writer=writer, dpi=150)
        print(f"Saved animation  ->  {out_mp4}")
    except Exception:
        out_gif = output_folder / f"{case_name}_centerline_pressure_anim.gif"
        print("ffmpeg not available — falling back to GIF")
        anim.save(str(out_gif), writer='pillow', fps=10, dpi=120)
        print(f"Saved animation  ->  {out_gif}")
    finally:
        plt.close(fig)


def plot_centerline_pressure_3d(output_folder: Path, case_name: str,
                                pressures_all: np.ndarray, Q_in: np.ndarray,
                                cl_dist: np.ndarray, frame_stride: int):
    """
    3D surface plot: x = distance from inlet (mm), y = Q_inlet (mL/s), z = pressure (mmHg).

    The time axis is subsampled by frame_stride to keep the surface tractable.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3D projection

    _n_cl_points, n_snapshots = pressures_all.shape
    P_mmHg = pressures_all / 133.3

    t_idx = np.arange(0, n_snapshots, frame_stride)
    P_sub = P_mmHg[:, t_idx]           # (n_cl_points, n_t_sub)
    Q_sub = Q_in[t_idx]                 # (n_t_sub,)

    X, Y = np.meshgrid(cl_dist, Q_sub, indexing='ij')   # (n_cl_points, n_t_sub)

    fig = plt.figure(figsize=(13, 8))
    ax  = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, P_sub, cmap='viridis',
                           linewidth=0, antialiased=True, alpha=0.92)

    ax.set_xlabel('Distance from Inlet (mm)', fontweight='bold', labelpad=12)
    ax.set_ylabel('Inlet Flow Rate$ (mL/s)', fontweight='bold', labelpad=12)
    ax.set_zlabel('Pressure (mmHg)',    fontweight='bold', labelpad=12)
    ax.set_title(f'{case_name}: Centerline Pressure', fontweight='bold', pad=14)

    fig.colorbar(surf, ax=ax, shrink=0.45, aspect=12, pad=0.1,
                 label='Pressure (mmHg)')

    out_png = output_folder / f"{case_name}_pressureCenterline3D.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3D plot    ->  {out_png}")


# ======================================================================================================
# MAIN
# ======================================================================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Compute pressure drop (inlet to outlet) vs inlet flowrate.")
    ap.add_argument("--input_folder",      required=True,              help="Folder with CFD HDF5 snapshots")
    ap.add_argument("--mesh_folder",       required=True,              help="Folder with mesh HDF5 file")
    ap.add_argument("--output_folder",     required=True,              help="Output directory for results")
    ap.add_argument("--case_name",         required=True,              help="Case name (used in output filenames)")
    ap.add_argument("--centerline_csv",    required=True,              help="CSV file with centerline point coordinates (Points:0/1/2)")
    ap.add_argument("--inlet_point_id",    type=int,   required=True,  help="Row index into centerline CSV for the inlet point")
    ap.add_argument("--outlet_point_id",   type=int,   required=True,  help="Row index into centerline CSV for the outlet point")
    ap.add_argument("--density",           type=float, default=1057,   help="Blood density [kg/m3] (default: 1057)")
    ap.add_argument("--period_seconds",    type=float, default=0.915,  help="Flow period in seconds (default: 0.915)")
    ap.add_argument("--timesteps_per_cyc", type=int,   default=None,   help="Timesteps per cycle (parsed from filename if omitted)")
    ap.add_argument("--save_freq",         type=int,   default=1,      help="Snapshot save frequency: 1 = every timestep, 5 = every 5th timestep, etc. (default: 1)")
    ap.add_argument("--flowrate_min",      type=float, default=2.0,    help="Lower inlet flowrate limit for plot x-axis [mL/s] (default: 2.0)")
    ap.add_argument("--flowrate_max",      type=float, default=10.0,   help="Upper inlet flowrate limit for plot x-axis [mL/s] (default: 10.0)")
    ap.add_argument("--n_process",         type=int,   default=max(1, mp.cpu_count() - 1), help="Number of parallel reader processes (default: #CPUs - 1)")
    ap.add_argument("--frame_stride",      type=int,   default=20,    help="Animation frame stride: use every Nth snapshot (default: 20)")
    return ap.parse_args()


def main():
    args = parse_args()

    input_folder  = Path(args.input_folder)
    mesh_folder   = Path(args.mesh_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("compute_PressureDrop.py")
    print(f"  Input folder    : {input_folder}")
    print(f"  Mesh folder     : {mesh_folder}")
    print(f"  Output folder   : {output_folder}")
    print(f"  Centerline CSV  : {args.centerline_csv}")
    print(f"  Inlet  point ID : {args.inlet_point_id}")
    print(f"  Outlet point ID : {args.outlet_point_id}")
    print("=" * 120 + "\n")

    # ---- Load volume mesh coordinates ----
    mesh_file  = list(mesh_folder.glob('*.h5'))[0]
    vol_coords = load_volume_mesh_coords(mesh_file)
    print(f"Loaded volume mesh: {vol_coords.shape[0]} nodes  ({mesh_file.name})\n")

    # ---- Identify inlet / outlet volume nodes from centerline CSV ----
    centerline_coords = read_centerline_coords_from_csv(args.centerline_csv)
    inlet_coord  = centerline_coords[args.inlet_point_id]
    outlet_coord = centerline_coords[args.outlet_point_id]

    inlet_vol_id  = find_nearest_volume_node(vol_coords, inlet_coord)
    outlet_vol_id = find_nearest_volume_node(vol_coords, outlet_coord)

    print(f"Inlet  centerline coord (row {args.inlet_point_id:4d}) : {inlet_coord}  ->  volume node {inlet_vol_id}")
    print(f"Outlet centerline coord (row {args.outlet_point_id:4d}) : {outlet_coord}  ->  volume node {outlet_vol_id}\n")

    # ---- Map ALL centerline points to nearest volume nodes ----
    print(f"Mapping all {len(centerline_coords)} centerline points to volume mesh nodes ...")
    all_vol_node_ids = find_nearest_volume_nodes(vol_coords, centerline_coords)
    cl_dist = compute_centerline_distances(centerline_coords, args.inlet_point_id)
    print(f"Centerline length: {cl_dist[-1]:.1f} mm\n")

    # ---- Find & sort CFD snapshot HDF5 files ----
    CFD_h5_files = sorted(input_folder.glob('*_curcyc_*up.h5'), key=extract_timestep_from_h5_filename)
    if not CFD_h5_files:
        print(f"No CFD snapshots found in {input_folder}!")
        sys.exit(1)

    # ---- Parse temporal parameters ----
    timesteps_per_cyc = args.timesteps_per_cyc
    if timesteps_per_cyc is None:
        timesteps_per_cyc = extract_sim_params_from_h5_filename(CFD_h5_files[0])
        print(f"Parsed timesteps_per_cycle = {timesteps_per_cyc} from filename.\n")

    # ---- Read pressure time series at ALL centerline nodes ----
    pressures_all = read_pressure_timeseries_at_nodes(CFD_h5_files, all_vol_node_ids, args.density, args.n_process)   # (n_cl_points, n_snapshots) [Pa]

    # ---- Extract inlet / outlet from the full array ----
    P_inlet  = pressures_all[args.inlet_point_id,  :]   # (n_snapshots,) [Pa]
    P_outlet = pressures_all[args.outlet_point_id, :]   # (n_snapshots,) [Pa]

    # ---- Compute pressure drop ----
    dP = P_inlet - P_outlet      # (n_snapshots,) [Pa]

    # ---- Build inlet flowrate array (Q = 2*t, ramp-specific) ----
    n_snapshots = len(CFD_h5_files)
    dt          = args.save_freq * args.period_seconds / timesteps_per_cyc  # [s] per saved snapshot
    time_array  = np.arange(n_snapshots) * dt                               # [s]
    Q_in        = 2.0 * time_array                                          # [mL/s]

    print(f"\nQ_inlet range: {Q_in.min():.2f} to {Q_in.max():.2f} mL/s  ({n_snapshots} snapshots)")
    print(f"dP range     : {dP.min():.2f} to {dP.max():.2f} Pa\n")

    # ---- Save numerical results ----
    out_npz = output_folder / f"{args.case_name}_pressureInOut.npz"
    np.savez(out_npz,
             Q_in=Q_in, dP=dP, P_inlet=P_inlet, P_outlet=P_outlet,
             pressures_all=pressures_all,
             inlet_vol_id=np.array(inlet_vol_id),  outlet_vol_id=np.array(outlet_vol_id),
             inlet_coord=inlet_coord,               outlet_coord=outlet_coord)
    print(f"Saved data  ->  {out_npz}")


    # ---- Geometry reference: surface mesh + centerline colored by distance ----
    print("\nGenerating geometry image with centerlines...")
    plot_centerline(output_folder, args.case_name, mesh_file, centerline_coords, cl_dist, inlet_point_id=args.inlet_point_id)


    # ---- Plot pressure drop ----
    plot_params = {"Q_min": args.flowrate_min, "Q_max": args.flowrate_max}
    plot_pressure_drop(output_folder, args.case_name, Q_in, dP, P_inlet, P_outlet, plot_params)

    # ---- Animate P vs. distance from inlet over time ----
    print(f"\nGenerating centerline pressure animation (frame stride = {args.frame_stride}) ...")
    animate_centerline_pressure(output_folder, args.case_name, pressures_all, Q_in, cl_dist, frame_stride=args.frame_stride)

    # ---- 3D surface: distance from inlet x time x pressure ----
    print(f"\nGenerating 3D surface plot (time subsampled every {args.frame_stride} snapshots) ...")
    plot_centerline_pressure_3d(output_folder, args.case_name, pressures_all, Q_in, cl_dist,frame_stride=args.frame_stride)




if __name__ == '__main__':
    main()
