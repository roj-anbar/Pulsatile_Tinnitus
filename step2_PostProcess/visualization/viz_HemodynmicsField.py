# -----------------------------------------------------------------------------------------------------------------------
# viz_HemodynamicsField.py
# Visualizes velocity streamlines, velocity magnitude isosurface, and Q-criterion
# isosurface from a single CFD snapshot resolved by target simulation time.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2026-05
#
# PURPOSE:
#   - Locates the HDF5 snapshot closest to a user-supplied target time [s] given
#     save_freq and timesteps-per-cycle (parsed from filename if not provided).
#   - Reads only that one snapshot (velocity field u); no other files are loaded.
#   - Renders velocity streamlines seeded from a sphere source (center + radius).
#   - Renders a 3D velocity magnitude isosurface at a user-specified isovalue (single color).
#   - Renders Q-criterion isosurface (via PyVista's built-in compute_derivative filter)
#     in a single color.
#   - Overlays geometry silhouette outline on every figure for spatial reference.
#   - Camera parameters (position, focal point, view-up, parallel scale) apply to all figures.
#
# REQUIREMENTS:
#   - h5py, numpy, pyvista, pyyaml
#   - On Trillium: virtual environment called "pyvista36"
#
# EXECUTION:
#   - Run directly on a login/debug node:
#       > module load StdEnv/2023 gcc/12.3 python/3.12.4
#       > source $HOME/virtual_envs/pyvista36/bin/activate
#
# EXAMPLE CLI:
#       > python viz_HemodynmicsField.py                     \
#           --config_file        <path_to_viz_config.yaml>        \
#           --mesh_folder   <path_to_mesh_folder>            \
#           --input_folder  <path_to_CFD_results_folder>     \
#           --output_folder <path_to_output_folder>          \
#           --case_name     PTSeg028_base_0p64               \
#           --target_time   2.53                             \
#           --save_freq     5                                \
#           --velocity_isovalue 0.5                          \
#           --qcri_isovalue     8000
#
# INPUTS (CLI — change per run):
#   --config_file             Path to YAML config file with case-specific defaults  [optional]
#   --mesh_folder        Folder containing BSLSolver HDF5 mesh file            [REQUIRED]
#   --input_folder       Folder with CFD HDF5 snapshots (*_curcyc_*up.h5)     [REQUIRED]
#   --output_folder      Output directory for PNG files                        [REQUIRED]
#   --case_name          Case identifier used in filenames and figure titles   [REQUIRED]
#   --target_time        Desired simulation time [s] to visualize              [REQUIRED]
#   --save_freq          Snapshot save frequency: every Nth timestep           (default: 5)
#   --snapshot_label     Optional output label override                        (default: t<actual_time>s)
#   --velocity_isovalue  Velocity magnitude isosurface value [m/s]            (default: 0.5)
#   --qcri_isovalue      Q-criterion isosurface value(s) [1/s2]               (default: 1000)
#
# INPUTS (config file — case-specific, rarely change):
#   period_seconds       Flow period [s]
#   timesteps_per_cyc    Timesteps per cycle (parsed from filename if omitted)
#   stream_seed_coord    Seed point [x y z] for streamline sphere source
#   cam_position         [x y z]
#   cam_focal_point      [x y z]
#   cam_view_up          [x y z]
#   cam_parallel_scale   Enables parallel (orthographic) projection at this scale
#   window_size          Render window [W H]
#
# OUTPUTS:
#   - <case_name>_<snapshot_label>_streamlines.png
#   - <case_name>_<snapshot_label>_velocityIso<value>.png
#   - <case_name>_<snapshot_label>_QcriIso<value>.png
#
# NOTES:
#   - Velocity in HDF5 is (N_nodes, 3) [m/s].  Mesh coordinates are in [mm].
#   - Coordinates are converted mm -> m before Q-criterion computation so that
#     Q has physical units of [1/s2].
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------

import re
import sys
import argparse
import numpy as np
import h5py
import pyvista as pv
pv.start_xvfb()        # start virtual X11 framebuffer for headless rendering on compute nodes
from pathlib import Path


# ======================================================================================================
# SNAPSHOT LOOKUP
# ======================================================================================================

def _extract_timestep(h5_file: Path) -> int:
    """Extract integer timestep from filename pattern '*_ts=<int>_*'."""
    match = re.search(r'_ts=(\d+)', h5_file.stem)
    if match is None:
        raise ValueError(f"'{h5_file.name}' has no '_ts=<int>' pattern.")
    return int(match.group(1))


def _extract_timesteps_per_cyc(h5_file: Path) -> int:
    """Parse timesteps-per-cycle from filename pattern '_ts<int>'."""
    match = re.search(r'_ts(\d+)', h5_file.stem)
    if match is None:
        raise ValueError(
            f"'{h5_file.name}' has no '_ts<int>' pattern. "
            "Supply --timesteps_per_cyc on the CLI instead."
        )
    return int(match.group(1))


def find_snapshot_at_time(input_folder: Path,
                          target_time: float,
                          save_freq: int,
                          period_seconds: float,
                          timesteps_per_cyc):
    """
    Locate the CFD snapshot closest to target_time [s].

    Snapshots are sorted by the '_ts=<int>' counter in their filenames.
    dt per saved snapshot = save_freq * period_seconds / timesteps_per_cyc.

    Returns (snapshot_file, frame_index, actual_time, timesteps_per_cyc).
    """
    h5_files = sorted(input_folder.glob('*_curcyc_*up.h5'), key=_extract_timestep)
    if not h5_files:
        raise FileNotFoundError(f"No CFD snapshots (*_curcyc_*up.h5) found in {input_folder}")

    if timesteps_per_cyc is None:
        timesteps_per_cyc = _extract_timesteps_per_cyc(h5_files[0])

    dt         = save_freq * period_seconds / timesteps_per_cyc
    time_array = np.arange(len(h5_files)) * dt

    frame_idx   = int(np.argmin(np.abs(time_array - target_time)))
    actual_time = float(time_array[frame_idx])

    return h5_files[frame_idx], frame_idx, actual_time, timesteps_per_cyc


# ======================================================================================================
# MESH & DATA I/O
# ======================================================================================================

_VTK_TETRA = 10
_VTK_HEXA  = 12


def load_mesh_from_h5(mesh_file: Path) -> pv.UnstructuredGrid:
    """Load volume mesh from BSLSolver/FEniCS HDF5 mesh file.

    Expects:
      Mesh/coordinates : (N, 3) float -- node coordinates [mm]
      Mesh/topology    : (M, k) int   -- cell connectivity (0-indexed)
    """
    with h5py.File(mesh_file, 'r') as h5:
        coords = np.array(h5['Mesh/coordinates'], dtype=np.float64)
        topo   = np.array(h5['Mesh/topology'],   dtype=np.int64)

    n_cells, nodes_per_cell = topo.shape
    if nodes_per_cell == 4:
        vtk_type = _VTK_TETRA
    elif nodes_per_cell == 8:
        vtk_type = _VTK_HEXA
    else:
        raise ValueError(f"Unsupported element: {nodes_per_cell} nodes/cell. "
                         "Expected 4 (tet) or 8 (hex).")

    cells      = np.column_stack([np.full(n_cells, nodes_per_cell, dtype=np.int64), topo]).ravel()
    cell_types = np.full(n_cells, vtk_type, dtype=np.uint8)

    return pv.UnstructuredGrid(cells, cell_types, coords)


def load_velocity_snapshot(snapshot_h5: Path) -> np.ndarray:
    """Read velocity field from one HDF5 snapshot.  Returns (N, 3) float [m/s]."""
    with h5py.File(snapshot_h5, 'r') as h5:
        u = np.array(h5['Solution']['u'], dtype=np.float64)
    if u.ndim == 1:
        u = u.reshape(-1, 3)
    return u


def attach_fields(grid: pv.UnstructuredGrid, u: np.ndarray) -> pv.UnstructuredGrid:
    """Attach velocity vector and magnitude as point arrays to the mesh."""
    grid['velocity']           = u
    grid['velocity_magnitude'] = np.linalg.norm(u, axis=1)
    return grid


# ======================================================================================================
# Q-CRITERION
# ======================================================================================================

def compute_qcriterion(grid: pv.UnstructuredGrid) -> np.ndarray:
    """
    Compute Q-criterion [1/s2] using PyVista's built-in compute_derivative filter.

    Converts coordinates mm -> m before differentiation so that the velocity
    gradient du_i/dx_j has units [1/s] and Q is in [1/s2].
    """
    grid_m = grid.copy()
    grid_m.points = grid.points * 1e-3    # mm -> m

    result = grid_m.compute_derivative(scalars='velocity', qcriterion=True)
    return result['qcriterion']


# ======================================================================================================
# CAMERA & RENDERING HELPERS
# ======================================================================================================

def make_plotter(window_size: list) -> pv.Plotter:
    """Create an off-screen PyVista plotter with a white background."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background('white')
    pl.enable_anti_aliasing('ssaa')
    return pl


def apply_camera(plotter: pv.Plotter, cam_params: dict):
    """Set camera position, focal point, view-up, and optional parallel scale.

    If none of the params are provided, falls back to PyVista's reset_camera()
    which auto-fits the view to the full mesh bounds.
    """
    any_set = False
    plotter.camera.parallel_projection = True
    if cam_params.get('position') is not None:
        plotter.camera.position = tuple(cam_params['position'])
        any_set = True
    if cam_params.get('focal_point') is not None:
        plotter.camera.focal_point = tuple(cam_params['focal_point'])
        any_set = True
    if cam_params.get('view_up') is not None:
        plotter.camera.up = tuple(cam_params['view_up'])
        any_set = True
    if cam_params.get('parallel_scale') is not None:
        plotter.camera.parallel_scale = cam_params['parallel_scale']
        any_set = True

    if any_set:
        plotter.camera_set = True   # prevent auto reset_camera() from overriding user params
    else:
        plotter.reset_camera()      # auto-fit to mesh bounds when no camera params given


def add_geometry_outline(plotter: pv.Plotter, grid: pv.UnstructuredGrid,
                         color: str = 'black', line_width: float = 1.5, opacity: float = 0.9):
    """Overlay the silhouette outline of the geometry (surface is invisible; only the outline is drawn)."""
    surface = grid.extract_surface()
    plotter.add_mesh(surface, color='white', opacity=0.1, show_scalar_bar=False,
                     silhouette=dict(color='silver', line_width=1, opacity=0.5))


def _scalar_bar_args(title: str) -> dict:
    return {'title': title, 'n_labels': 5, 'color': 'black', 'bold': True}


def save_screenshot(plotter: pv.Plotter, output_path: Path, scale: int = 2):
    plotter.screenshot(str(output_path), scale=scale)   # renders at scale× window_size
    plotter.close()


# ======================================================================================================
# FIGURE 1: VELOCITY STREAMLINES
# ======================================================================================================

def render_streamlines(grid: pv.UnstructuredGrid,
                       seed_coord: list,
                       cam_params: dict,
                       output_path: Path,
                       title: str,
                       window_size: list,
                       plot_seed: bool = True):
    """Render velocity streamlines from a sphere seed source as lines."""
    print("  Computing streamlines ...")
    streamlines = grid.streamlines(
        vectors='velocity',
        source_center=seed_coord,
        source_radius=10,
        n_points=1000,
        initial_step_length=0.05,
        interpolator_type='cell',   # more robust than 'point' for complex unstructured meshes
    )

    pl = make_plotter(window_size)
    apply_camera(pl, cam_params)
    add_geometry_outline(pl, grid)

    if streamlines.n_points > 0:
        pl.add_mesh(streamlines,
                    scalars='velocity_magnitude',
                    cmap='viridis',
                    line_width=6,
                    show_scalar_bar=True,
                    scalar_bar_args=_scalar_bar_args('Velocity (m/s)'))
    else:
        print("  Warning: no streamlines generated. Check stream_seed_coord and domain bounds.")

    if plot_seed:
        seed_sphere = pv.Sphere(center=seed_coord, radius=10)
        pl.add_mesh(seed_sphere, color='yellow', opacity=0.5, show_scalar_bar=False)

    pl.add_title(title, color='black', font_size=20)
    save_screenshot(pl, output_path)


# ======================================================================================================
# FIGURE 2: VELOCITY MAGNITUDE ISOSURFACE
# ======================================================================================================

def render_velocity_isosurface(grid: pv.UnstructuredGrid,
                               isovalue: float,
                               cam_params: dict,
                               output_path: Path,
                               title: str,
                               window_size: list):
    """Render the |u| = isovalue isosurface in a single color."""
    print(f"  Extracting velocity isosurface at |u| = {isovalue} m/s ...")
    iso = grid.contour(isosurfaces=[isovalue], scalars='velocity_magnitude')

    pl = make_plotter(window_size)
    apply_camera(pl, cam_params)
    add_geometry_outline(pl, grid)

    if iso.n_points > 0:
        pl.add_mesh(iso, color='red', show_scalar_bar=False)
    else:
        print(f"  Warning: no isosurface at |u| = {isovalue} m/s. "
              "Adjust --velocity_isovalue (velocity range printed above).")

    pl.add_title(title, color='black', font_size=20)
    save_screenshot(pl, output_path)


# ======================================================================================================
# FIGURE 3: Q-CRITERION ISOSURFACE
# ======================================================================================================

def render_qcriterion(grid: pv.UnstructuredGrid,
                      qcri_isovalue,
                      cam_params: dict,
                      output_path: Path,
                      title: str,
                      window_size: list):
    """Compute and render Q-criterion isosurface(s) in a single color."""
    print("  Computing Q-criterion ...")
    Q = compute_qcriterion(grid)
    grid['Qcriterion'] = Q
    print(f"  Q range: {Q.min():.3g}  to  {Q.max():.3g}  [1/s2]")

    isovalues = [qcri_isovalue] if isinstance(qcri_isovalue, (int, float)) else list(qcri_isovalue)
    print(f"  Extracting isosurface at Q = {isovalues} [1/s2] ...")
    iso = grid.contour(isosurfaces=isovalues, scalars='Qcriterion')

    pl = make_plotter(window_size)
    apply_camera(pl, cam_params)
    add_geometry_outline(pl, grid)

    if iso.n_points > 0:
        pl.add_mesh(iso, color='cyan', show_scalar_bar=False)
    else:
        print(f"  Warning: no isosurface found at Q = {isovalues}. "
              "Adjust --qcri_isovalue (Q range printed above).")

    pl.add_title(title, color='black', font_size=20)
    save_screenshot(pl, output_path)


# ======================================================================================================
# ARGUMENT PARSING
# ======================================================================================================

def parse_args():
    import yaml

    # Pre-parse to get --config before building the full parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config_file', default=None)
    pre_args, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(
        description="Visualize streamlines, velocity contour, and Q-criterion from one CFD snapshot.")

    # ---- Config file ----
    ap.add_argument('--config_file', default=None,
                    help="Path to YAML config file with case-specific defaults (CLI args override)")

    # ---- Mesh & data (required) ----
    ap.add_argument('--mesh_folder',    required=True, help="Folder with BSLSolver HDF5 mesh (Mesh/coordinates + Mesh/topology)")
    ap.add_argument('--input_folder',   required=True, help="Folder with CFD HDF5 snapshots (*_curcyc_*up.h5)")
    ap.add_argument('--output_folder',  required=True, help="Output directory for PNG files")
    ap.add_argument('--case_name',      required=True, help="Case identifier used in filenames and figure titles")
    ap.add_argument('--snapshot_label', default=None,  help="Output label override (default: t<actual_time>s)")

    # ---- Snapshot selection (CLI) ----
    ap.add_argument('--target_time', type=float, required=True, help="Desired simulation time [s] to visualize")
    ap.add_argument('--save_freq',   type=int,   default=5,     help="Snapshot save frequency: every Nth timestep (default: 5)")

    # ---- CFD cycle parameters (config) ----
    ap.add_argument('--period_seconds',    type=float, default=None, help="Flow period [s]")
    ap.add_argument('--timesteps_per_cyc', type=int,   default=None, help="Timesteps per cycle (parsed from filename if omitted)")

    # ---- Isosurface values (CLI) ----
    ap.add_argument('--velocity_isovalue', type=float, default=0.5,    help="Velocity magnitude isosurface value [m/s] (default: 0.5)")
    ap.add_argument('--qcri_isovalue',     type=float, default=1000.0, help="Q-criterion isosurface value(s) [1/s2] (default: 1000)")

    # ---- Streamline seed (config) ----
    ap.add_argument('--stream_seed_coord', type=float, nargs=3, default=None, help="Seed point [x y z] for streamline sphere source")

    # ---- Camera (config) ----
    ap.add_argument('--cam_position',       type=float, nargs=3, default=None, help="Camera position [x y z]")
    ap.add_argument('--cam_focal_point',    type=float, nargs=3, default=None, help="Camera focal point [x y z]")
    ap.add_argument('--cam_view_up',        type=float, nargs=3, default=None, help="Camera view-up vector [x y z]")
    ap.add_argument('--cam_parallel_scale', type=float,          default=None, help="Enable parallel projection at this scale")

    # ---- Output (config) ----
    ap.add_argument('--window_size', type=int, nargs=2, default=None, help="Render window size [W H]")

    # Load config as defaults so CLI args always win
    if pre_args.config_file:
        with open(pre_args.config_file) as f:
            cfg = yaml.safe_load(f)
        ap.set_defaults(**cfg)

    return ap.parse_args()


# ======================================================================================================
# MAIN
# ======================================================================================================

def main():
    args = parse_args()

    input_folder  = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    window_size   = args.window_size if args.window_size is not None else [1920, 1080]
    period        = args.period_seconds if args.period_seconds is not None else 0.915

    print("=" * 100)
    print("viz_HemodynamicsField.py")
    print(f"  Config        : {args.config_file}")
    print(f"  Input folder  : {input_folder}")
    print(f"  Output folder : {output_folder}")
    print(f"  Case name     : {args.case_name}")
    print(f"  Target time   : {args.target_time} s")
    print("=" * 100 + "\n")

    # ---- Resolve snapshot from target time ----
    print("Finding snapshot file ...")
    snapshot_file, frame_idx, actual_time, timesteps_per_cyc = find_snapshot_at_time(
        input_folder, args.target_time,
        save_freq         = args.save_freq,
        period_seconds    = period,
        timesteps_per_cyc = args.timesteps_per_cyc,
    )
    dt = args.save_freq * period / timesteps_per_cyc
    print(f"  Target time    : {args.target_time:.6f} s")
    print(f"  Resolved frame : {frame_idx}  (0-based index in sorted snapshot list)")
    print(f"  Actual time    : {actual_time:.6f} s  (dt = {dt:.6f} s/frame)")
    print(f"  Time error     : {abs(actual_time - args.target_time) * 1e3:.3f} ms")
    print(f"  Snapshot file  : {snapshot_file.name}\n")

    snapshot_label = args.snapshot_label or f"t{actual_time:.4f}s"

    # ---- Load mesh ----
    mesh_file = next(Path(args.mesh_folder).glob('*.h5'), None)
    if mesh_file is None:
        print(f"ERROR: no .h5 mesh file found in {args.mesh_folder}")
        sys.exit(1)
    print(f"Loading mesh ...")
    grid = load_mesh_from_h5(mesh_file)
    print(f"  {grid.n_points} nodes, {grid.n_cells} cells\n")

    # ---- Load velocity snapshot ----
    print("Loading velocity ...")
    u = load_velocity_snapshot(snapshot_file)
    vel_mag = np.linalg.norm(u, axis=1)
    print(f"  Velocity magnitude:  min={vel_mag.min():.4f}  max={vel_mag.max():.4f} [m/s]\n")
    grid = attach_fields(grid, u)

    # ---- Shared camera dict ----
    cam_params = {
        'position':       args.cam_position,
        'focal_point':    args.cam_focal_point,
        'view_up':        args.cam_view_up,
        'parallel_scale': args.cam_parallel_scale,
    }

    # ---- Figure 1: Streamlines ----
    print("Rendering streamlines ...")
    out_stream = output_folder / f"{args.case_name}_{snapshot_label}_streamlines.png"
    render_streamlines(
        grid,
        seed_coord  = args.stream_seed_coord,
        cam_params  = cam_params,
        output_path = out_stream,
        title       = f"{args.case_name}  --  Velocity Streamlines  |  t = {actual_time:.4f} s",
        window_size = window_size,
    )

    # ---- Figure 2: Velocity magnitude isosurface ----
    print("\nRendering velocity magnitude isosurface ...")
    out_vel = output_folder / f"{args.case_name}_{snapshot_label}_velocityIso{args.velocity_isovalue}.png"
    render_velocity_isosurface(
        grid,
        isovalue    = args.velocity_isovalue,
        cam_params  = cam_params,
        output_path = out_vel,
        title       = f"{args.case_name}  --  |u| = {args.velocity_isovalue} m/s  |  t = {actual_time:.4f} s",
        window_size = window_size,
    )

    # ---- Figure 3: Q-criterion ----
    print("\nRendering Q-criterion isosurface ...")
    out_qcrit = output_folder / f"{args.case_name}_{snapshot_label}_QcriIso{args.qcri_isovalue}.png"
    render_qcriterion(
        grid,
        qcri_isovalue = args.qcri_isovalue,
        cam_params    = cam_params,
        output_path   = out_qcrit,
        title         = f"{args.case_name}  --  Q-criterion  |  t = {actual_time:.4f} s",
        window_size   = window_size,
    )

    print("\nDone.\n")


if __name__ == '__main__':
    main()
