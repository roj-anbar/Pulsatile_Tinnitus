# -----------------------------------------------------------------------------------------------------------------------
# viz_HemodynamicsField.py
# Visualizes velocity streamlines, velocity magnitude contour, and Q-criterion
# isosurface from a single CFD snapshot resolved by target simulation time.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2026-05
#
# PURPOSE:
#   - Locates the HDF5 snapshot closest to a user-supplied target time [s] given
#     save_freq and timesteps-per-cycle (parsed from filename if not provided).
#   - Reads only that one snapshot (velocity field u); no other files are loaded.
#   - Renders velocity streamlines with configurable colormap, seed node IDs, and line width.
#   - Renders velocity magnitude contour on a user-specified slice plane.
#   - Renders Q-criterion isosurface (via PyVista's built-in compute_derivative filter)
#     colored by velocity magnitude or Q itself.
#   - Overlays geometry surface edges on every figure for spatial reference.
#   - Camera parameters (position, focal point, view-up, parallel scale) apply to all figures.
#
# REQUIREMENTS:
#   - h5py, numpy, pyvista
#   - On Trillium: virtual environment called "pyvista36"
#
# EXECUTION:
#   - Run directly on a login/debug node:
#       > module load StdEnv/2023 gcc/12.3 python/3.12.4
#       > source $HOME/virtual_envs/pyvista36/bin/activate
#
# EXAMPLE CLI:
#       > python viz_HemodynamicsField.py                    \
#           --mesh_folder       <path_to_mesh_folder>        \
#           --input_folder      <path_to_CFD_results_folder> \
#           --output_folder     <path_to_output_folder>      \
#           --case_name         PTSeg028_base_0p64           \
#           --target_time       2.79                         \
#           --save_freq         5                            \
#           --stream_seed_ids   100 500 9853                 \
#           --stream_colormap   jet                          \
#           --stream_line_width 2.0                          \
#           --vel_colormap      turbo                        \
#           --vel_clim          0.0 1.5                      \
#           --vel_slice_normal  0 1 0                        \
#           --qcrit_values      5000 20000                   \
#           --cam_position      0 0 500                      \
#           --cam_focal_point   0 0 0                        \
#           --cam_view_up       0 1 0                        \
#           --cam_parallel_scale 50
#
# INPUTS:
#   --mesh_folder        Folder containing BSLSolver HDF5 mesh file (Mesh/coordinates + Mesh/topology)
#   --mesh_vtu           Path to mesh VTU file (alternative to --mesh_folder)
#   --input_folder       Folder with CFD HDF5 snapshots (*_curcyc_*up.h5)   [REQUIRED]
#   --output_folder      Output directory for PNG files                       [REQUIRED]
#   --case_name          Case name (used in filenames and titles)             [REQUIRED]
#   --target_time        Desired simulation time [s] to visualize            [REQUIRED]
#   --save_freq          Snapshot save frequency: every Nth timestep (default: 1)
#   --period_seconds     Flow period [s] (default: 0.915)
#   --timesteps_per_cyc  Timesteps per cycle (parsed from filename if omitted)
#   --snapshot_label     Optional output label override (default: t<actual_time>s)
#
#   Streamlines:
#   --stream_colormap    Colormap name            (default: jet)
#   --stream_seed_ids    Volume mesh node IDs used as streamline seeds  (default: 0)
#   --stream_line_width  Line width when rendering as lines             (default: 2.0)
#   --stream_max_time    Maximum streamline integration time            (default: 1.0)
#   --stream_tube_radius Tube radius [mesh units]; 0 = render as lines  (default: 0)
#
#   Velocity contour (planar slice):
#   --vel_colormap       Colormap name            (default: jet)
#   --vel_clim           Color limits [vmin vmax]
#   --vel_slice_normal   Slice plane normal [nx ny nz]  (default: 0 0 1)
#   --vel_slice_origin   Slice plane origin [x y z]     (default: mesh centroid)
#
#   Q-criterion:
#   --qcrit_colormap     Colormap name            (default: plasma)
#   --qcrit_values       Isosurface value(s) [1/s2]  (default: 1000)
#   --qcrit_clim         Color limits [vmin vmax]
#   --qcrit_color_by     velocity_magnitude | Q_criterion  (default: velocity_magnitude)
#
#   Camera (shared across all three figures):
#   --cam_position       [x y z]
#   --cam_focal_point    [x y z]
#   --cam_view_up        [x y z]
#   --cam_parallel_scale Enables parallel (orthographic) projection at this scale
#
#   Output:
#   --window_size        Render window [W H]  (default: 1920 1080)
#
# OUTPUTS:
#   - <case_name>_streamlines_<snapshot_label>.png
#   - <case_name>_velocityContour_<snapshot_label>.png
#   - <case_name>_Qcriterion_<snapshot_label>.png
#
# NOTES:
#   - Velocity in HDF5 is (N_nodes, 3) [m/s].  Mesh coordinates are in [mm].
#   - Coordinates are converted mm -> m before Q-criterion computation so that
#     Q has physical units of [1/s2].  Q-criterion is computed via PyVista's
#     built-in compute_derivative(qcriterion=True) filter.
#   - Only the resolved snapshot is read; no other snapshots are loaded.
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------

import re
import sys
import argparse
import numpy as np
import h5py
import pyvista as pv
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

def compute_q_criterion(grid: pv.UnstructuredGrid) -> np.ndarray:
    """
    Compute Q-criterion [1/s2] using PyVista's built-in compute_derivative filter.

    Converts coordinates mm -> m before differentiation so that the velocity
    gradient du_i/dx_j has units [1/s] and Q is in [1/s2].
    """
    grid_m = grid.copy()
    grid_m.points = grid.points * 1e-3    # mm -> m

    result = grid_m.compute_derivative(scalars='velocity', qcriterion=True)
    return result['Q-criterion']


# ======================================================================================================
# CAMERA & RENDERING HELPERS
# ======================================================================================================

def make_plotter(window_size: list) -> pv.Plotter:
    """Create an off-screen PyVista plotter with a white background."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background('white')
    return pl


def apply_camera(plotter: pv.Plotter, cam_params: dict):
    """Set camera position, focal point, view-up, and optional parallel scale."""
    if cam_params.get('position') is not None:
        plotter.camera.position = tuple(cam_params['position'])
    if cam_params.get('focal_point') is not None:
        plotter.camera.focal_point = tuple(cam_params['focal_point'])
    if cam_params.get('view_up') is not None:
        plotter.camera.up = tuple(cam_params['view_up'])
    if cam_params.get('parallel_scale') is not None:
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = cam_params['parallel_scale']
    plotter.camera_set = True   # prevent auto reset_camera()


def add_surface_edges(plotter: pv.Plotter, grid: pv.UnstructuredGrid,
                      color: str = 'black', line_width: float = 0.5, opacity: float = 0.3):
    """Extract and overlay the outer surface wireframe of the volume mesh."""
    surface = grid.extract_surface()
    plotter.add_mesh(surface, style='wireframe', color=color, line_width=line_width, opacity=opacity)


def _scalar_bar_args(title: str) -> dict:
    return {'title': title, 'n_labels': 5, 'color': 'black', 'bold': True}


def save_screenshot(plotter: pv.Plotter, output_path: Path):
    plotter.screenshot(str(output_path))
    plotter.close()
    print(f"  Saved  ->  {output_path}")


# ======================================================================================================
# FIGURE 1: VELOCITY STREAMLINES
# ======================================================================================================

def render_streamlines(grid: pv.UnstructuredGrid,
                       seed_ids: list,
                       colormap: str,
                       line_width: float,
                       max_time: float,
                       tube_radius: float,
                       cam_params: dict,
                       output_path: Path,
                       title: str,
                       window_size: list):
    """
    Render velocity streamlines seeded at specified volume mesh node IDs.

    Seeds are the node coordinates at the given indices.
    Streamlines are rendered as tubes if tube_radius > 0, otherwise as lines.
    """
    seed_points = grid.points[np.asarray(seed_ids, dtype=int)]
    source      = pv.PolyData(seed_points)

    print("  Computing streamlines ...")
    streamlines = grid.streamlines_from_source(
        source,
        vectors='velocity',
        max_time=max_time,
        integration_direction='both',
        initial_step_length=0.1,
        step_unit='cl',
        interpolator_type='p',
    )

    pl = make_plotter(window_size)
    add_surface_edges(pl, grid)

    if streamlines.n_points > 0:
        render_mesh = streamlines.tube(radius=tube_radius) if tube_radius else streamlines
        pl.add_mesh(render_mesh,
                    scalars='velocity_magnitude',
                    cmap=colormap,
                    line_width=line_width,
                    show_scalar_bar=True,
                    scalar_bar_args=_scalar_bar_args('Velocity (m/s)'))
    else:
        print("  Warning: no streamlines generated. Check --stream_seed_ids and domain bounds.")

    pl.add_title(title, color='black', font_size=10)
    apply_camera(pl, cam_params)
    save_screenshot(pl, output_path)


# ======================================================================================================
# FIGURE 2: VELOCITY MAGNITUDE CONTOUR (PLANAR SLICE)
# ======================================================================================================

def render_velocity_contour(grid: pv.UnstructuredGrid,
                             slice_normal: list,
                             slice_origin: list,
                             colormap: str,
                             clim: list,
                             cam_params: dict,
                             output_path: Path,
                             title: str,
                             window_size: list):
    """Render velocity magnitude on a planar slice through the domain."""
    origin = slice_origin if slice_origin is not None else grid.center

    print("  Computing velocity slice ...")
    slc = grid.slice(normal=slice_normal, origin=origin)

    pl = make_plotter(window_size)
    add_surface_edges(pl, grid)

    mesh_kwargs = dict(
        scalars='velocity_magnitude',
        cmap=colormap,
        show_scalar_bar=True,
        scalar_bar_args=_scalar_bar_args('Velocity (m/s)'),
    )
    if clim is not None:
        mesh_kwargs['clim'] = clim

    pl.add_mesh(slc, **mesh_kwargs)
    pl.add_title(title, color='black', font_size=10)
    apply_camera(pl, cam_params)
    save_screenshot(pl, output_path)


# ======================================================================================================
# FIGURE 3: Q-CRITERION ISOSURFACE
# ======================================================================================================

def render_qcriterion(grid: pv.UnstructuredGrid,
                       qcrit_values: list,
                       colormap: str,
                       clim: list,
                       color_by: str,
                       cam_params: dict,
                       output_path: Path,
                       title: str,
                       window_size: list):
    """Render Q-criterion isosurface(s) colored by velocity magnitude or Q itself."""
    print("  Extracting Q-criterion isosurface ...")
    iso = grid.contour(isosurfaces=qcrit_values, scalars='Q_criterion')

    pl = make_plotter(window_size)
    add_surface_edges(pl, grid)

    if iso.n_points > 0:
        cbar_title = 'Velocity (m/s)' if color_by == 'velocity_magnitude' else 'Q-criterion (1/s2)'
        mesh_kwargs = dict(
            scalars=color_by,
            cmap=colormap,
            show_scalar_bar=True,
            scalar_bar_args=_scalar_bar_args(cbar_title),
        )
        if clim is not None:
            mesh_kwargs['clim'] = clim
        pl.add_mesh(iso, **mesh_kwargs)
    else:
        print(f"  Warning: no isosurface found at Q = {qcrit_values}.  "
              "Adjust --qcrit_values (Q range printed above).")

    pl.add_title(title, color='black', font_size=10)
    apply_camera(pl, cam_params)
    save_screenshot(pl, output_path)


# ======================================================================================================
# ARGUMENT PARSING
# ======================================================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Visualize streamlines, velocity contour, and Q-criterion from one CFD snapshot.")

    # ---- Mesh & data ----
    mesh_grp = ap.add_mutually_exclusive_group(required=True)
    mesh_grp.add_argument('--mesh_folder', help="Folder with BSLSolver HDF5 mesh (Mesh/coordinates + Mesh/topology)")
    mesh_grp.add_argument('--mesh_vtu',    help="Path to mesh VTU file (alternative to --mesh_folder)")
    ap.add_argument('--input_folder',      required=True,             help="Folder with CFD HDF5 snapshots (*_curcyc_*up.h5)")
    ap.add_argument('--output_folder',     required=True,             help="Output directory for PNG files")
    ap.add_argument('--case_name',         required=True,             help="Case identifier used in filenames and figure titles")
    ap.add_argument('--target_time',       type=float, required=True, help="Desired simulation time [s] to visualize")
    
    ap.add_argument('--save_freq',         type=int,   default=5,     help="Snapshot save frequency: every Nth timestep (default: 1)")
    ap.add_argument('--period_seconds',    type=float, default=0.915, help="Flow period [s] (default: 0.915)")
    ap.add_argument('--timesteps_per_cyc', type=int,   default=None,  help="Timesteps per cycle (parsed from filename if omitted)")
    ap.add_argument('--snapshot_label',    default=None,
                    help="Output label override (default: t<actual_time>s)")

    # ---- Streamlines ----
    ap.add_argument('--stream_colormap',    default='jet',
                    help="Colormap for streamlines (default: jet)")
    ap.add_argument('--stream_seed_ids',    type=int, nargs='+', default=[0],
                    help="Volume mesh node IDs used as streamline seeds (default: 0)")
    ap.add_argument('--stream_line_width',  type=float, default=2.0,
                    help="Line width when rendering streamlines as lines (default: 2.0)")
    ap.add_argument('--stream_max_time',    type=float, default=1.0,
                    help="Maximum streamline integration time (default: 1.0)")
    ap.add_argument('--stream_tube_radius', type=float, default=0.0,
                    help="Tube radius [mesh units]; 0 = render as lines (default: 0)")

    # ---- Velocity contour ----
    ap.add_argument('--vel_colormap',     default='jet',
                    help="Colormap for velocity magnitude contour (default: jet)")
    ap.add_argument('--vel_clim',         type=float, nargs=2, default=None,
                    help="Color limits [vmin vmax] for velocity contour")
    ap.add_argument('--vel_slice_normal', type=float, nargs=3, default=[0.0, 0.0, 1.0],
                    help="Slice plane normal vector (default: 0 0 1)")
    ap.add_argument('--vel_slice_origin', type=float, nargs=3, default=None,
                    help="Slice plane origin [x y z] (default: mesh centroid)")

    # ---- Q-criterion ----
    ap.add_argument('--qcrit_colormap',  default='plasma',
                    help="Colormap for Q-criterion isosurface (default: plasma)")
    ap.add_argument('--qcrit_values',    type=float, nargs='+', default=[1000.0],
                    help="Q-criterion isosurface value(s) [1/s2] (default: 1000)")
    ap.add_argument('--qcrit_clim',      type=float, nargs=2, default=None,
                    help="Color limits [vmin vmax] for Q-criterion figure")
    ap.add_argument('--qcrit_color_by',  default='velocity_magnitude',
                    choices=['velocity_magnitude', 'Q_criterion'],
                    help="Field to color Q-criterion isosurface (default: velocity_magnitude)")

    # ---- Camera (shared across all figures) ----
    ap.add_argument('--cam_position',       type=float, nargs=3, default=None,
                    help="Camera position [x y z]")
    ap.add_argument('--cam_focal_point',    type=float, nargs=3, default=None,
                    help="Camera focal point [x y z]")
    ap.add_argument('--cam_view_up',        type=float, nargs=3, default=None,
                    help="Camera view-up vector [x y z]")
    ap.add_argument('--cam_parallel_scale', type=float, default=None,
                    help="Enable parallel projection at this scale")


    # ---- Output ----
    ap.add_argument('--window_size', type=int, nargs=2, default=[1920, 1080],
                    help="Render window size [W H] (default: 1920 1080)")

    return ap.parse_args()


# ======================================================================================================
# MAIN
# ======================================================================================================

def main():
    args = parse_args()

    input_folder  = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("viz_HemodynamicsField.py")
    print(f"  Input folder  : {input_folder}")
    print(f"  Output folder : {output_folder}")
    print(f"  Case name     : {args.case_name}")
    print(f"  Target time   : {args.target_time} s")
    print("=" * 100 + "\n")

    # ---- Resolve snapshot from target time ----
    print("Locating snapshot ...")
    snapshot_file, frame_idx, actual_time, timesteps_per_cyc = find_snapshot_at_time(
        input_folder, args.target_time,
        save_freq         = args.save_freq,
        period_seconds    = args.period_seconds,
        timesteps_per_cyc = args.timesteps_per_cyc,
    )
    dt = args.save_freq * args.period_seconds / timesteps_per_cyc
    print(f"  Target time       : {args.target_time:.6f} s")
    print(f"  Resolved frame    : {frame_idx}  (0-based index in sorted snapshot list)")
    print(f"  Actual time       : {actual_time:.6f} s  (dt = {dt:.6f} s/frame)")
    print(f"  Time error        : {abs(actual_time - args.target_time) * 1e3:.3f} ms")
    print(f"  Snapshot file     : {snapshot_file.name}\n")

    snapshot_label = args.snapshot_label or f"t{actual_time:.4f}s"

    # ---- Load mesh ----
    if args.mesh_folder:
        mesh_file = next(Path(args.mesh_folder).glob('*.h5'), None)
        if mesh_file is None:
            print(f"ERROR: no .h5 mesh file found in {args.mesh_folder}")
            sys.exit(1)
        print(f"Loading mesh from HDF5: {mesh_file.name} ...")
        grid = load_mesh_from_h5(mesh_file)
    else:
        print(f"Loading mesh from VTU: {args.mesh_vtu} ...")
        grid = pv.read(args.mesh_vtu)
    print(f"  {grid.n_points} nodes, {grid.n_cells} cells\n")

    # ---- Load velocity snapshot ----
    print(f"Loading velocity: {snapshot_file.name} ...")
    u = load_velocity_snapshot(snapshot_file)
    vel_mag = np.linalg.norm(u, axis=1)
    print(f"  shape: {u.shape}")
    print(f"  |u|  min={vel_mag.min():.4f}  max={vel_mag.max():.4f}  mean={vel_mag.mean():.4f}  [m/s]\n")
    grid = attach_fields(grid, u)

    # ---- Compute Q-criterion ----
    print("Computing Q-criterion ...")
    Q = compute_q_criterion(grid)
    grid['Q_criterion'] = Q
    print(f"  Q range: {Q.min():.3g}  to  {Q.max():.3g}  [1/s2]")
    print(f"  (positive Q = vortex-dominated; set --qcrit_values within this range)\n")

    # ---- Shared parameter dicts ----
    cam_params = {
        'position':       args.cam_position,
        'focal_point':    args.cam_focal_point,
        'view_up':        args.cam_view_up,
        'parallel_scale': args.cam_parallel_scale,
    }

    # ---- Figure 1: Streamlines ----
    print("Rendering streamlines ...")
    out_stream = output_folder / f"{args.case_name}_streamlines_{snapshot_label}.png"
    render_streamlines(
        grid,
        seed_ids    = args.stream_seed_ids,
        colormap    = args.stream_colormap,
        line_width  = args.stream_line_width,
        max_time    = args.stream_max_time,
        tube_radius = args.stream_tube_radius if args.stream_tube_radius > 0 else None,
        cam_params  = cam_params,
        output_path = out_stream,
        title       = f"{args.case_name}  --  Velocity Streamlines  |  t = {actual_time:.4f} s",
        window_size = args.window_size,
    )

    # ---- Figure 2: Velocity contour ----
    print("\nRendering velocity magnitude contour ...")
    out_vel = output_folder / f"{args.case_name}_velocityContour_{snapshot_label}.png"
    render_velocity_contour(
        grid,
        slice_normal = args.vel_slice_normal,
        slice_origin = args.vel_slice_origin,
        colormap     = args.vel_colormap,
        clim         = args.vel_clim,
        cam_params   = cam_params,
        output_path  = out_vel,
        title        = f"{args.case_name}  --  Velocity Magnitude  |  t = {actual_time:.4f} s",
        window_size  = args.window_size,
    )

    # ---- Figure 3: Q-criterion ----
    print("\nRendering Q-criterion isosurface ...")
    out_qcrit = output_folder / f"{args.case_name}_Qcriterion_{snapshot_label}.png"
    render_qcriterion(
        grid,
        qcrit_values = args.qcrit_values,
        colormap     = args.qcrit_colormap,
        clim         = args.qcrit_clim,
        color_by     = args.qcrit_color_by,
        cam_params   = cam_params,
        output_path  = out_qcrit,
        title        = f"{args.case_name}  --  Q-criterion  |  t = {actual_time:.4f} s",
        window_size  = args.window_size,
    )

    print("\nDone.\n")


if __name__ == '__main__':
    main()
