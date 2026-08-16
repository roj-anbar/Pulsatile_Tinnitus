# -----------------------------------------------------------------------------------------------------------------------
# viz_HemodynamicsVideo.py
# Creates a 3-panel MP4 animation: velocity isosurface | streamlines | Q-criterion
# evolving together across all CFD snapshots (or a user-specified range).
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2026-08
#
# PURPOSE:
#   - Discovers all HDF5 snapshots in temporal order and renders them as a video.
#   - Each video frame contains three stacked rows sharing the same camera:
#       Row 0 – Velocity magnitude isosurface (solid color)
#       Row 1 – Velocity streamlines (colored by magnitude, annotated with time)
#       Row 2 – Q-criterion isosurface (single color)
#   - Frames are rendered in parallel (one process per frame) using multiprocessing,
#     then stitched into MP4 with ffmpeg.
#
# REQUIREMENTS:
#   - h5py, numpy, pyvista, pyyaml
#   - ffmpeg in PATH (e.g. module load ffmpeg on Trillium)
#   - On Trillium: virtual environment called "pyvista36"
#
# EXECUTION:
#   - Run directly on a login/debug node:
#       > module load StdEnv/2023 gcc/12.3 python/3.12.4
#       > module load ffmpeg
#       > source $HOME/virtual_envs/pyvista36/bin/activate
#
# EXAMPLE CLI:
#       > python viz_HemodynamicsVideo.py                        \
#           --config_file        <path_to_viz_config.yaml>       \
#           --mesh_folder        <path_to_mesh_folder>           \
#           --input_folder       <path_to_CFD_results_folder>    \
#           --output_folder      <path_to_output_folder>         \
#           --case_name          PTSeg028_base_0p64              \
#           --save_freq          5                               \
#           --velocity_isovalue  0.5                             \
#           --qcri_isovalue      50000                           \
#           --vel_max            2.0                             \
#           --framerate          100                             \
#           --n_workers          192
#
# INPUTS (CLI — change per run):
#   --config_file             Path to YAML config file with case-specific defaults  [optional]
#   --mesh_folder             Folder containing BSLSolver HDF5 mesh file            [REQUIRED]
#   --input_folder            Folder with CFD HDF5 snapshots (*_curcyc_*up.h5)     [REQUIRED]
#   --output_folder           Output directory for the video                        [REQUIRED]
#   --case_name               Case identifier used in the output filename           [REQUIRED]
#   --velocity_isovalue       Velocity magnitude isosurface value [m/s]            (default: 0.5)
#   --qcri_isovalue           Q-criterion isosurface value [1/s2]                  (default: 1000)
#   --vel_max                 Streamline colormap upper bound [m/s]                 (default: 2.0)
#   --framerate               Output video frame rate [fps]                         (default: 60)
#   --start_frame             First snapshot index, 0-based inclusive               (default: 0)
#   --end_frame               Last snapshot index, 0-based exclusive                (default: all)
#   --frame_stride            Render every Nth snapshot                             (default: 1)
#   --n_workers               Number of parallel worker processes                   (default: all CPUs)
#   --keep_frames             Keep per-frame PNG files after stitching              (default: delete)
#
# INPUTS (config file -- case-specific, rarely change):
#   period_seconds       Flow period [s]
#   timesteps_per_cyc    Timesteps per cycle (parsed from filename if omitted)
#   save_freq            Snapshot save frequency
#   stream_seed_coord    Seed center [x y z] for streamline sphere source
#   stream_seed_radius   Seed sphere radius (mesh units, mm)
#   cam_position         [x y z]
#   cam_focal_point      [x y z]
#   cam_view_up          [x y z]
#   cam_parallel_scale   Enables parallel (orthographic) projection at this scale
#   window_size          Render window [W H] -- total across all 3 panels
#
# OUTPUTS:
#   - <output_folder>/<case_name>_hemodynamics_video.mp4
#   - <output_folder>/<case_name>_frames/frame_XXXXXX.png  (deleted unless --keep_frames)
#
# NOTES:
#   - Each worker process loads the mesh independently and renders one frame to PNG.
#   - Coordinates are converted mm -> m before Q-criterion so Q has units [1/s2].
#   - window_size is the total render window; each panel occupies one third of the height.
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------

import re
import sys
import shutil
import argparse
import subprocess
import multiprocessing as mp
import numpy as np
import imageio
import pyvista as pv
pv.start_xvfb()        # start virtual X11 framebuffer for headless rendering on compute nodes
from pathlib import Path

# Shared utilities from the static-snapshot companion script (same directory)
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from viz_HemodynmicsField import (
    load_mesh_from_h5,
    load_velocity_snapshot,
    attach_fields,
    compute_qcriterion,
    apply_camera,
    add_geometry_outline,
)


# ======================================================================================================
# SNAPSHOT DISCOVERY
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


def get_all_snapshots(input_folder: Path, timesteps_per_cyc=None):
    """Return (h5_files_sorted, timesteps_per_cyc) for all snapshots in the folder."""
    h5_files = sorted(input_folder.glob('*_curcyc_*up.h5'), key=_extract_timestep)
    if not h5_files:
        raise FileNotFoundError(f"No CFD snapshots (*_curcyc_*up.h5) found in {input_folder}")
    if timesteps_per_cyc is None:
        timesteps_per_cyc = _extract_timesteps_per_cyc(h5_files[0])
    return h5_files, timesteps_per_cyc


# ======================================================================================================
# MULTI-PANEL PLOTTER FACTORY
# ======================================================================================================

def make_panel_plotter(window_size: list) -> pv.Plotter:
    """Create a single off-screen plotter for one panel with white background."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background('white')
    pl.enable_anti_aliasing('ssaa')
    pl.enable_depth_peeling()
    return pl


# ======================================================================================================
# PANEL RENDERING FUNCTIONS
# Each function adds actors to the *currently active* subplot of the plotter.
# Call pl.subplot(row, 0) before invoking, then apply_camera(pl, cam_params) after.
# ======================================================================================================

def add_velocity_iso_panel(pl: pv.Plotter, grid: pv.UnstructuredGrid, isovalue: float):
    """Render velocity magnitude isosurface into the active subplot."""
    iso = grid.contour(isosurfaces=[isovalue], scalars='velocity_magnitude')
    if iso.n_points > 0:
        pl.add_mesh(iso, color='red', specular=1.0, specular_power=50,
                    ambient=0.1, smooth_shading=True, show_scalar_bar=False)
    else:
        print(f'    [VelocityIso] No surface at |u| = {isovalue} m/s')
    add_geometry_outline(pl, grid)
    pl.add_text('Velocity Isosurface', position='upper_left', font_size=20, color='black')
    pl.add_text(f'isovalue = {isovalue} m/s', position='lower_right', font_size=14, color='black')


def add_streamlines_panel(pl: pv.Plotter, grid: pv.UnstructuredGrid,
                          seed_coord: list, seed_radius: float,
                          vel_max: float = 2.0, t_label: float = None,
                          plot_seed_cloud: bool = False):
    """Render velocity streamlines into the active subplot."""
    streamlines = grid.streamlines(
        vectors='velocity',
        source_center=seed_coord,
        source_radius=seed_radius,
        n_points=120,
        initial_step_length=0.05,
        min_step_length=0.01,
        max_step_length=0.2,
        max_steps=10000,
        terminal_speed=0.05,
        interpolator_type='point',
        integrator_type=45,
        integration_direction='forward',
    )
    if streamlines.n_points > 0:
        tubes = streamlines.tube(radius=0.1)
        pl.add_mesh(tubes,
                    scalars='velocity_magnitude',
                    cmap='turbo',
                    clim=[0, vel_max],
                    specular=0.8, specular_power=30, ambient=0.15,
                    smooth_shading=True,
                    show_scalar_bar=True,
                    scalar_bar_args=dict(
                        title='Velocity (m/s)',
                        n_labels=5, fmt='%.2f',
                        height=0.05, width=0.2,
                        vertical=False,
                        position_x=0.95, position_y=0.2,
                        title_font_size=20,
                        label_font_size=20,
                    ))
    else:
        print('    [Streamlines] No streamlines generated -- check seed coord/radius.')
    add_geometry_outline(pl, grid)
    if plot_seed_cloud:
        seed_sphere = pv.Sphere(center=seed_coord, radius=5)
        pl.add_mesh(seed_sphere, color='yellow', opacity=0.3, show_scalar_bar=False)
    pl.add_text('Streamlines', position='upper_left', font_size=20, color='black')
    #if t_label is not None:
    #    pl.add_text(f't = {t_label:.4f} s', position='lower_right', font_size=14, color='black')


def add_qcriterion_panel(pl: pv.Plotter, grid: pv.UnstructuredGrid, isovalue: float):
    """Render Q-criterion isosurface into the active subplot."""
    Q = compute_qcriterion(grid)
    g = grid.copy()
    g['Qcriterion'] = Q
    iso = g.contour(isosurfaces=[isovalue], scalars='Qcriterion')
    if iso.n_points > 0:
        pl.add_mesh(iso, color='#FF6A00', smooth_shading=True,
                    specular=0.5, specular_power=20, ambient=0.2,
                    opacity=0.95, show_scalar_bar=False)
    else:
        print(f'    [Q-criterion]  No surface at Q = {isovalue} [1/s2]')
    add_geometry_outline(pl, grid)
    pl.add_text('Q-Criterion', position='upper_left', font_size=20, color='black')
    pl.add_text(f'Q = {isovalue:.0f} 1/s2', position='lower_right', font_size=14, color='black')


# ======================================================================================================
# PARALLEL FRAME WORKER
# Must be a module-level function (not nested) for multiprocessing to find it after fork.
# Each panel is rendered with its own independent plotter, then stacked vertically.
# ======================================================================================================

def render_frame_worker(task: dict) -> int:
    """Render one video frame to a PNG file. Returns frame index for progress tracking."""

    frame_i           = task['frame_i']
    t                 = task['t']
    snapshot_file     = Path(task['snapshot_file'])
    mesh_file         = Path(task['mesh_file'])
    cam_params        = task['cam_params']
    velocity_isovalue = task['velocity_isovalue']
    qcri_isovalue     = task['qcri_isovalue']
    vel_max           = task['vel_max']
    seed_coord        = task['stream_seed_coord']
    seed_radius       = task['stream_seed_radius']
    window_size       = task['window_size']
    output_png        = Path(task['output_png'])

    grid = load_mesh_from_h5(mesh_file)
    u    = load_velocity_snapshot(snapshot_file)
    attach_fields(grid, u)

    flowrate = 2.0 * t   # Qin [mL/s] = 2 * t [s]
    panels   = []

    # --- Panel 0: Velocity isosurface ---
    pl = make_panel_plotter(window_size)
    add_velocity_iso_panel(pl, grid, velocity_isovalue)
    apply_camera(pl, cam_params)
    pl.add_text(f'Case A', position='upper_left', font_size=35, color='black') #, bold=True)
    pl.add_text(f'Flow rate = {flowrate:.2f} mL/s', position='upper_right', font_size=30, color='black') #, bold=True)
    panels.append(pl.screenshot(return_img=True))
    pl.close()

    # --- Panel 1: Streamlines ---
    pl = make_panel_plotter(window_size)
    add_streamlines_panel(pl, grid,
                          seed_coord=seed_coord, seed_radius=seed_radius,
                          vel_max=vel_max, t_label=t)
    apply_camera(pl, cam_params)
    panels.append(pl.screenshot(return_img=True))
    pl.close()

    # --- Panel 2: Q-criterion ---
    pl = make_panel_plotter(window_size)
    add_qcriterion_panel(pl, grid, qcri_isovalue)
    apply_camera(pl, cam_params)
    panels.append(pl.screenshot(return_img=True))
    pl.close()

    # Stack panels vertically and save combined frame
    combined = np.vstack(panels)
    imageio.imwrite(str(output_png), combined)

    return frame_i


# ======================================================================================================
# VIDEO STITCHING
# ======================================================================================================

def stitch_video(frames_dir: Path, out_video: Path, framerate: int):
    """Stitch sequentially-numbered PNG frames into an MP4 using ffmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(framerate),
        '-i', str(frames_dir / 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        str(out_video),
    ]
    print('Stitching frames into video ...')
    print('  ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


# ======================================================================================================
# ARGUMENT PARSING
# ======================================================================================================

def parse_args():
    import yaml

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config_file', default=None)
    pre_args, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(
        description='Create a 3-panel hemodynamics video (velocity iso | streamlines | Q-criterion).')

    # ---- Config file ----
    ap.add_argument('--config_file', default=None,
                    help='Path to YAML config file with case-specific defaults (CLI args override)')

    # ---- Mesh & data (required) ----
    ap.add_argument('--mesh_folder',   required=True, help='Folder with BSLSolver HDF5 mesh (Mesh/coordinates + Mesh/topology)')
    ap.add_argument('--input_folder',  required=True, help='Folder with CFD HDF5 snapshots (*_curcyc_*up.h5)')
    ap.add_argument('--output_folder', required=True, help='Output directory for the video file')
    ap.add_argument('--case_name',     required=True, help='Case identifier used in the output filename')

    # ---- CFD cycle parameters ----
    ap.add_argument('--period_seconds',    type=float, default=None, help='Flow period [s] (default: 0.915)')
    ap.add_argument('--timesteps_per_cyc', type=int,   default=None, help='Timesteps per cycle (parsed from filename if omitted)')
    ap.add_argument('--save_freq',         type=int,   default=5,    help='Snapshot save frequency: every Nth timestep (default: 5)')

    # ---- Isosurface values ----
    ap.add_argument('--velocity_isovalue', type=float, default=0.5,    help='Velocity magnitude isosurface [m/s] (default: 0.5)')
    ap.add_argument('--qcri_isovalue',     type=float, default=1000.0, help='Q-criterion isosurface [1/s2] (default: 1000)')
    ap.add_argument('--vel_max',           type=float, default=2.0,    help='Streamline colormap upper bound [m/s] (default: 2.0)')

    # ---- Streamline seed ----
    ap.add_argument('--stream_seed_coord',  type=float, nargs=3, default=None, help='Seed center [x y z] for streamline sphere source')
    ap.add_argument('--stream_seed_radius', type=float,          default=None, help='Seed sphere radius (mesh units, mm)')

    # ---- Video options ----
    ap.add_argument('--framerate',    type=int,  default=60,   help='Output video frame rate [fps] (default: 60)')
    ap.add_argument('--start_frame',  type=int,  default=0,    help='First snapshot index, 0-based inclusive (default: 0)')
    ap.add_argument('--end_frame',    type=int,  default=None, help='Last snapshot index, 0-based exclusive (default: all frames)')
    ap.add_argument('--frame_stride', type=int,  default=1,    help='Render every Nth snapshot (default: 1 = no skipping)')
    ap.add_argument('--n_workers',    type=int,  default=None, help='Parallel worker processes (default: all available CPUs)')
    ap.add_argument('--keep_frames',  action='store_true',     help='Keep per-frame PNG files after stitching (default: delete)')

    # ---- Camera ----
    ap.add_argument('--cam_position',       type=float, nargs=3, default=None, help='Camera position [x y z]')
    ap.add_argument('--cam_focal_point',    type=float, nargs=3, default=None, help='Camera focal point [x y z]')
    ap.add_argument('--cam_view_up',        type=float, nargs=3, default=None, help='Camera view-up vector [x y z]')
    ap.add_argument('--cam_parallel_scale', type=float,          default=None, help='Parallel (orthographic) projection scale')

    # ---- Output ----
    ap.add_argument('--window_size', type=int, nargs=2, default=None,
                    help='Per-panel render window size [W H]; panels are stacked vertically (default: [1280, 720])')

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

    window_size = args.window_size if args.window_size is not None else [1280, 720]
    period      = args.period_seconds if args.period_seconds is not None else 0.915

    print('=' * 100)
    print('viz_HemodynamicsVideo.py')
    print(f'  Config           : {args.config_file}')
    print(f'  Input folder     : {input_folder}')
    print(f'  Output folder    : {output_folder}')
    print(f'  Case name        : {args.case_name}')
    print(f'  Velocity isovalue: {args.velocity_isovalue} m/s')
    print(f'  Q-criterion iso  : {args.qcri_isovalue} 1/s2')
    print(f'  Vel colormap max : {args.vel_max} m/s')
    print(f'  Frame rate       : {args.framerate} fps')
    print(f'  Window size      : {window_size}')
    print('=' * 100 + '\n')

    # Validate streamline seed
    if args.stream_seed_coord is None or args.stream_seed_radius is None:
        print('ERROR: --stream_seed_coord and --stream_seed_radius are required.')
        sys.exit(1)

    # ---- Discover mesh file ----
    mesh_file = next(Path(args.mesh_folder).glob('*.h5'), None)
    if mesh_file is None:
        print(f'ERROR: no .h5 mesh file found in {args.mesh_folder}')
        sys.exit(1)

    # ---- Discover all snapshots ----
    h5_files, timesteps_per_cyc = get_all_snapshots(input_folder, args.timesteps_per_cyc)
    dt       = args.save_freq * period / timesteps_per_cyc
    start    = args.start_frame
    end      = args.end_frame if args.end_frame is not None else len(h5_files)
    stride   = args.frame_stride
    frames   = h5_files[start:end:stride]
    n_frames = len(frames)

    n_workers = args.n_workers if args.n_workers is not None else mp.cpu_count()
    n_workers = min(n_workers, n_frames)

    print(f'Snapshots found   : {len(h5_files)}')
    print(f'Rendering frames  : [{start}, {end})  stride={stride}  ({n_frames} frames)')
    print(f'dt per video frame: {dt * stride:.4f} s')
    print(f'Video duration    : {n_frames / args.framerate:.1f} s at {args.framerate} fps')
    print(f'Workers           : {n_workers}\n')

    # ---- Camera dict ----
    cam_params = {
        'position':       args.cam_position,
        'focal_point':    args.cam_focal_point,
        'view_up':        args.cam_view_up,
        'parallel_scale': args.cam_parallel_scale,
    }

    # ---- Frames directory (holds per-frame PNGs in order) ----
    frames_dir = output_folder / f'{args.case_name}_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build task list ----
    task_list = []
    for i, snapshot_file in enumerate(frames):
        global_frame_idx = start + i * stride
        t = global_frame_idx * dt
        task_list.append({
            'frame_i':            i,
            't':                  t,
            'snapshot_file':      str(snapshot_file),
            'mesh_file':          str(mesh_file),
            'cam_params':         cam_params,
            'velocity_isovalue':  args.velocity_isovalue,
            'qcri_isovalue':      args.qcri_isovalue,
            'vel_max':            args.vel_max,
            'stream_seed_coord':  args.stream_seed_coord,
            'stream_seed_radius': args.stream_seed_radius,
            'window_size':        window_size,
            'output_png':         str(frames_dir / f'frame_{i:06d}.png'),
        })

    # ---- Parallel rendering ----
    print(f'Rendering {n_frames} frames with {n_workers} parallel workers ...')
    with mp.Pool(processes=n_workers) as pool:
        for done_i in pool.imap_unordered(render_frame_worker, task_list):
            print(f'  Done frame {done_i + 1:04d}/{n_frames}', flush=True)

    # ---- Stitch into MP4 ----
    end_label = end if args.end_frame is not None else 'end'
    out_video = output_folder / (
        f'{args.case_name}'
        f'_fr{start}-{end_label}'
        f'_stride{stride}'
        f'_fps{args.framerate}'
        f'_hemodynamics_video.mp4'
    )
    stitch_video(frames_dir, out_video, args.framerate)

    # ---- Clean up frames directory unless requested to keep ----
    if not args.keep_frames:
        shutil.rmtree(frames_dir)
        print(f'  Frames directory removed: {frames_dir}')

    print(f'\nVideo saved: {out_video}')
    print('Done.\n')


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)   # workers inherit parent's loaded VTK/pyvista — no re-import needed
    main()
