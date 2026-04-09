# -----------------------------------------------------------------------------------------------------------------------
# viz_SpectrogramRegions.py
# To visualize ROI regions used to generate regional spectrograms.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2026-04
#
# PURPOSE:
#   - This script is part of the BSL post-processing pipeline.
#   - Reads the CFD HDF5 mesh and the spectrogram-regions CSV, assigns each surface mesh
#     point to an anatomical ROI, and renders a color-coded PNG image.
#   - Each region is colored by the REGION_COLORS dict defined in this script.
#
# REQUIREMENTS:
#   - pyvista, numpy
#   - On Trillium: virtual environment called "pyvista36"
#       > module load StdEnv/2023 gcc/12.3 python/3.12.4
#       > source $HOME/virtual_envs/pyvista36/bin/activate
#       > module load vtk/9.3.0
#
# EXAMPLE CLI (required arguments):
#       > python viz_SpectrogramRegions.py \
#           --mesh_stl       cases/PTSeg028_base_0p64/step0_PreProcess/mesh/PTSeg028_base_0p64.stl \
#           --centerline_csv cases/PTSeg028_base_0p64/step1_CFD/data/PTSeg028_base_0p64_centerline_points_clip_v7.csv \
#           --regions_csv    cases/PTSeg028_base_0p64/step2_PostProcess/PTSeg028_base_0p64_spectrogram_regions.csv \
#           --output_png     cases/PTSeg028_base_0p64/step2_PostProcess/PTSeg028_base_0p64_ROI_regions.png
#
# INPUTS:
#   --mesh_stl       STL file of surface mesh
#   --centerline_csv Centerline CSV   (columns: Points:0/1/2, FrenetTangent:0/1/2)
#   --regions_csv    Regions   CSV    (columns: ROI_start_center_id, ROI_end_center_id,
#                                               region_name [optional], region_abbrev [optional])
#   --output_png     Output PNG path  (default: <regions_csv_dir>/<case>_ROI_regions.png)
#
# OPTIONAL:
#   --bg_color      Background color string           (default: white)
#   --mesh_opacity  Opacity of non-ROI mesh           (default: 0.12)
#   --mesh_color    Color string for non-ROI mesh     (default: lightgrey)
#   --window_size   Image width height in pixels      (default: 1800 2000)
#   --add_legend    Flag: add a region color legend
#   --camera_pos    Camera view preset: auto|xy|xz|yz (default: auto)
#
# HOW REGION CLIPPING WORKS:
#   Each region is bounded by two cross-sectional cut planes derived from the centerline:
#
#     Start plane:  origin = centerline_coords[ROI_start_center_id]
#                   normal = FrenetTangent    [ROI_start_center_id]
#     End   plane:  origin = centerline_coords[ROI_end_center_id]
#                   normal = FrenetTangent    [ROI_end_center_id]
#
#   The FrenetTangent is the unit tangent to the centerline — perpendicular to the vessel
#   cross-section at that point — so it is exactly the cut-plane normal we need.
#
#   Clipping is done with pyvista mesh.clip() (wraps VTK vtkClipPolyData), the same
#   filter used in ParaView.  Two sequential clips isolate each region:
#
#     Step A – clip at start plane, keep downstream side  (invert=True)
#     Step B – clip at end   plane, keep upstream   side  (invert=False)
#
#   This produces geometrically clean cross-section edges at each boundary (VTK
#   interpolates new vertices on the cut plane) rather than jagged triangle edges.
#
# OUTPUTS:
#   PNG image of the colored mesh saved at --output_png.
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------

import argparse
from pathlib import Path

#import h5py
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True    # must be set before any Plotter is created


# ======================================================================================================
# REGION COLOR DEFINITIONS
# ======================================================================================================
# Edit RGB tuples (values in [0.0, 1.0]) to change colors for each region.
# Keys must match 'region_name' values in the regions CSV (case-sensitive).


REGION_COLORS = {
    "1_SSS":      (0.53, 0.81, 0.98),   # light sky blue
    "2_TS":       (0.00, 0.90, 0.90),   # cyan
    "3_TSS":      (0.58, 0.44, 0.86),   # medium purple
    "4_PSR":      (1.00, 0.41, 0.71),   # hot pink
    "5_SS":       (0.24, 0.70, 0.44),   # medium sea green
    "6_JB":       (0.20, 0.80, 0.20),   # lime green
}


# ======================================================================================================
# STEP 1 – LOADING INPUT FILES
# ======================================================================================================

def load_surface_mesh(mesh_file:Path) -> pv.PolyData:
    """
    Build a Pyvista PolyData surface from the wall mesh stored in a BSLSolver-style HDF5 file.

    Expects HDF5 layout:
      Mesh/Wall/coordinates : (Npoints, 3) float    – XYZ node positions
      Mesh/Wall/topology    : (Ncells, 3 or 4) int  – triangle connectivity
      Mesh/Wall/pointIds    : (Npoints,) int        – global volume-mesh point IDs
    """

    with h5py.File(mesh_file, 'r') as h5:
        coords = np.array(h5['Mesh/Wall/coordinates'])      # coords of wall points (n_points, 3)
        cells  = np.array(h5['Mesh/Wall/topology'])         # connectivity of wall points (n_cells, 3) -> triangles
        point_ids   = np.array(h5['Mesh/Wall/pointIds'])    # mapping to volume point IDs (n_points,)
        
    # Create connectivity array compatible with VTK --> requires a size prefix per cell (here '3' for triangles)
    n_cells        = cells.shape[0]
    node_per_cell  = 3                                                      # the surface cells are triangles with size of 3 (3 nodes per elem)
    cell_size      = np.full((n_cells, 1), node_per_cell, dtype=np.int64)   # array of size (n_cells, 1) filled with 3 
    cells_vtk      = np.hstack([cell_size, cells]).ravel()                  # horizontal stacking of arrays / ravel: flattens the array into a 1d array
        
    # Build surface and attach point ID
    surf = pv.PolyData(coords, cells_vtk)
    surf.point_data['vtkOriginalPtIds'] = point_ids

    return surf


def load_centerline_csv(csv_path: Path):
    """
    Read all centerline points and store XYZ positions and Frenet tangents.

    Column names after numpy strips surrounding quotes:
      Points0, Points1, Points2
      FrenetTangent0, FrenetTangent1, FrenetTangent2

    Returns
    -------
    coords  : (N_cl, 3)  XYZ positions along the centerline
    normals : (N_cl, 3)  unit tangent vectors (= cut-plane normals)
    """
    data    = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    coords  = np.column_stack([data["Points0"],         data["Points1"],        data["Points2"]])
    normals = np.column_stack([data["FrenetTangent0"],  data["FrenetTangent1"], data["FrenetTangent2"]])

    return coords, normals


def load_regions_csv(csv_path: Path) -> list:
    """
    Read ROI region definitions from the spectrogram regions CSV.

    Required columns : ROI_start_center_id, ROI_end_center_id
    Optional columns : region_name, region_abbrev

    Returns a list of dicts, one per region row.
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")

    int_cols   = {"ROI_start_center_id", "ROI_end_center_id"}
    str_cols   = {"region_fullname", "region_shortname"}

    regions = []
    for row in data:
        rec = {}
        for key in data.dtype.names:
            val = row[key]
            if key in int_cols:
                rec[key] = int(val)
            elif key in str_cols:
                rec[key] = str(val).strip()
        regions.append(rec)
    return regions


# HELPER – pick color for a region
def get_region_color(roi_params: dict) -> tuple:
    """
    Return the RGB color tuple for a region.
    Looks up REGION_COLORS by region_shortname.
    Falls back to mid-grey if the key is not found in REGION_COLORS.
    """
    name = roi_params.get("region_shortname", "")
    return REGION_COLORS.get(name, (0.5, 0.5, 0.5))
  


# ======================================================================================================
# STEP 4 – CLIP EACH REGION FROM THE SURFACE MESH
# ======================================================================================================

def clip_region_of_mesh(surf: pv.PolyData,
                        region_params: dict,
                        centerline_coords: np.ndarray,
                        centerline_normals: np.ndarray) -> pv.PolyData:
    """
    Extract the vessel segment for one region by applying two sequential plane clips.

    PyVista clip() convention (wraps VTK vtkClipPolyData):
      invert=False  keeps the half where  normal · (P - origin) <= 0
      invert=True   keeps the half where  normal · (P - origin) >= 0

    Because the FrenetTangent points in the direction of increasing centerline index:
      - Start clip (invert=True)  → keep downstream side  (past the start plane)
      - End   clip (invert=False) → keep upstream   side  (before the end plane)

    """
    start = region_params["ROI_start_center_id"]
    end   = region_params["ROI_end_center_id"]

    # Step A: clip at start plane, keep the downstream side
    clipped  = surf.clip(normal=centerline_normals[start],  origin=centerline_coords[start], invert=True)
    
    # Step B: clip at end plane,   keep the upstream side
    clipped = clipped.clip(normal=centerline_normals[end], origin=centerline_coords[end], invert=False)

    return clipped


def clip_all_regions(surf: pv.PolyData,
                    region_params: list,
                    centerline_coords: np.ndarray,
                    centerline_normals: np.ndarray) -> list:
    """
    Clip each region and return a list of (region_dict, clipped_PolyData) tuples.
    Each tuple pairs the original region metadata with its clipped mesh.
    clipped_regions[i] is (region_params, roi_surf)
    """
    clipped_regions = []
    for idx, roi_params in enumerate(region_params):
        clipped_roi = clip_region_of_mesh(surf, roi_params, centerline_coords, centerline_normals)
        clipped_regions.append((roi_params, clipped_roi))

    return clipped_regions



# ======================================================================================================
# STEP 5 – RENDER AND SAVE PNG
# ======================================================================================================

def render_and_save(surf: pv.PolyData,
                    clipped_regions: list,
                    output_png: Path,
                    bg_color: str,
                    mesh_opacity: float,
                    mesh_color: str,
                    window_size: tuple,
                    camera_pos: str):
    """
    Render the mesh with colored ROI regions and save as PNG.

    Strategy:
      1. Full mesh  – semi-transparent, neutral color  (anatomical context background)
      2. Each clipped ROI mesh – solid, region-specific color layered on top
    """

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background(bg_color)

    # ---- Full mesh as semi-transparent background --------------------------------------
    plotter.add_mesh(surf,
                     color=mesh_color,
                     opacity=mesh_opacity,
                     smooth_shading=True,
                     show_scalar_bar=False)

    # ---- Colored clipped ROI actors ----------------------------------------------------

    # for idx, (roi_params, roi_surf) in enumerate(clipped_regions):
    #     color = get_region_color(roi_params)

    #     plotter.add_mesh(roi_surf,
    #                      color=color,
    #                      opacity=1.0,
    #                      smooth_shading=True,
    #                      show_scalar_bar=False)



    # ---- Camera ------------------------------------------------------------------------
    if camera_pos in ("xy", "xz", "yz"):
        plotter.camera_position = camera_pos
    else:
        plotter.camera_position = "iso"   # isometric view

    plotter.reset_camera()
    plotter.camera.zoom(1.2)

    # ---- Lighting (soft diffuse, good for medical meshes) ------------------------------
    plotter.enable_lightkit()

    # ---- Save --------------------------------------------------------------------------
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_png), transparent_background=False)
    plotter.close()
    print(f"    PNG saved → {output_png}")


# ======================================================================================================
# CLI
# ======================================================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Render anatomical ROI regions on the CFD surface mesh and save a PNG.")

    # Define paths
    p.add_argument("--path_mesh_stl",       required=True,  type=Path,  help="Path to surface mesh STL file.")
    p.add_argument("--path_centerline_csv", required=True,  type=Path,  help="Centerline CSV with Points:0/1/2 and FrenetTangent:0/1/2 columns.")
    p.add_argument("--path_regions_csv",    required=True,  type=Path,  help="Spectrogram regions CSV with ROI_start/end_center_id columns.")
    p.add_argument("--path_output_png",     required=True,  type=Path,  help="Output PNG path (default: <regions_csv_dir>/<case>_ROI_regions.png).")

    # Rendering properties
    p.add_argument("--bg_color",     default="white",       type=str,   help="Background color (default: white).")
    p.add_argument("--mesh_opacity", default=0.1,           type=float, help="Opacity of the non-ROI mesh (default: 0.1).")
    p.add_argument("--mesh_color",   default="lightgrey",   type=str,   help="Color of the non-ROI mesh (default: lightgrey).")

    # Camera params
    p.add_argument("--window_size",  default=[1800, 2000],  type=int,   nargs=2, metavar=("WIDTH", "HEIGHT"), help="Output image size in pixels (default: 1800 2000).")
    p.add_argument("--camera_pos",   default="auto",        choices=["auto", "xy", "xz", "yz"], help="Camera view preset: auto (isometric) | xy | xz | yz (default: auto).")

    return p.parse_args()


# ======================================================================================================
# MAIN
# ======================================================================================================

def main():
    args = parse_args()


    print("\n" + "=" * 200)
    print("  viz_SpectrogramRegions.py")
    print("=" * 70)
    print(f"  [info] Mesh STL file  : {args.path_mesh_stl}")
    print(f"  [info] Centerline CSV : {args.path_centerline_csv}")
    print(f"  [info] Regions CSV    : {args.path_regions_csv}")
    print(f"  [info] Output PNG     : {args.path_output_png}")
    print("=" * 200 + "\n")

    # Step 1 – Mesh
    print("[1/5] Loading surface mesh from STL file...")
    #surf = load_surface_mesh(mesh_file)
    surf = pv.read(args.path_mesh_stl)
    print(f"      {surf.n_points} points, {surf.n_cells} cells\n")

    # Step 2 – Centerline
    print("[2/5] Loading centerline from CSV file...")
    centerline_coords, centerline_normals = load_centerline_csv(args.path_centerline_csv)

    # Step 3 – Regions
    print("[3/5] Loading regions from CSV file...")
    region_params = load_regions_csv(args.path_regions_csv)

    # Step 4 – Clip each region
    print("[4/5] Clipping all regions from surface mesh ...")
    clipped_regions = clip_all_regions(surf, region_params, centerline_coords, centerline_normals)


    # Step 5 – Render
    print("[5/5] Rendering and saving PNG ...")
    render_and_save(surf,
                    clipped_regions,
                    output_png   = args.path_output_png,
                    bg_color     = args.bg_color,
                    mesh_opacity = args.mesh_opacity,
                    mesh_color   = args.mesh_color,
                    window_size  = tuple(args.window_size),
                    camera_pos   = args.camera_pos)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
