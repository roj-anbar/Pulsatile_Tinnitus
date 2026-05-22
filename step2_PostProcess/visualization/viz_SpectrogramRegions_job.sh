#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# viz_SpectrogramRegions_job.sh
# SLURM wrapper to run viz_SpectrogramRegions.py for a specific case on Trillium-style clusters.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2026-04
#
# PURPOSE:
#   - Define all case-specific paths and rendering parameters.
#   - Renders the CFD surface mesh with each anatomical ROI colored differently and saves a PNG.
#
# REQUIREMENTS:
#   - viz_SpectrogramRegions.py (in the same directory as this script)
#   - A virtual environment including pyvista (called 'pyvista36' on ranbar's Trillium account)
#   - The rendering is problematic on Trillium, so running this script locally.
#
# EXECUTION:
#   - Submit via SLURM:  sbatch viz_SpectrogramRegions_job.sh
#   - Run interactively:
#       module load StdEnv/2023 gcc/12.3 python/3.12.4
#       source $HOME/virtual_envs/pyvista36/bin/activate
#       module load vtk/9.3.0
#       bash viz_SpectrogramRegions_job.sh
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --time=00:20:00
#SBATCH --job-name PT_VizRegions
#SBATCH --output=PT_VizRegions_%j.txt

set -euo pipefail
echo "Job started: $(date)"

# ---------------------------------- Define Paths -----------------------------------------------------------------------
CASE=PTSeg028_base_0p64                                                             # Case name
BASE_DIR=$SCRATCH/My_Projects/Study1_PTRamp/cases/$CASE                             # Parent directory of the case
MESH_STL="$BASE_DIR/step1_CFD/data/${CASE}.h5"                                      # Path to STL mesh surface file
CENTERLINE_CSV="$BASE_DIR/step1_CFD/data/${CASE}_centerline_points.csv"             # Path to centerline CSV
REGIONS_CSV="$BASE_DIR/step2_PostProcess/configs/${CASE}_spectrogram_regions.csv"  # Path to spectrogram regions CSV
OUTPUT_PNG="$BASE_DIR/step2_PostProcess/Spectrogram_wallpressure/${CASE}_spec_regions.png"     # Path to output PNG

SCRIPT="/scratch/ranbar/My_Projects/Study1_PTRamp/scripts/step2_PostProcess/visualization/viz_SpectrogramRegions.py"

# --------------------------------- Load Modules ------------------------------------------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.12.4
source $HOME/virtual_envs/pyvista36/bin/activate
module load vtk/9.3.0

# --------------------------------- Create Output Directories -----------------------------------------------------------
mkdir -p "$(dirname "$OUTPUT_PNG")"



# --------------------------------- Run Script --------------------------------------------------------------------------
python "$SCRIPT" \
    --path_mesh_stl       "$MESH_STL"    \
    --path_centerline_csv "$CENTERLINE_CSV" \
    --path_regions_csv    "$REGIONS_CSV"    \
    --path_output_png     "$OUTPUT_PNG"     
#    --bg_color            "white"           \
#    --mesh_opacity        0.1               \
#    --mesh_color          "lightgrey"       \
#    --window_size         1800 2000         \
#    --camera_pos          "auto"




#---------------------- For running directly from commandline use below ---------------------------
# Note1: You HAVE to load the modules first from terminal then run below
# Note2: You HAVE to comment this part if submitting this file through sbatch

# python viz_SpectrogramRegions.py \
#     --path_mesh_stl       "/Users/rojin/Dropbox/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step1_CFD/data/PTSeg028_base_0p64.h5" \
#     --path_centerline_csv "/Users/rojin/Dropbox/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step1_CFD/data/PTSeg028_base_0p64_centerline_points.csv" \
#     --path_regions_csv    "/Users/rojin/Dropbox/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step2_PostProcess/PTSeg028_base_0p64_spectrogram_regions_test.csv"    \
#     --path_output_png     "/Users/rojin/Dropbox/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step2_PostProcess/Visualization/PTSeg028_base_0p64_spec_regions.png"


wait
