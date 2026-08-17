#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# viz_HemodynamicsVideo_job.sh
# SLURM wrapper to run viz_HemodynamicsVideo.py for a specific case on Trillium-style clusters.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2026-08
#
# EXECUTION:
#   sbatch viz_HemodynamicsVideo_job.sh
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=01:00:00
#SBATCH --job-name PT_HemodynamicsVideo
#SBATCH --output=PT_HemodynamicsVideo_%j.txt

set -euo pipefail
echo "Job started: $(date)"

# ---------------------------------- Define Paths -----------------------------------------------------------------------
CASE=PTSeg043_noLabbe_base
BASE_DIR=$SCRATCH/My_Projects/Study1_PTRamp/cases/$CASE
MESH_FOLDER="$BASE_DIR/step1_CFD/data"
INPUT="$BASE_DIR/step1_CFD/results/${CASE}_ts10000_cy6_saveFreq5"
OUTPUT="$BASE_DIR/step2_PostProcess/Hemodynamics/Videos"
CONFIG="$BASE_DIR/step2_PostProcess/configs/${CASE}_viz_config.yaml"

SCRIPT="/scratch/ranbar/My_Projects/Study1_PTRamp/scripts/step2_PostProcess/visualization/viz_HemodynamicsVideo.py"

# --------------------------------- Load Modules ------------------------------------------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.12.4
module load ffmpeg/7.1.1
source $HOME/virtual_envs/pyvista36/bin/activate
module load vtk/9.3.0


# ------------------------------ Export Directories --------------------------------------------------------------------
mkdir -p "$OUTPUT"
mkdir -p "$SCRATCH/.config/mpl"
export MPLCONFIGDIR=$SCRATCH/.config/mpl


# ------------------------------ Run Script ----------------------------------------------------------------------------
python "$SCRIPT"                            \
    --case_name         "$CASE"             \
    --input_folder      "$INPUT"            \
    --mesh_folder       "$MESH_FOLDER"      \
    --output_folder     "$OUTPUT"           \
    --config_file       "$CONFIG"           \
    --save_freq         5                   \
    --velocity_isovalue 0.5                 \
    --qcri_isovalue     50000               \
    --vel_max           2.0                 \
    --start_frame       2185                \
    --end_frame         8740                \
    --framerate         100                  \
    --frame_stride      4                 \


#---------------------- For running directly from commandline use below ---------------------------
# Note1: You HAVE to load the modules first from terminal then run below
# Note2: You HAVE to comment this part if submitting this file through sbatch

# python viz_HemodynamicsVideo.py                         \
#     --case_name         "PTSeg043_noLabbe_base"            \
#     --input_folder      "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step1_CFD/results/PTSeg043_noLabbe_base_ts10000_cy6_saveFreq5/" \
#     --mesh_folder       "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step1_CFD/data"   \
#     --output_folder     "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step2_PostProcess/Hemodynamics/Videos" \
#     --config_file       "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step2_PostProcess/configs/PTSeg043_noLabbe_base_viz_config.yaml" \
#     --save_freq         5                           \
#     --velocity_isovalue 0.5                         \
#     --qcri_isovalue     50000                       \
#     --vel_max           2.0                         \
#     --framerate         100                         \
#     --frame_stride      50                          \
#     --start_frame       8185                        \
#     --end_frame         8745                           


echo "Job finished: $(date)"
