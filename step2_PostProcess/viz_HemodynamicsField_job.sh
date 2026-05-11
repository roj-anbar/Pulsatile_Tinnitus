#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# viz_HemodynamicsField_job.sh
# SLURM wrapper to run viz_HemodynamicsField.py for a specific case on Trillium-style clusters.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2026-05
#
# EXECUTION:
#   sbatch viz_HemodynamicsField_job.sh
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:20:00
#SBATCH --job-name PT_HemodynamicsField
#SBATCH --output=PT_HemodynamicsField_%j.txt

set -euo pipefail
echo "Job started: $(date)"

# ---------------------------------- Define Paths -----------------------------------------------------------------------
CASE=PTSeg028_base_0p64
BASE_DIR=$SCRATCH/My_Projects/Study1_PTRamp/cases/$CASE
MESH_FOLDER="$BASE_DIR/step1_CFD/data"
INPUT="$BASE_DIR/step1_CFD/results/${CASE}_ts10000_cy6_saveFreq5"
OUTPUT="$BASE_DIR/step2_PostProcess/HemodynamicsField"

SCRIPT="/scratch/ranbar/My_Projects/Study1_PTRamp/scripts/step2_PostProcess/viz_HemodynmicsField.py"

# --------------------------------- Load Modules ------------------------------------------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.12.4
source $HOME/virtual_envs/pyvista36/bin/activate

# ------------------------------ Export Directories --------------------------------------------------------------------
mkdir -p "$OUTPUT"
mkdir -p "$SCRATCH/.config/mpl"
export MPLCONFIGDIR=$SCRATCH/.config/mpl

# Suppress VTK/PyVista display (required for off-screen rendering on compute nodes)
export DISPLAY=""

# ------------------------------ Run Script ----------------------------------------------------------------------------
python "$SCRIPT"                     \
    --mesh_folder       "$MESH_FOLDER"   \
    --input_folder      "$INPUT"         \
    --output_folder     "$OUTPUT"        \
    --case_name         "$CASE"          \
    --target_time       2.79             \
    --save_freq         5                \
    --stream_seed_ids   100 500 9853     \
    --vel_isovalue      0.5              \
    --qcrit_values      5000             \
    --cam_position      0 0 500          \
    --cam_focal_point   0 0 0            \
    --cam_view_up       0 1 0            \
    --cam_parallel_scale 50              


#---------------------- For running directly from commandline use below ---------------------------
# Note1: You HAVE to load the modules first from terminal then run below
# Note2: You HAVE to comment this part if submitting this file through sbatch



echo "Job finished: $(date)"
