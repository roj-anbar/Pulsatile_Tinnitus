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

SCRIPT="/scratch/ranbar/My_Projects/Study1_PTRamp/scripts/step2_PostProcess/visualization/viz_HemodynmicsField.py"

# --------------------------------- Load Modules ------------------------------------------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.12.4
source $HOME/virtual_envs/pyvista36/bin/activate

# ------------------------------ Export Directories --------------------------------------------------------------------
mkdir -p "$OUTPUT"
mkdir -p "$SCRATCH/.config/mpl"
export MPLCONFIGDIR=$SCRATCH/.config/mpl


# ------------------------------ Run Script ----------------------------------------------------------------------------
python "$SCRIPT"                     \
    --case_name         "$CASE"          \
    --input_folder      "$INPUT"         \
    --mesh_folder       "$MESH_FOLDER"   \
    --output_folder     "$OUTPUT"        \
    --target_time       2.79             \
    --save_freq         5                \
    --stream_seed_ids   100 500 9853     \
    --velocity_isovalue      0.5              \
    --qcri_isovalue      5000             \
    --cam_position      -285 -33.8 102   \
    --cam_focal_point   20.6 -35.1 -2.8            \
    --cam_view_up       0.32 0.04 0.94            \
    --cam_parallel_scale 101              

#---------------------- For running directly from commandline use below ---------------------------
# Note1: You HAVE to load the modules first from terminal then run below
# Note2: You HAVE to comment this part if submitting this file through sbatch

python viz_HemodynmicsField.py                     \
    --case_name         "PTSeg028_base_0p64"          \
    --input_folder      "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step1_CFD/results/PTSeg028_base_0p64_ts10000_cy6_saveFreq5/"         \
    --mesh_folder       "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step1_CFD/data"   \
    --output_folder     "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step2_PostProcess/Hemodynamics/Field"        \
    --target_time       4             \
    --save_freq         5                \
    --stream_seed_ids   100 500 9853     \
    --velocity_isovalue      0.5              \
    --qcri_isovalue      5000             \
    --cam_position      -285 -33.8 102   \
    --cam_focal_point   20.6 -35.1 -2.8            \
    --cam_view_up       0.32 0.04 0.94            \
    --cam_parallel_scale 101        

echo "Job finished: $(date)"
