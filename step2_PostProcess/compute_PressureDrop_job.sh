#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# compute_PressureDrop_job.sh
# SLURM wrapper to run compute_PressureDrop.py for a specific case on Trillium-style clusters.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2026-04
#
# EXECUTION:
#   sbatch compute_PressureDrop_job.sh
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=00:20:00
#SBATCH --job-name PT_PressureDrop
#SBATCH --output=PT_PressureDrop_%j.txt

set -euo pipefail
echo "Job started: $(date)"

# ---------------------------------- Define Paths -----------------------------------------------------------------------
CASE=PTSeg028_base_0p64
BASE_DIR=$SCRATCH/My_Projects/Study1_PTRamp/cases/$CASE
MESH_FOLDER="$BASE_DIR/step1_CFD/data"
CENTERLINE="$MESH_FOLDER/${CASE}_centerline_points.csv"
INPUT="$BASE_DIR/step1_CFD/results/${CASE}_ts10000_cy6_saveFreq1"
OUTPUT="$BASE_DIR/step2_PostProcess/Pressure"

SCRIPT="/scratch/ranbar/My_Projects/Study1_PTRamp/scripts/step2_PostProcess/compute_PressureDrop.py"

# --------------------------------- Load Modules ------------------------------------------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.12.4
source $HOME/virtual_envs/pyvista36/bin/activate

# ------------------------------ Export Directories --------------------------------------------------------------------
mkdir -p "$OUTPUT"
mkdir -p "$SCRATCH/.config/mpl"
export MPLCONFIGDIR=$SCRATCH/.config/mpl

# ------------------------------ Run Script ----------------------------------------------------------------------------
python "$SCRIPT" \
    --case_name         "$CASE"       \
    --input_folder      "$INPUT"      \
    --mesh_folder       "$MESH_FOLDER"\
    --output_folder     "$OUTPUT"     \
    --centerline_csv    "$CENTERLINE" \
    --inlet_point_id    1333          \
    --outlet_point_id   0             \
    --save_freq         1             \
    --flowrate_min      2.0           \
    --flowrate_max      10.0


# ------------------------------ Run directly from terminal -----------------------------------------------------------
python compute_PressureDrop.py \
    --case_name         "PTSeg043_noLabbe_base"       \
    --input_folder      "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step1_CFD/results/PTSeg043_noLabbe_base_ts10000_cy6_saveFreq5/"      \
    --mesh_folder       "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step1_CFD/data" \
    --output_folder     "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step2_PostProcess/Pressure"     \
    --centerline_csv    "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step1_CFD/data/PTSeg043_noLabbe_base_centerline_points.csv" \
    --inlet_point_id    1          \
    --outlet_point_id   614            \
    --save_freq         5             \
    --flowrate_min      2.0           \
    --flowrate_max      10.0


echo "Job finished: $(date)"
