#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# compute_SPI_job.sh
# SLURM wrapper to run compute_SPI.py for a specific case on Trillium style clusters.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2025-10
#
# PURPOSE:
#   - Define all case parameters for performing post-processing on CFD results.
#   - Optional flags let you override key settings without editing the file.
#
# REQUIREMENTS:
#   - compute_SPI.py (in the same directory as this bash file)
#   - A virtual environment including pyvista
#
# EXECUTION:
#   - Run this script from terminal by:
#     <sbatch compute_SPI_job.sh>
#
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=00:59:59
#SBATCH --job-name PT_SPI
#SBATCH --output=PT_SPI_%j.txt


set -euo pipefail

# ---------------------------------- Define Paths ---------------------------------------
CASE=PTSeg043_noLabbe_base
BASE_DIR=$SCRATCH/PT/PT_Ramp/PT_cases/$CASE
MESH="$BASE_DIR/data"
INPUT="$BASE_DIR/results/${CASE}_ts10000_cy6_saveFreq1"
OUTPUT="$BASE_DIR/post-process/SPI_WallPressure"
SCRIPT="$SLURM_SUBMIT_DIR/compute_SPI_old.py"  # ensure to submit from script dir


# --------------------------------- Load Modules ----------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.12.4
source $HOME/virtual_envs/pyvista36/bin/activate
module load vtk/9.3.0


# ------------------------------ Export Directories --------------------------------------
mkdir -p "$OUTPUT"
mkdir -p "$SCRATCH/.config/mpl" "$SCRATCH/.local/share/pyvista" "$SCRATCH/.local/temp"

export MPLCONFIGDIR=$SCRATCH/.config/mpl
export PYVISTA_USERDATA_PATH=$SCRATCH/.local/share/pyvista
export XDG_RUNTIME_DIR=$SCRATCH/.local/temp
export TEMPDIR=$SCRATCH/.local/temp
#export TMPDIR=$SCRATCH/.local/temp
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=true
#export OMP_NUM_THREADS=1


# ------------------------------ Run Scripts ---------------------------------------------

python "$SCRIPT" \
    --case_name     "$CASE" \
    --input_folder  "$INPUT" \
    --mesh_folder   "$MESH" \
    --output_folder "$OUTPUT" \
    --n_process     "${SLURM_TASKS_PER_NODE}" \
    --window_length  5000 \
    --window_overlap 0.9

#python compute_SPI_old.py \
#    --input_folder      "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/results/PTSeg028_base_0p64_ts10000_cy6_Q=2t_saveFreq1" \
#    --mesh_folder       "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/data" \
#    --case_name         "PTSeg028_base_0p64" \
#    --output_folder     "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/post-process/SPI_wall_pressure/cy6_test" \
#    --n_process         192 \
#    --window_length     5000 \
#    --window_overlap    0.9

#python compute_SPI.py \
#    --input_folder      "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/results/PTSeg028_base_0p64_ts10000_cy6_Q=2t_saveFreq1" \
#    --mesh_folder       "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/data" \
#    --case_name         "PTSeg028_base_0p64" \
#    --output_folder     "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/post-process/SPI_wall_pressure/cy6_test" \
#    --n_process         192 \
#    --window_length     5000 \
#    --overlap_fraction  0.5


wait