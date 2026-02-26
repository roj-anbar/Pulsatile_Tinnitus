#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# compute_Qcriterion_job.sh
# SLURM wrapper to run compute_Qcriterion.py for a specific case on Trillium style clusters.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2025-10
#
# PURPOSE:
#   - Define all case parameters for performing post-processing on CFD results.
#   - Optional flags let you override key settings without editing the file.
#
# REQUIREMENTS:
#   - compute_Qcriterion.py (in the same directory as this bash file)
#   - A virtual environment including pyvista
#
# EXECUTION:
#   - Run this script from terminal by:
#     <sbatch compute_Qcriterion_job.sh>
#
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=08:59:59
#SBATCH --job-name PT_Qcriterion
#SBATCH --output=PT_Qcriterion_%j.txt


set -euo pipefail

# ---------------------------------- Define Paths ---------------------------------------
CASE=PTSeg028_base
BASE_DIR=$SCRATCH/My_Projects/Study1_PTRamp/cases/$CASE
MESH="$BASE_DIR/step1_CFD/data"
INPUT="$BASE_DIR/step1_CFD/results/${CASE}_ts10000_cy6_saveFreq5"
OUTPUT="$BASE_DIR/step2_PostProcess/Qcriterion/cy6_saveFreq5"
SCRIPT="$SLURM_SUBMIT_DIR/compute_Qcriterion.py"  # ensure to submit from script dir


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
    --mesh_folder             "$MESH" \
    --input_folder            "$INPUT" \
    --output_folder           "$OUTPUT" \
    --case_name               "$CASE" \
#    --flag_normalize_velocity 


#python compute_Qcriterion.py \
#    --case_name     "PTSeg028_base" \
#    --mesh_folder   "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base/step1_CFD/data" \
#    --input_folder  "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base/step1_CFD/results/PTSeg028_base_ts10000_cy6_saveFreq5" \
#    --output_folder "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base/step2_PostProcess/Qcriterion/cy6_saveFreq5" \
    
#    --flag_normalize_velocity 

wait