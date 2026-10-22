#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# resample_cfd_results_job.sh
# SLURM wrapper to resample a CFD results folder to every nth timestep.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2026-10
#
# PURPOSE:
#   - Keep every nth timestep from a dense CFD results folder.
#   - Creates a new output folder with symlinks (no data copying) to the
#     selected HDF5 files and rewrites the XDMF so ParaView can open it directly.
#
# REQUIREMENTS:
#   - resample_cfd_results.py (in the same directory as this bash file)
#   - Python 3 (stdlib only, no extra packages needed)
#
# EXECUTION:
#   sbatch resample_cfd_results_job.sh
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name PT_Resample
#SBATCH --output=PT_Resample_%j.txt


set -euo pipefail
echo "Job started: $(date)"

# ---------------------------------- Define Paths -------------------------------------------------------------------------------
CASE=PTSeg043_noLabbe_base                                                                       # Case name
BASE_DIR=$SCRATCH/My_Projects/Study1_PTRamp/cases/$CASE                                          # Parent directory of the case
RESULTS_DIR="$BASE_DIR/step1_CFD/results_inletBC_ramp/rampSlope2mLs2"                            # Parent folder of CFD result runs

INPUT_FOLDERNAME="${CASE}_ts10000_cy6_saveFreq1"                                                 # Source results folder name
INPUT_XDMF="$RESULTS_DIR/$INPUT_FOLDERNAME/${CASE}_ts10000_cy6.xdmf"                             # Source XDMF file

STRIDE=5000                                                                                      # Keep every nth timestep
START=0                                                                                          # Index offset (0 = start from the first timestep)

OUTPUT_FOLDERNAME="${CASE}_ts10000_cy6_saveFreq${STRIDE}"                                         # Output folder (auto-rename saveFreq)
OUTPUT_FOLDER="$RESULTS_DIR/$OUTPUT_FOLDERNAME"                                                   # Full path to output folder

SCRIPT="$SCRATCH/My_Projects/Study1_PTRamp/scripts/utilities/resample_cfd_results.py"


# --------------------------------- Load Modules -------------------------------------------------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.12.4


# ------------------------------ Run Script ------------------------------------------------------------------------------------
python -u "$SCRIPT" \
          --input_xdmf "$INPUT_XDMF" \
          --output_path "$OUTPUT_FOLDER" \
          --stride      "$STRIDE" \
          --start       "$START"


wait
echo "Job finished: $(date)"
