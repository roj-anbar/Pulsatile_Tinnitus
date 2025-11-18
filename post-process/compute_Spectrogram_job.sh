#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=00:59:59
#SBATCH --job-name PT_Spectrogram
#SBATCH --output=PT_Spectrogram_%j.txt


set -euo pipefail

# ---------------------------------- Define Paths ---------------------------------------
CASE=PTSeg028_base_0p64
BASE_DIR=$SCRATCH/PT/PT_Ramp/PT_cases/$CASE
MESH="$BASE_DIR/data"
INPUT="$BASE_DIR/results/${CASE}_ts10000_cy6_Q=2t_saveFreq1"
OUTPUT="$BASE_DIR/post-process/Spectrogram_wall_pressure/cy6"

SCRIPT="$SLURM_SUBMIT_DIR/compute_Spectrogram.py"  # ensure to submit from script dir


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
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=true
#export OMP_NUM_THREADS=1


# ------------------------------ Run Scripts ---------------------------------------------


#    --timesteps_per_cyc 10000 \

python "$SCRIPT" \
    --input_folder      "$INPUT" \
    --mesh_folder       "$MESH" \
    --output_folder     "$OUTPUT" \
    --case_name         "$CASE" \
    --n_process         "${SLURM_TASKS_PER_NODE}" \
    --period_seconds    0.915 \
    --spec_quantity     "pressure" \
    --ROI_type          "cylinder" \
    --ROI_center_csv    "$MESH/${CASE}_centerline_points.csv" \
    --ROI_radius        8 \
    --ROI_height        2 \
    --save_ROI_flag     True \
    --window_length     5000 \
    --overlap_fraction  0.9


python compute_Spectrogram2.py \
    --input_folder      "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/results/PTSeg028_base_0p64_ts10000_cy6_Q=2t_saveFreq1" \
    --mesh_folder       "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/data" \
    --case_name         "PTSeg028_base_0p64" \
    --output_folder     "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/post-process/Spectrogram_wall_pressure/cy6_test" \
    --n_process         192 \
    --spec_quantity     "pressure" \
    --ROI_type          "sphere" \
    --ROI_center_csv    "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/data/PTSeg028_base_0p64_centerline_points.csv" \
    --ROI_radius        8 \
    --ROI_height        2 \
    --save_ROI_flag     True \
    --window_length     5000 \
    --overlap_fraction  0.9

wait


# 