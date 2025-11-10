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
INPUT="$BASE_DIR/results/${CASE}_ts10000_cy6_saveFreq1"
OUTPUT="$BASE_DIR/post-process/Spectrogram/cy6_saveFreq1"

SCRIPT="$SLURM_SUBMIT_DIR/compute_Spectrograms.py"  # ensure to submit from script dir


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

#python "$SCRIPT" \
#    --input_folder     "$INPUT" \
#    --mesh_folder      "$MESH" \
#    --output_folder    "$OUTPUT" \
#    --case_name        "$CASE" \
#    --n_process        "${SLURM_TASKS_PER_NODE}" \
#    --period           0.915 \
#    --num_cycles       6 \
#    --spec_quantity    "pressure" \
#    --ROI_center_pid   4000 \
#    --ROI_radius       2


# (46.2749 -20.3932 10.7888)
#id=39: (33.6404, -9.21902, -3.76951)


python compute_Spectrogram.py \
    --input_folder      "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/results/PTSeg028_base_0p64_ts10000_cy6_Q=2t_saveFreq1" \
    --mesh_folder       "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/data" \
    --case_name         "PTSeg028_base_0p64" \
    --output_folder     "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/post-process/Spectrorgam_wall_pressure/cy6_saveFreq1" \
    --n_process         192 \
    --period_seconds    0.915 \
    --timesteps_per_cyc 10000 \
    --spec_quantity     "pressure" \
    --ROI_center        33.6404 -9.21902 -3.76951 \
    --ROI_radius        3 \
    --window_length     4000 \
    --overlap_fraction  0.8

wait
