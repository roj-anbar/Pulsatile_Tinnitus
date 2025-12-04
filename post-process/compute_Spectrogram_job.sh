#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=00:59:59
#SBATCH --job-name PT_Spectrogram
#SBATCH --output=PT_Spectrogram_%j.txt


set -euo pipefail

# ---------------------------------- Define Paths -------------------------------------------------------------------------------
CASE=PTSeg043_noLabbe_base                                              # Case name
BASE_DIR=$SCRATCH/PT/PT_Ramp/PT_cases/$CASE                             # Parent directory of the case
MESH="$BASE_DIR/data"                                                   # Path to mesh data folder containing the h5 mesh
CENTERLINE="$MESH/${CASE}_centerline_points_v6.csv"                     # Path to centerline csv file used to construct ROIs
INPUT="$BASE_DIR/results/${CASE}_ts10000_cy6_saveFreq1"                 # Path to CFD results folder containing timeseries HDF5 files
OUTPUT="$BASE_DIR/post-process/Spectrogram_wall_pressure/cy6_saveFreq1" # Path to saving spectrogram files

SCRIPT="$SLURM_SUBMIT_DIR/compute_Spectrogram.py"  # ensure to submit from script dir


# --------------------------------- Load Modules -------------------------------------------------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.12.4
source $HOME/virtual_envs/pyvista36/bin/activate
module load vtk/9.3.0


# ------------------------------ Export Directories ----------------------------------------------------------------------------
mkdir -p "$OUTPUT"
mkdir -p "$SCRATCH/.config/mpl"
#mkdir -p "$SCRATCH/.local/share/pyvista" "$SCRATCH/.local/temp"

export MPLCONFIGDIR=$SCRATCH/.config/mpl                    #matplotlib config
export PYVISTA_OFF_SCREEN=true                              #tells pyvista to use off-screen rendering

#export TMPDIR=$SCRATCH/.local/temp                          #temporary directory used by python
#export XDG_RUNTIME_DIR=$SCRATCH/.local/temp                 #runtime directory used by GUI stuff
#export PYVISTA_USERDATA_PATH=$SCRATCH/.local/share/pyvista  #pyvista userdata
#export PYVISTA_USE_PANEL=true
#export OMP_NUM_THREADS=1


# ------------------------------ Run Scripts ------------------------------------------------------------------------------------


#    --timesteps_per_cyc 10000 \

python "$SCRIPT" \
    --case_name         "$CASE" \
    --input_folder      "$INPUT" \
    --mesh_folder       "$MESH" \
    --output_folder     "$OUTPUT" \
    --n_process         "${SLURM_TASKS_PER_NODE}" \
    --period_seconds    0.915 \
    --spec_quantity     "pressure" \
    --ROI_type          "sphere" \
    --ROI_center_csv    "$CENTERLINE" \
    --ROI_radius        8 \
    --ROI_height        2 \
    --save_ROI_flag     True \
    --multi_ROI_flag    True \
    --window_length     5000 \
    --overlap_fraction  0.9


python compute_Spectrogram.py \
    --case_name         "PTSeg028_base_0p64" \
    --input_folder      "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/results/PTSeg028_base_0p64_ts10000_cy6_Q=2t_saveFreq1" \
    --mesh_folder       "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/data" \
    --output_folder     "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/post-process/Spectrogram_wall_pressure/cy6_saveFreq1" \
    --n_process         192 \
    --spec_quantity     "pressure" \
    --ROI_type          "cylinder" \
    --ROI_center_csv    "$SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/data/PTSeg028_base_0p64_centerline_points_clip_v7.csv" \
    --ROI_radius        8 \
    --ROI_height        1 \
    --save_ROI_flag     True \
    --multi_ROI_flag    True \
    --window_length     5000 \
    --overlap_fraction  0.9

wait

