#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=00:59:59
#SBATCH --job-name PT_Spectrogram
#SBATCH --output=PT_Spectrogram_%j.txt


set -euo pipefail

# ---------------------------------- Define Paths -------------------------------------------------------------------------------
CASE=PTSeg106_base_0p64                                             # Case name
BASE_DIR=$SCRATCH/PT/PT_Ramp/cases/$CASE                             # Parent directory of the case
MESH="$BASE_DIR/data"                                                   # Path to mesh data folder containing the h5 mesh
CENTERLINE="$MESH/${CASE}_centerline_points.csv"                     # Path to centerline csv file used to construct ROIs
INPUT="$BASE_DIR/results/${CASE}_ts10000_cy6_saveFreq1"                 # Path to CFD results folder containing timeseries HDF5 files
OUTPUT="$BASE_DIR/post-process/Spectrogram_wall_pressure/cy6_saveFreq1/run2_centerline_fine" # Path to saving spectrogram files

SCRIPT="/scratch/ranbar/PT/PT_Ramp/scripts/post-process/compute_Spectrogram.py"  # ensure to submit from script dir


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


#--timesteps_per_cyc 10000 \

# Run the script once for each topological region of interest (i.e. stenosis, sigmoid sinus, ...)

python "$SCRIPT" \
    --case_name             "$CASE" \
    --input_folder          "$INPUT" \
    --mesh_folder           "$MESH" \
    --output_folder         "$OUTPUT" \
    --ROI_center_csv        "$CENTERLINE" \
    --n_process             "${SLURM_TASKS_PER_NODE}" \
    --spec_quantity         "pressure" \
    --window_length         5000 \
    --overlap_fraction      0.9 \
    --ROI_type              "cylinder" \
    --ROI_radius            8 \
    --ROI_height            2 \
    --ROI_start_center_id   1280 \
    --ROI_end_center_id     1360 \
    --ROI_stride            4 \
    --multi_ROI_flag        True


python "$SCRIPT" \
    --case_name             "$CASE" \
    --input_folder          "$INPUT" \
    --mesh_folder           "$MESH" \
    --output_folder         "$OUTPUT" \
    --ROI_center_csv        "$CENTERLINE" \
    --n_process             "${SLURM_TASKS_PER_NODE}" \
    --spec_quantity         "pressure" \
    --window_length         5000 \
    --overlap_fraction      0.9 \
    --ROI_type              "cylinder" \
    --ROI_radius            8 \
    --ROI_height            2 \
    --ROI_start_center_id   1200 \
    --ROI_end_center_id     1240 \
    --ROI_stride            4 \
    --multi_ROI_flag        True


python "$SCRIPT" \
    --case_name             "$CASE" \
    --input_folder          "$INPUT" \
    --mesh_folder           "$MESH" \
    --output_folder         "$OUTPUT" \
    --ROI_center_csv        "$CENTERLINE" \
    --n_process             "${SLURM_TASKS_PER_NODE}" \
    --spec_quantity         "pressure" \
    --window_length         5000 \
    --overlap_fraction      0.9 \
    --ROI_type              "cylinder" \
    --ROI_radius            8 \
    --ROI_height            2 \
    --ROI_start_center_id   1100 \
    --ROI_end_center_id     1182 \
    --ROI_stride            4 \
    --multi_ROI_flag        True


python "$SCRIPT" \
    --case_name             "$CASE" \
    --input_folder          "$INPUT" \
    --mesh_folder           "$MESH" \
    --output_folder         "$OUTPUT" \
    --ROI_center_csv        "$CENTERLINE" \
    --n_process             "${SLURM_TASKS_PER_NODE}" \
    --spec_quantity         "pressure" \
    --window_length         5000 \
    --overlap_fraction      0.9 \
    --ROI_type              "cylinder" \
    --ROI_radius            8 \
    --ROI_height            2 \
    --ROI_start_center_id   990 \
    --ROI_end_center_id     1090 \
    --ROI_stride            4 \
    --multi_ROI_flag        True


# For running directly from commandline use below
#python compute_Spectrogram.py \
#    --case_name             "PTSeg043_noLabbe_base" \
#    --input_folder          "$SCRATCH/PT/PT_Ramp/cases/PTSeg043_noLabbe_base/results/PTSeg043_noLabbe_base_ts10000_cy6_saveFreq1" \
#    --mesh_folder           "$SCRATCH/PT/PT_Ramp/cases/PTSeg043_noLabbe_base/data" \
#    --output_folder         "$SCRATCH/PT/PT_Ramp/cases/PTSeg043_noLabbe_base/post-process/Spectrogram_wall_pressure/cy6_saveFreq1" \
#    --ROI_center_csv        "$SCRATCH/PT/PT_Ramp/cases/PTSeg043_noLabbe_base/data/PTSeg043_noLabbe_base_centerline_points.csv" \
#    --n_process             192 \
#    --spec_quantity         "pressure" \
#    --window_length         5000 \
#    --overlap_fraction      0.9 \
#    --ROI_type              "cylinder" \
#    --ROI_radius            8 \
#    --ROI_height            2 \
#    --ROI_start_center_id   986 \
#    --ROI_end_center_id     1080 \
#    --ROI_stride            2 \
#    --multi_ROI_flag        True


wait

