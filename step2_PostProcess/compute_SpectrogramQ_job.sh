#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=00:59:59
#SBATCH --job-name PT_Spectrogram
#SBATCH --output=PT_Spectrogram_%j.txt


set -euo pipefail

# ---------------------------------- Define Paths -------------------------------------------------------------------------------
CASE=PTSeg028_base_0p64                                             # Case name
BASE_DIR=$SCRATCH/My_Projects/Study1_PTRamp/cases/$CASE                            # Parent directory of the case
MESH="$BASE_DIR/step1_CFD/data"                                     # Path to mesh data folder containing the h5 mesh
CENTERLINE="$MESH/${CASE}_centerline_points.csv"                    # Path to centerline csv file used to construct ROIs
INPUT="$BASE_DIR/step2_PostProcess/Qcriterion/cy3_saveFreq5"   # Path to CFD results folder containing timeseries HDF5 files
OUTPUT="$BASE_DIR/step2_PostProcess/Qcriterion/cy3_saveFreq5/specs"      # Path to saving spectrogram files

SCRIPT="/scratch/ranbar/My_Projects/Study1_PTRamp/scripts/step2_PostProcess/compute_Spectrogram.py"


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


# ------------------------------ Run Scripts ------------------------------------------------------------------------------------


# Run the script once for each anatomical region of interest (i.e. stenosis, sigmoid sinus, ...).
# You can obtain the ROI parameters (e.g. ROI_start_center_id, ROI_end_center_id, ...) from the spreadsheet below:
# https://utoronto-my.sharepoint.com/:x:/g/personal/rojin_anbarafshan_mail_utoronto_ca/IQAE4WxcfxZtTa3r6agUO6xUAUsh_3MP7hhnwm9_BzdpsV0?e=WVALzt



# Region 1: Transverse Sinus
python "$SCRIPT" \
    --case_name             "$CASE" \
    --input_folder          "$INPUT" \
    --mesh_folder           "$MESH" \
    --output_folder         "$OUTPUT" \
    --ROI_center_csv        "$CENTERLINE" \
    --n_process             "${SLURM_TASKS_PER_NODE}" \
    --spec_quantity         "qcriterion" \
    --window_length         2732 \
    --ROI_type              "cylinder" \
    --ROI_radius            8 \
    --ROI_height            2 \
    --ROI_stride            2 \
    --ROI_start_center_id   627 \
    --ROI_end_center_id     664 \
    --flag_multi_ROI        \
    --timesteps_per_cyc     10000 \
    --period_seconds        0.915




#--------- For running directly from commandline use below
#python compute_Spectrogram.py \
#    --case_name             "PTSeg028_base_0p64" \
#    --input_folder          "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step2_PostProcess/Qcriterion/cy3_saveFreq5/" \
#    --mesh_folder           "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step1_CFD/data" \
#    --output_folder         "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step2_PostProcess/Qcriterion/cy3_saveFreq5/specs" \
#    --ROI_center_csv        "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step1_CFD/data/PTSeg028_base_0p64_centerline_points.csv" \
#    --n_process             192 \
#    --spec_quantity         "qcriterion" \
#    --window_length         2732 \
#    --ROI_type              "cylinder" \
#    --ROI_radius            8 \
#    --ROI_height            2 \
#    --ROI_start_center_id   627 \
#    --ROI_end_center_id     664 \
#    --ROI_stride            2 \
#    --flag_multi_ROI        \
#    --timesteps_per_cyc     10000 \
#    --period_seconds        0.915

    #--flag_save_ROI


wait

