#!/bin/bash

########SBATCH -A ctb-steinman
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --time=01:00:00
#SBATCH --job-name PT_SPI
#SBATCH --output=PT_SPI_%j.txt

export MPLCONFIGDIR=$SCRATCH/.config/mpl
export PYVISTA_USERDATA_PATH=$SCRATCH/.local/share/pyvista
export XDG_RUNTIME_DIR=$SCRATCH/.local/temp
export TEMPDIR=$SCRATCH/.local/temp
export TMPDIR=$SCRATCH/.local/temp
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=true
export OMP_NUM_THREADS=5

module load StdEnv/2023 gcc/12.3 python/3.12.4
source $HOME/virtual_envs/pyvista36/bin/activate
module load vtk/9.3.0

#(cd $SCRATCH/Swirl/swirl_files/Groccia && mpirun -np 5 python $PROJECT/Swirl/scripts/make_swirl_figs.py . $PROJECT/Swirl/swirl_cases/Groccia_refined_0p64/data Groccia_refined_0p64  ../fig_data ../figs 100 swirl)&

#Don't have R data for these cases:
#(cd $SCRATCH/Swirl/swirl_files/PerturbNewt600 && mpirun -np 5 python $PROJECT/Swirl/scripts/make_swirl_figs.py . $PROJECT/Swirl/swirl_cases/PerturbNewt600/data PerturbNewt600  ../fig_data ../figs 100 R mono)&

#(cd $SCRATCH/Swirl/OL_files/PTSeg043_base_0p64 && mpirun -np 5 python $PROJECT/Swirl/scripts/make_swirl_figs.py . $PROJECT/Swirl/swirl_cases/PTSeg043_base_0p64/data PTSeg043_base_0p64  ../../swirl_files/fig_data ../../swirl_files/figs 100 Omega)&

#(cd $PROJECT/Swirl/swirl_files/case_A/case_028_low_orig && mpirun -np 1 python $PROJECT/Swirl/scripts/make_swirl_figs_TKE.py . $PROJECT/mesh_rez/data/cases/case_A/case_028_low/data/ Case_028_low_orig  $SCRATCH/Swirl/swirl_files/fig_data $SCRATCH/Swirl/swirl_files/figs 100 swirl)
#(cd $SCRATCH/Swirl/swirl_files/PTSeg043 && mpirun -np 1 python $PROJECT/Swirl/scripts/make_swirl_figs_var.py . $PROJECT/Swirl/swirl_cases/PTSeg043_base_0p64/data/ PTSeg043_base_0p64  ../fig_data ../figs 100 swirl)
#(cd $SCRATCH/Swirl/swirl_files/PTSeg043 && mpirun -np 1 python $PROJECT/Swirl/scripts/make_swirl_figs_TKE.py . $PROJECT/Swirl/swirl_cases/PTSeg043_base_0p64/data/ PTSeg043_base_0p64  ../fig_data ../figs 100 swirl)
#python $PROJECT/Swirl/scripts/make_grid_KE.py PTSeg043_base_0p64  fig_data figs

#(cd $SCRATCH/Swirl/swirl_files/PTSeg043 && python $PROJECT/Swirl/scripts/filter_velocity_swirl.py . $PROJECT/Swirl/swirl_cases/PTSeg043_base_0p64/data/ PTSeg043_base_0p64  ../fig_data ../figs)


python compute-post-metrics_SPI.py \
    --input_folder $SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/results/PTSeg028_base_0p64_ts10000_cy6_Q=2t_saveFreq1 \
    --mesh_folder $SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/data/ \
    --case_name PTSeg028_base_0p64 \
    --output_folder $SCRATCH/PT/PT_Ramp/PT_cases/PTSeg028_base_0p64/post-process/SPI_pressure


wait