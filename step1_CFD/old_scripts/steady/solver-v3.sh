#!/bin/bash

#SBATCH --output=hpclog/%x_%j.txt
#SBATCH --export=ALL


# Navigate to the script directory
cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1


# This is the location of the BSLSolver in your directory
BSLSOLVER_HOME="/home/$scinet_user/BSLSolver"


### ---------- ADDED BY Rojin A. to avoid JIT compilation errors ----------- ###
###############################################################################################
# Setup container variables
###############################################################################################

# Slurm tasks (fallback to num_cores if exported, else 1)
NP="${SLURM_NTASKS:-1}"


PATH_UFL_SHIM="/scratch/$scinet_user/pyshims"
PATH_BSLSOLVER="$BSLSOLVER_HOME"
PATH_PYTHON="${PATH_UFL_SHIM}:${PATH_BSLSOLVER}"

PATH_JIT_CACHE="/scratch/$scinet_user/PT/PT_Ramp/steady/.cache"
mkdir -p "$PATH_JIT_CACHE"

export APPTAINERENV_PYTHONPATH="$PATH_PYTHON${PYTHONPATH:+:$PYTHONPATH}"
export APPTAINERENV_DIJITSO_CACHE_DIR="$PATH_JIT_CACHE"



# Bind what you need; add more binds if your data/code live elsewhere
BIND_OPTS="--bind /scratch:/scratch --bind $SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR --pwd $SLURM_SUBMIT_DIR"


#echo "PY_PATH= $PATH_PYTHON"
#echo "BIND_OPTS= $BIND_OPTS"
#echo "SLURM_SUBMIT_DIR= $SLURM_SUBMIT_DIR"
echo "Number of cores= $NP"

###############################################################################################
# Ensure consistency between file names
###############################################################################################
#this is where the naming.py script is run. all it does is creates a naming structure that will be standard and used throughout, including in the post-processing. then we just set a variable to use as a shorter of the casename
casename_full="PTSeg028_base_ts10000"

results_folder="./results/$casename_full"
echo "Finished naming"


###############################################################################################
# Check for restarts and set log file name
###############################################################################################

# Now we need to check and see if we are restarting or running for the first time
simdone="0"
restart_no=0 #Note that after the first restart, the restart number will always be 1 because we are using Oasis and not Mehdi's io.py

# Variable for the current log file
log_file="./logs/${casename_full}_${SLURM_JOB_ID}"


###############################################################################################
# Beginning of the actual simulation using Oasis
###############################################################################################

echo "Pulsatile Tinnitus CFD v.2.0 BioMedical Simulation Lab - University of Toronto"
echo "Case Name: " $casename
echo "Number of Cycles: " $cycles
echo "Number of Time Steps per Cycle: " $timesteps_per_cycle
echo "Number of Time Steps to skip to output: " $save_frequency
echo "Case Full Name: $casename_full"
echo "Results folder: $results_folder"
echo "Log File: $log_file"
#echo "Restart Number: " \$(( \$restart_no+1 ))


#echo \$(date) > \$log_file

#APPTAINERENV_PYTHONPATH="$PATH_PYTHON${PYTHONPATH:+:$PYTHONPATH}" \
apptainer exec \
  --env HYDRA_LAUNCHER=fork \
  $BIND_OPTS ~/fenics-legacy-updated2.sif \
  mpirun -n 190 oasis NSfracStep problem=Artery_ramp \
  uOrder=$uOrder \
  timesteps=$timesteps_per_cycle \
  period=$period \
  cycles=$cycles\
  save_frequency=$save_frequency \
  mesh_name=$casename \
  checkpoint=$checkpoint &>> $log_file

echo $(date) >> $log_file
echo "hpclog/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.txt" >> "$log_file"
sleep 30

