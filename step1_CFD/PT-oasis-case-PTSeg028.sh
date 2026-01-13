#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# PT-oasis-case-casename.sh
# Case-specific launcher for Oasis CFD jobs on SLURM (Trillium style clusters).
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2025-09
#
# PURPOSE:
#   - Define all case parameters for CFD in one place and submit oasis-solver.sh via sbatch.
#   - Optional flags let you override key settings without editing the file.
#
# REQUIREMENTS:
#   - oasis-solver.sh (the job script this wrapper submits)
#
# EXECUTION:
#   - Run this script from terminal by:
#     <./PT-oasis-case-casename.sh>
#
# NOTE:
#   - This script should be ran from the PT case directory containing the mesh data (under /data folder).
#   - "PT-oasis-solver.sh" and "PT-oasis-problem.py" should be copied as well to the same directory.
#
# Adapted from solver-v2.sh written by 2022 Anna Haley (ahaley@mie.utoronto.ca) and solver.sh written by 2018 Mehdi Najafi (mnuoft@gmail.com). 
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

set -euo pipefail


#--------------------------------------- Input parameters -------------------------------------------------
# Define all the input parameters to run the CFD solver Oasis job script.

# Anything listed between set -a and set +a can be used as variables in any subsequent commands.
# This is used as an alternative to exporting each variable separately.
set -a 

scinet_user=ranbar                # your cluster username (need to specify this or this won't work at all)
group_name=def-steinman           # group allocation to run the job under
debug=off                          # job partition -> choose between 'on'/'off' (Whether or not you are using debug node)
num_cores=100                     # number of cores to use per node (everything runs on a single node) 
required_time="10:59:59"          # amount of time cluster will need to run the case (max 24 hours)
post_processing_time_minutes=180  # amount of time needed to post-process the case (this is run on a single proc)

casename="PTSeg028_base_0p64"     # What your case will be called in the output files & on cluster -- should be the same name as this script without the sh
cycles=6                          # number of cycles to run, determines total simulation time (default: 2)
period=915.0                      # waveform period [ms] (default: 915 ms)
timesteps_per_cycle=10000         # number of timesteps for each cycle (default: 2000)
viscosity=0.0035                  # kinematic viscosity [mm^2/ms] (≡ m^2/s in consistent units) (default: 0.0035 mm^2/ms)
uOrder=1                          # velocity FE order (default: 1)

save_frequency=5                  # write solution every N steps (default: 5)
checkpoint=500                    # write restart every N steps
save_first_cycle=True             # flag to save first cycle or not (default: False)
#solver_env_name=oasis            # solver environment name (specific to you)

set +a


#------------------------------------ Set parameters ----------------------------------------

# Assign the partition on cluster
if [ $debug == "on" ]; then 
  partition="debug"   # max 1 hour
else
  partition="compute" # max 24 hours
fi


# Create a jobname for SLURM
if [[ ! -z jobname ]]; then
  jobname="${casename}_C${cycles}_ts${timesteps_per_cycle}"
fi

# Create log directories
mkdir -p ./logs ./hpclog


#------------------------------------------- Submit the job ----------------------------------------------------
# All config is exported via environment (sbatch --export=ALL).
# oasis-solver.sh should read these variables (e.g., $cycles, $save_frequency, …).

# Submit the job with defined parameters
sbatch --export=ALL \
       --account=$group_name \
       --time=$required_time \
       --job-name=$jobname \
       --nodes=1 \
       --ntasks-per-node=$num_cores \
       --partition=$partition \
       ./PT-oasis-solver.sh "$@"

# Note: <./oasis-solver.sh "$@">
# "$@" just includes any additional keyword arguments passed when you run this script.
# eg. the command "./in_default.sh hello" would pass the variable hello to this script,
# and it would be then passed to solver.sh file subsequently, and could be accessed with $1.
