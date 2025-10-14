#!/bin/bash
#Anything listed between set -a and set +a can be used as variables in any subsequent commands (this is a workaround for sourcing or exporting every variable)
set -a 
#CFD job script input file
#you need to give your username or this won't work at all
scinet_user=ranbar
# The group allocation to run the job under
group_name=def-steinman
#solver environment name (specific to you)
solver_env_name=oasis
#Whether or not you are using debug node
debug=off
#What your case will be called in the output files & on Niagara -- This should be the same name as this script without the sh
casename="PTSeg028_base_0p64"
#Number of cycles to run
cycles=1 #2 #not printing the first cycle
#waveform period
period=915.0 #default is 915 ms
#Number of timesteps for each cycle 
timesteps_per_cycle=2000 #39060 #default is 2000
#Velocity order
uOrder=1 #default is 1
#How many timesteps before saving
save_frequency=100 #13 #default 5
checkpoint=100
#Amount of time Niagara will need to run the case (max 24 hours)
estimated_required_time="11:59:59"
#Amount of time needed to post-process the case (this is run on a single proc)
post_processing_time_minutes=180
#Number of cores to use per node (everything run on a singe node) 
num_cores=190
#Whether or not to save ftle field. Default is False
#save_ftle=False
set +a



#Run the submission script to Niagara
#"$@" just includes any additional keyword arguments passed when you run this script (eg. the command "./in_default.sh hello" would pass the variable hello to this script, and it would be then passed to the solver.sh file subsequently, and could be accessed with $1)
#./solver-v2_simple.sh "$@"

# Check partitioning
if [ $debug == "on" ]; then 
  partition="debug" #max 1 hour
else
  partition="compute" #max 24 hours
fi

# Create a jobname for SLURM
if [[ ! -z jobname ]]; then
  jobname="${casename}_C${cycles}_ts${timesteps_per_cycle}"
fi

#make some log directories
if [ -d "./logs" ]; then
    echo "logs directory exists." 
else
    echo "Attention: logs directory does not exist. Making one now."
    mkdir ./logs
    mkdir ./hpclog
fi


# Submit the job with defined parameters
sbatch --export=ALL \
       --account=$group_name \
       --time=$estimated_required_time \
       --job-name=$jobname \
       --nodes=1 \
       --ntasks-per-node=$num_cores \
       --partition=$partition \
       ./solver-v3.sh "$@"