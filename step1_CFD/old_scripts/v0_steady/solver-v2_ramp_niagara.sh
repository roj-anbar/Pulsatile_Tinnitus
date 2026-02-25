#!/bin/bash
#///////////////////////////////////////////////////////////////
#// CFD job script wrapper
#// Adapted from 2018 Mehdi Najafi (mnuoft@gmail.com) by Anna Haley 2022 (ahaley@mie.utoronto.ca) 
#///////////////////////////////////////////////////////////////

#Check partitioning
if [ $debug == "on" ]; then 
  partition="debug" #max 1 hour
else
  partition="compute" #max 24 hours
fi

#create a jobname for SLURM
if [[ ! -z jobname ]]; then
  jobname="${casename}_C${cycles}_TS${timesteps_per_cycle}"
fi

#make some log directories
if [ -d "./logs" ]; then
    echo "logs directory exists." 
else
    echo "Error: logs directory does not exist. Making one now."
    mkdir ./logs
    mkdir ./hpclog
fi

#you might want to force the job, or you might want to clean up the case file, so you would just add the keywords force or clean to your argument list when you execute your input file (eg. in_default.sh)
force="No";
clean="No";
restart_no_given=-1
for var in "$@"; do
  if [ "$var" == "force" ]; then force="Yes"; fi
  if [ "$var" == "clean" ]; then clean="Yes"; fi
  #if you send an argument like "5" to the executable input file, it will interpret that as restart #5
  if [ $var -eq $var 2> /dev/null ]; then restart_no_given=$var; fi
done

export LD_LIBRARY_PATH=/home/s/steinman/$scinet_user/.conda/envs/$solver_env_name/lib:/scinet/niagara/software/2020a/opt/intel-2020u1-intelmpi-2020u1/boost/1.69.0/lib:$LD_LIBRARY_PATH
 
#submit everything that follows until "EOST" is read at the end of this file
sbatch --time=$estimated_required_time --job-name=$jobname --nodes=1 --ntasks-per-node=$num_cores --ntasks-per-node=$num_cores --partition=$partition  << EOST
#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=hpclog/art_%x_stdout_%j.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

### ---------- ADDED BY Rojin A. to avoid JIT compilation errors ----------- ###
export DIJITSO_CACHE_DIR=/scratch/s/steinman/ranbar/PT/.cache  

mkdir -p "$DIJITSO_CACHE_DIR"


#this is the location of the solver in your files
SOLVER_HOME=/home/s/steinman/$scinet_user/BSLSolver

#provided the environment was set as per the readme, the following just loads the correct modules and activates the conda environment
module load NiaEnv/.2020a intel/2020u1 intelmpi/2020u1 intelpython3/2020u1 cmake/3.16.3 boost/1.69.0 eigen/3.3.7 hdf5/1.8.21 netcdf/4.6.3 gmp/6.2.0 mpfr/4.0.2 swig/4.0.1 petsc/3.10.5 trilinos/12.12.1 fenics/2019.1.0
source activate /home/s/steinman/$scinet_user/.conda/envs/$solver_env_name/

##############################################
#ensure consistency between file names
##############################################
#this is where the naming.py script is run. all it does is creates a naming structure that will be standard and used throughout, including in the post-processing. then we just set a variable to use as a shorter of the casename
casename_full=\$(bslsolver naming $casename $uOrder $cycles $timesteps_per_cycle)
results_folder=./results/\$casename_full
echo "Finished naming"
##############################################
#check for restarts and set log file name
##############################################
#set a variable that gives the number of saves per cycle
uco=$(( $timesteps_per_cycle / $save_frequency))
#acs is the total number of saves we are expecting to see, which includes all the cycles we wanted to print
acs=$(( ($timesteps_per_cycle / $save_frequency)*($cycles-1)))

#check if we want to clean up old stuff
if [ $clean == "Yes" ]; then
  echo "(!)  Removing any previous results and logs."
  rm -r \$results_folder 2>/dev/null
  rm  ./logs/\${casename_full}_restart_* 2>/dev/null
fi

#now we need to check and see if we are restarting or running for the first time
simdone="0"
restart_no=0 #Note that after the first restart, the restart number will always be 1 because we are using Oasis and not Mehdi's io.py
if [ $restart_no_given -lt 0 ]; then
  echo "No restart number given!"
  # determines what the last checkpoint folder was if there was no provided number
  for l in \$(ls -1 \$results_folder/data/ 2>/dev/null); do if [[ \$l -gt \$restart_no ]]; then restart_no=\$l; fi; done
  #counts the number of saved tsteps already accounted for in the folder and saves as uc variable
  uc=\$(ls -1 \$results_folder/*_up.h5 2>/dev/null | wc -l)
  #checks to see if the number of saves (uc) is the same as the number of expected saves per cycle (uco). I changed this - AH.
  #OLD: if [ \$uco -eq \$uc ]; then
  if [ \$acs -eq \$uc ]; then 
    simdone="1"
  fi
else
  restart_no=$restart_no_given
fi

#variable for the current log file
log_file=./logs/\${casename_full}_restart_\${restart_no}

##############################################
#begin the actual simulation using Oasis
##############################################
echo "Aneurysm CFD v.2.0 BioMedical Simulation Lab - University of Toronto"
echo "Case Name: " $casename
echo "Number of Cycles: " $cycles
echo "Number of Time Steps per Cycle: " $timesteps_per_cycle
echo "Number of Time Steps to skip to output: " $save_frequency
echo "Case Full Name: " \$casename_full
echo "Results folder: " \$results_folder
echo "Log File: " \$log_file
echo "Restart Number: " \$(( \$restart_no+1 ))
#restart_folder=\${results_folder}/data/\${restart_no}/Checkpoint #this is not accurate for new oasis restart_folder=\$restart_folder
#mpirun -n $num_cores oasis NSfracStep problem=Artery_ramp uOrder=$uOrder timesteps=$timesteps_per_cycle period=$period  cycles=$cycles save_frequency=$save_frequency mesh_name=$casename &>> \$log_file

# Only run the solver if the simulation is not already finished
if [ \$simdone == "0" ]; then

  # Case A: Restarting from a previous checkpoint
  if [ \$restart_no -gt 0 ]; then
    if [ -f \$results_folder/data/\$restart_no/incomplete ] || [ $force == "Yes" ]; then
      echo "Restart #: " \$(( \$restart_no+1 ))
      echo "Log file: " \$log_file
      restart_folder=\${results_folder}/data/\${restart_no}/Checkpoint
      echo "restart_folder: " \$restart_folder
      echo \$(date) > \$log_file
      mpirun -n $num_cores oasis NSfracStep problem=Artery_ramp uOrder=$uOrder timesteps=$timesteps_per_cycle period=$period cycles=$cycles save_frequency=$save_frequency \
             mesh_name=$casename restart_folder=\$restart_folder checkpoint=$checkpoint &>> \$log_file
    else
      echo "<!> Something went wrong! You should inspect what happened at Checkpoint#\${restart_no} in order to avoid an infinite loop at this point."
      echo "Here it is: \$restart_folder"
      echo "At the previous restart the solver crashed!"
      exit
    fi

  # Case B: Fresh run (no prior checkpoint)
  else
    echo "Restart #: " \$(( \$restart_no+1 ))
    echo "Log file: " \$log_file
    echo \$(date) > \$log_file
    mpirun -n $num_cores oasis NSfracStep problem=Artery_ramp uOrder=$uOrder timesteps=$timesteps_per_cycle period=$period cycles=$cycles save_frequency=$save_frequency \
           mesh_name=$casename checkpoint=$checkpoint &>> \$log_file
  fi
  echo \$(date) >> \$log_file
  echo "hpclog/art_\${SLURM_JOB_NAME}_stdout_\${SLURM_JOB_ID}.txt" >> \$log_file
  sleep 30
fi

# Auto-resubmit block: This will just run this shell script again in case the solver did not finish.
if [ -f \$log_file ]; then
  if [ -f \$results_folder/data/\$restart_no/incomplete ]; then
    echo "Resubmitting to resume CFD simulation: #" \$(( \$restart_no+1 ))
    run_name=(${casename}.sh)
    out=\$(ssh nia-login03 "cd \$SLURM_SUBMIT_DIR; bash "\${run_name[@]}"")
    echo "\$out"
    echo ""\${run_name[@]}" resubmitted"
  fi
fi

# This is where the postprocessing script (wss hemodynamics) is run:
if [ -f \$results_folder/data/\$restart_no/complete ]; then
    # simulation is done
    echo "Simulation is finished."
    echo "Sleeping for 15 seconds to let the I/O settle down."
    sleep 15
    if [ ! -f \$results_folder/*_hemodynamics_w.tec ] || [ "$force" == "Yes" ]; then
        echo "Submitting hemodynamic calculations job for:" $casename " stored at " \$results_folder
        out=\$((ssh nia-login01 "cd \$SLURM_SUBMIT_DIR; python \$SOLVER_HOME/BSLSolver/Post/hemodynamic_indices_run_me.py \$results_folder -t $post_processing_time_minutes") 2>&1)
        echo "\$out"
    else
        echo "Hemodynamics file were previously generated. Nothing to do!"
    fi
fi

# This is where the postprocessing scripts (wss hemodynamics) is run:
if [ -f \$results_folder/data/\$restart_no/complete ]; then
    # simulation is done
    echo "Simulation is finished."
    echo "Sleeping for 15 seconds to let the I/O settle down."
    sleep 15
    uc=\$(ls -1 \$results_folder/*_up.h5 2>/dev/null | wc -l)
    if [ \$acs -ne \$uc ]; then
      echo "<!> No enough outputs found! Expecting " \$acs "files but found " \$uc " files!"
      echo "What to do: inspect everything first. ONLY, if there was an IO issue:"
      echo "1) Try removing file:" \$results_folder/data/\$restart_no/complete
      echo "2) Re-run this script: bash \$0"
    else
      echo "Submitting post-processing jobs for:" \$casename " stored at " \$results_folder
      wc=\$(ls -1 \$results_folder/wss_files/*_wss.h5 2>/dev/null | wc -l)
      if [ \$acs -ne \$wc ]; then
        echo "Submitting wss calculations job for:" \$casename " stored at " \$results_folder
        out=\$((ssh nia-login01 "cd \$SLURM_SUBMIT_DIR; python \$SOLVER_HOME/BSLSolver/Post/wss_run_me.py \$results_folder -t $post_processing_time_minutes")  2>&1)
        echo "\$out"
        wss_jid="\${out#*Submitted batch job}";wss_jid=\${wss_jid##* };
        # echo "<\$wss_jid>  $0 $1 \$SLURM_SUBMIT_DIR"
        if [ ! -z \$wss_jid ]; then
          #sleep 5
          out=\$((ssh nia-login01 "cd \$SLURM_SUBMIT_DIR; printf '#!/bin/bash\nssh nia-login01 \"cd \$SLURM_SUBMIT_DIR; bash $0\"' | sbatch --dependency=afterany:\$wss_jid --time=00:15:00 --mail-type=NONE --nodes=1 --ntasks-per-node=1 --job-name ${jobname}_1 --output=hpclog/art_%x_pstdout_%j.txt") 2>&1)
          echo "\$out"
        fi
      else
        if [ ! -f \$results_folder/*_hemodynamics_w.tec ] || [ "$force" == "Yes" ]; then
          echo "Submitting hemodynamic calculations job for:" $casename " stored at " \$results_folder
          out=\$((ssh nia-login01 "cd \$SLURM_SUBMIT_DIR; python \$SOLVER_HOME/BSLSolver/Post/hemodynamic_indices_run_me.py \$results_folder -t $post_processing_time_minutes") 2>&1)
          echo "\$out"
        else
          echo "Hemodynamics file were previously generated. Nothing to do!"
        fi
      fi
    fi
fi

EOST
