#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# paraview_parallel.sh
# SLURM job wrapper to run visualization scripts using Paraview in parallel.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2025-09
#
# PURPOSE:
#   - 
#
# REQUIREMENTS:
#   - paraview/6.0.0 (with MPI capabilities)
#
# EXECUTION:
#   - sbatch paraview_parallel.sh
#
#
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

#SBATCH --account=def-steinman
#SBATCH --job-name=paraview
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Requesting resources
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --time=23:59:00

# Export all environment vars
#SBATCH --export=ALL

# Make sure log directory exists
cd $SLURM_SUBMIT_DIR
mkdir -p ./logs

export XDG_CONFIG_HOME="/scratch/$USER/.config/Paraview"
mkdir -p $XDG_CONFIG_HOME

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 paraview/6.0.0

# Process the scripts in parallel
#srun pvbatch --force-offscreen-rendering --opengl-window-backend OSMesa paraview_velocity_isosurface.py
mpirun -np 100 pvbatch --force-offscreen-rendering --opengl-window-backend OSMesa paraview_Qcriterion.py