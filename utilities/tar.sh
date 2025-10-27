#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --time=2:59:00
#SBATCH --job-name tarring
#SBATCH --output=tarring_%j.txt
#SBATCH --mail-type=FAIL

# To untar a tarred case
tar --use-compress-program=pigz -xvf /project/rrg-steinman-ab/ahaleyyy/mesh_rez/data/cases/tarred_cases/case_B/case_043_low.tar.gz -C $PROJECT/PT/PT_cases/Anna_mesh_rez_cases/case_B
echo "Completed untarring case_B"
