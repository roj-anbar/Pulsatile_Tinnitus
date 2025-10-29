#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --time=2:59:00
#SBATCH --job-name tarring
#SBATCH --output=tarring_%j.txt
#SBATCH --mail-type=FAIL

# IMPORTANT NOTE:
# All the untarring should happen from tarred files directly on Anna's directory!!

# To untar a tarred case
tar --use-compress-program=pigz -xvf /project/rrg-steinman-ab/ahaleyyy/mesh_rez/data/cases/tarred_cases/case_B/case_043_low.tar.gz -C $PROJECT/PT/PT_cases/Anna_mesh_rez_cases/case_B
echo "Completed untarring case_B"


# To untar a specific folder/file from the tarred case
# Note: To obtain <"relative/path/to/folder"> use the below command to list the content of the tarred file:
# tar --use-compress-program=pigz -tf case_043_low.tar.gz | head -150

#tar --use-compress-program=pigz -xvf <tarred_file.tar.gz> <"relative/path/to/folder"> -C $PROJECT/<path_to_untar>
#tar --use-compress-program=pigz -xvf \
#/project/rrg-steinman-ab/ahaleyyy/Swirl/tarred_cases_swirl/PTSeg043_base_0p64.tar.gz \
#"project/rrg-steinman-ab/ahaleyyy/Swirl/swirl_cases/PTSeg043_base_0p64/data/" \
#-C $PROJECT/PT/PT_Swirl/swirl_cases/PTSeg043_base_0p64/