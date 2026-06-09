#!/bin/bash

#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --time=00:15:59
#SBATCH --job-name tarring
#SBATCH --output=tarring_%j.txt
#SBATCH --mail-type=FAIL

# IMPORTANT NOTE:
# All the untarring should happen from tarred files directly on Anna's directory!!
# You have to run this on a compute node (don't use login/debug nodes)
# The job might not show as completed but the files have been extracted.


#-----To see the content of the tar file without extracting
#tar -tf archive.tar

#------To untar a tarred case
#tar --use-compress-program=pigz -xvf /project/rrg-steinman-ab/ahaleyyy/DPQ/cases/tarred_cases/case_A/PTSeg028_base_0p512.tar.gz -C $PROJECT/BSL_cases/PT_cases/AnnaH_cases/Study2_DPQ
#echo "Completed untarring case"


#------To untar a specific folder/file from the tarred case
# Note: To obtain <"relative/path/to/folder"> use the below command to list the content of the tarred file:
# tar --use-compress-program=pigz -tf case_043_low.tar.gz | head -50

#tar --use-compress-program=pigz -xvf <tarred_file.tar.gz> <"relative/path/to/folder"> -C $PROJECT/<path_to_untar>

tar --use-compress-program=pigz -xvf \
/project/rrg-steinman-ab/ahaleyyy/DPQ/cases/tarred_cases/case_A/PTSeg028_base_0p512.tar.gz \
"PTSeg028_base_0p512/mesh/" \
-C $PROJECT/BSL_cases/PT_cases/AnnaH_cases/Study2_DPQ/case_A

echo "Completed untarring 1 ..."

#tar -xvzf \
#/project/rrg-steinman-ab/ahaleyyy/DPQ/cases/tarred_cases/case_A/PTSeg028_base_0p512.tar.gz \
#"PTSeg028_base_0p512/data/"
#-C $SCRATCH

#echo "Completed untarring 2 ..."