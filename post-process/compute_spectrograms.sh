#!/bin/bash

#SBATCH -A ctb-steinman
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
##########SBATCH --time=4:59:00
#SBATCH --time=1:00:00
#######SBATCH -p debug
#SBATCH --job-name specs
#SBATCH --output=spec_%j.txt
#SBATCH --mail-type=FAIL

export OMP_NUM_THREADS=10
export MPLCONFIGDIR=/scratch/s/steinman/ranbar/.config/mpl
export PYVISTA_USERDATA_PATH=/scratch/s/steinman/ranbar/.local/share/pyvista
export XDG_RUNTIME_DIR=/scratch/s/steinman/ranbar/.local/temp
export TEMPDIR=$SCRATCH/.local/temp
export TMPDIR=$SCRATCH/.local/temp

module load NiaEnv/2019b intelpython3/2019u5 gnu-parallel
source activate $HOME/../macdo708/.conda/envs/aneurisk_librosa
cd $SLURM_SUBMIT_DIR

#&& mkdir specs && mkdir imgs 
#SPEC_INPUT=$(find "$RESULTS_FOLDER" -mindepth 1 -maxdepth 1 -type d)


(cd $SCRATCH/PT_cases/case_A/case_028_low && python $SCRATCH/Scripts/post_processing/spectrograms/compute_spectrograms_point.py results/art_PTSeg* PTSeg028_low  specs imgs 1 && echo '106_low complete') #&


#(cd $PWD/PTSeg028/case_028_ultraultralow && mkdir specs && mkdir imgs && python ../../compute_spectrograms.py results/art_PTSeg028_* PTSeg028_ultraultralow specs imgs 1 && echo '028_ultraultralow_complete')&

#(cd $PWD/PTSeg028/case_028_ultralow && python ../../compute_spectrograms.py results/art_PTSeg028_ultralow_I1_FC_VENOUS_Q557_Per915_Newt370_ts11760_cy2_uO1 PTSeg028_ultralow specs imgs 1 && echo '028_ultralow_complete')&

#(cd $PWD/PTSeg028/case_028_low && python ../../compute_spectrograms.py results/art_PTSeg028_low_I1_FC_VENOUS_Q557_Per915_Newt370_ts15660_cy2_uO1 PTSeg028_low specs imgs 1 && echo '028_low_complete')&
#(cd $PWD/PTSeg043/case_043_low && python ../../compute_spectrograms.py results/art_PTSeg043_low_I1_FC_VENOUS_Q557_I2_FC_VENOUS_Q161_Per915_Newt370_ts12000_cy2_uO1 PTSeg043_low specs imgs 1 &&echo '043_low complete')&
#(cd $PWD/PTSeg002/case_002_low && python ../../compute_spectrograms.py results/art_PTSeg002_low_I3_FC_VENOUS_Q557_I1_FC_VENOUS_Q135_I2_FC_VENOUS_Q57_Per915_Newt370_ts7440_cy2_uO1 PTSeg002_low specs imgs 1 && echo '002_low complete')&
#(cd $PWD/PTSeg109/case_109_low && mkdir specs && mkdir imgs && python ../../compute_spectrograms.py results/art_PTSeg109_low_I2_FC_VENOUS_Q557_I1_FC_VENOUS_Q103_Per915_Newt370_ts4980_cy2_uO1 PTSeg109_low specs imgs 1 && echo '109_low complete')&

#(cd $PWD/PTSeg028/case_028_med && python ../../compute_spectrograms.py results/art_PTSeg028_med_I1_FC_VENOUS_Q557_Per915_Newt370_ts23340_cy2_uO1 PTSeg028_med specs imgs 1 && echo '028_med complete')&
#(cd $PWD/PTSeg043/case_043_med && python ../../compute_spectrograms.py results/art_PTSeg043_med_I2_FC_VENOUS_Q557_I1_FC_VENOUS_Q161_Per915_Newt370_ts18060_cy2_uO1 PTSeg043_med specs imgs 1 && echo '043_med complete')&
#(cd $PWD/PTSeg002/case_002_med && mkdir specs && mkdir imgs && SPEC_INPUT=$(find 'results' -mindepth 1 -maxdepth 3) && SPEC_INPUT0=$(find '../PTSeg002_cl_mapped_spectrospheres') && parallel -j 10 python ../../compute_spectrograms.py results/art_PTSeg002_med_I1_FC_VENOUS_Q557_I4_FC_VENOUS_Q135_I2_FC_VENOUS_Q57_Per915_Newt370_ts11160_cy2_uO1 PTSeg002_med specs imgs 1 ::: $SPEC_INPUT $SPEC_INPUT0 && echo '002_med complete')&
#(cd $PWD/PTSeg109/case_109_med && mkdir specs && mkdir imgs && SPEC_INPUT=$(find 'results' -mindepth 1 -maxdepth 3) && SPEC_INPUT0=$(find '../PTSeg109_cl_mapped_spectrospheres') && parallel -j 10 python ../../compute_spectrograms.py results/art_PTSeg109_med_I1_FC_VENOUS_Q557_I2_FC_VENOUS_Q103_Per915_Newt370_ts7500_cy2_uO1 PTSeg109_med specs imgs 1 ::: $SPEC_INPUT $SPEC_INPUT0 && echo '109_med complete')&

#(cd $PWD/PTSeg028/case_028_high && python ../../compute_spectrograms.py results/art_PTSeg028_high_I1_FC_VENOUS_Q557_Per915_Newt370_ts28800_cy2_uO1 PTSeg028_high  specs imgs 1 && echo '028_high complete')&
#(cd $PWD/PTSeg043/case_043_high && python ../../compute_spectrograms.py results/art_PTSeg043_high_I1_FC_VENOUS_Q557_I3_FC_VENOUS_Q161_Per915_Newt370_ts22080_cy2_uO1 PTSeg043_high specs imgs 1 && echo '043_high complete')&
#(cd $PWD/PTSeg109/case_109_high && mkdir specs && mkdir imgs && python ../../compute_spectrograms.py results/art_PTSeg109_high_I1_FC_VENOUS_Q557_I2_FC_VENOUS_Q103_Per915_Newt370_ts7500_cy2_uO1 PTSeg109_high specs imgs 1 && echo '109_high complete')&
#(cd $PWD/PTSeg002/case_002_high && mkdir specs && mkdir imgs && python ../../compute_spectrograms.py results/art_PTSeg002_high_I3_FC_VENOUS_Q557_I2_FC_VENOUS_Q135_I1_FC_VENOUS_Q57_Per915_Newt370_ts13260_cy2_uO1 PTSeg002_high specs imgs 1 && echo '002_high complete')&

#(cd $PWD/PTSeg043/case_043_ultraultralow && mkdir specs && mkdir imgs && python ../../compute_spectrograms.py results/art_PTSeg* PTSeg043_ultraultralow specs imgs 1 &&echo '043_ultraultralow complete')&
#(cd $PWD/PTSeg043/case_043_ultralow && python ../../compute_spectrograms.py results/art_PTSeg* PTSeg043_ultralow specs imgs 1 &&echo '043_ultralow complete')&
#(cd $PWD/PTSeg043/case_043_low && python ../../compute_spectrograms.py results/art_PTSeg* PTSeg043_low specs imgs 1 &&echo '043_low complete')&
#(cd $PWD/PTSeg043/case_043_med && mkdir specs && mkdir imgs && python ../../compute_spectrograms.py results/art_PTSeg* PTSeg043_med specs imgs 1 &&echo '043_med complete')&
#(cd $PWD/PTSeg043/case_043_med_ref && mkdir specs && mkdir imgs && python ../../compute_spectrograms.py results/art_PTSeg* PTSeg043_med_ref specs imgs 1 &&echo '043_med_ref complete')&
#(cd $PWD/PTSeg043/case_043_high && python ../../compute_spectrograms.py results/art_PTSeg* PTSeg043_high specs imgs 1 &&echo '043_high complete')&



wait
