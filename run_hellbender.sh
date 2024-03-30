#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1
#SBATCH --time 2-00:00:00 #Time for the job to run
#SBATCH --job-name mutarget
##SBATCH -p gpu
#SBATCH -p xudong-gpu
#SBATCH -A xudong-lab


module load miniconda3
module load miniconda3

# Activate the Conda environment
#source activate /home/wangdu/.conda/envs/pytorch1.13.0
source activate /home/wangdu/data/env/splm_gvp

export TORCH_HOME=/cluster/pixstor/xudong-lab/duolin/torch_cache/
export HF_HOME=/cluster/pixstor/xudong-lab/duolin/transformers_cache/

#python train.py --config_path ./configs/config_supcon_hardfalse.yaml --result_path ./results_supcon_hardFalse_pft2/b20_p2_n4
#1535976 after finished should change into b10_p2_n4!!!!
#no supcon loss
#python train.py --config_path ./configs/config_supcon_hardtrue.yaml --result_path ./results_supcon_hardTrue_pft2/b10_p2_n4
#1535980 no supcon loss

#before this are all used supcon data sampling and supcon datasample, anchor may from same class in one batch

#python train.py --config_path ./configs/config_supcon_hardtrue.yaml --result_path ./results_supcon_hardTrue_pft2/cleanloader/b8_p3_n9
#1556043
#python train.py --config_path ./configs/config_supcon_hardtrue_esm8.yaml --result_path ./results_supcon_hardTrue_pft2_esm8/cleanloader/b8_p4_n12
#1556054


#python train.py --config_path ./configs/config_supcon_hardtrue_esm8.yaml \
#--predict 1 \
#--resume_path /cluster/pixstor/xudong-lab/duolin/MUTargetCLEAN/results_supcon_hardTrue_pft2_esm8/cleanloader/b8_p4_n12/2024-03-27__00-32-00/checkpoints/best_model.pth \
#--result_path ./results_supcon_hardTrue_pft2_esm8/cleanloader/b8_p4_n12/predict 



#python train.py --config_path ./configs/config_supcon_hardtrue.yaml \
#--predict 1 \
#--resume_path /cluster/pixstor/xudong-lab/duolin/MUTargetCLEAN/results_supcon_hardTrue_pft2/cleanloader/b8_p3_n9/2024-03-27__00-26-05/checkpoints/best_model.pth \
#--result_path ./results_supcon_hardTrue_pft2/cleanloader/b8_p3_n9/predict 

#python train.py --config_path ./configs/config_supcon_hardtrue_esmfix.yaml \
#--result_path ./config_supcon_hardtrue_esmfix/cleanloader/
#1565611 lr small  not work!
#1565887 lr bigger not work!

#using batchsample dataloader not clean but clean's pos and neg, try original codes.
python train.py --config_path ./configs/config_supcon_onlysampling.yaml --result_path ./results_supcon_hardTrue_onlysampling/b10_p2_n4
#1565886 stopped by error
#1566832 modified train.py
