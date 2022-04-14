#!/bin/bash
#SBATCH --job-name=ImageNetBKDLyy
#SBATCH --mail-user=yyluo9@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept8/fyp21/lj2104/lyy/ReviewKD/CIFAR-100/ensemble_log/R32_KD.log
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p gpu_24h

source activate KD
python3 train_imb.py --model ensemble --suffix baseline1 --dataset CIFAR100V2 --imb_factor 0.01 \
 --ensemble1 ensemble/CIFAR100V2_resnet32_resnet32_reviewkd1_best_1.pt \
 --ensemble2 ensemble/CIFAR100V2_resnet32_resnet32_reviewkd1_best_2.pt \
 --ensemble3 ensemble/CIFAR100V2_resnet32_resnet32_reviewkd1_best_3.pt