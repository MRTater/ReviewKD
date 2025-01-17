#!/bin/bash
#SBATCH --job-name=ImageNetBKDLyy
#SBATCH --mail-user=yyluo9@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept8/fyp21/lj2104/lyy/ReviewKD/log/CIFAR100_imb/R32_R32.log
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p gpu_24h

source activate KD
python3 train_imb.py --model resnet32 --teacher resnet32 \
 --teacher-weight checkpoints/CIFAR100V2_resnet32__baseline1_best.pt \
  --kd-loss-weight 1.0 --suffix reviewkd1 --dataset CIFAR100V2 --imb_factor 0.01
