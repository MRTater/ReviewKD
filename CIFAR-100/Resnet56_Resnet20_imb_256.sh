#!/bin/bash
#SBATCH --job-name=ImageNetBKDLyy
#SBATCH --mail-user=yyluo9@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept8/fyp21/lj2104/lyy/ReviewKD/log/CIFAR100_imb_256/R56_R20.log
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h

source activate KD
python3 train_imb.py --model resnet20 --teacher resnet56 \
 --teacher-weight checkpoints/CIFAR100V2_resnet56__baseline1_256_best.pt \
 --kd-loss-weight 0.6 --suffix reviewkd1 --dataset CIFAR100V2 --imb_factor 0.01 --batch_size 256