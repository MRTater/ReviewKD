#!/bin/bash
#SBATCH --job-name=ImageNetBKDLyy
#SBATCH --mail-user=yyluo9@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept8/fyp21/lj2104/lyy/ReviewKD/log/CIFAR100/Baseline_.log
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h

source activate KD
python3 train.py --model resnet20 --teacher resnet56 --teacher-weight checkpoints/cifar100_resnet56__baseline1_best.pt --kd-loss-weight 0.6 --suffix reviewkd1