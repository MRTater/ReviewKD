#!/bin/bash
#SBATCH --job-name=ImageNetBKDLyy
#SBATCH --mail-user=yyluo9@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept8/fyp21/lj2104/lyy/ReviewKD/log/CIFAR100/R20.log
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h

source activate KD
python3 train.py --model resnet20 --suffix baseline1