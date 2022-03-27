#!/bin/bash
#SBATCH --job-name=ImageNetBKDLyy
#SBATCH --mail-user=yyluo9@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept8/fyp21/lj2104/lyy/ReviewKD/log/BKD.log
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h

source activate KD
python3 imagenet_amp.py \
    -a resnet18 --save_dir output/r18-r34/ \
    -b 64 -j 4 -p 10 \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --teacher resnet34 \
    --review-kd-loss-weight 1.0 \
    /research/dept8/fyp21/lj2104/datasets/ImageNet