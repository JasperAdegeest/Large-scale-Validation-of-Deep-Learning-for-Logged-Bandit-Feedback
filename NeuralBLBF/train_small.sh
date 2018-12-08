#!/bin/bash

#Specification of the job requirements for the batch system (number of nodes, expected runtime, etc)
#SBATCH --job-name=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1


module purge
module load eb

module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176

DATA_TRAIN="$HOME/IR2/data/vw_compressed_train"
DATA_TEST="$HOME/IR2/data/vw_compressed_validate"
EPOCHS=10
LAMBDA=1
EMBD_DIM=64
MODEL_TYPE="SmallEmbedFFNN"

python3 -m NeuralBLBF --device_id 0 --train $DATA_TRAIN --test $DATA_TEST --epochs $EPOCHS \
    --stop_idx 100000000 --step_size 20000000 --batch_size 128 --enable_cuda --embedding_dim $EMBD_DIM \
    --model $MODEL_TYPE --save --lamb $LAMBDA
