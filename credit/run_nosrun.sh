#!/bin/bash


# Anaconda
#export PYTHONPATH=
#module use /package/x86_64/modulefiles
#module load miniforge3

# Python
#conda activate /nwpr/ait/p088/miniforge3/envs/physicnemomain
#conda activate /users/xb80/.conda/envs/corrdiff
#export PYTHONPATH=$PYTHONPATH:/users/xb80/source/modulus/build/lib:/users/xb80/.local/lib/python3.10/site-packages

# OpenMPI
#module use /package/x86_64/nvidia/hpc_sdk/modulefiles
#module load nvhpc-hpcx-cuda12/24.11



#torchrun \
#	--standalone \
#	--nnodes=1 \
#	--nproc-per-node=8 \
#	applications/train_cwa.py -c ../config/wxformer_6hr_cwa.yml

export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_ALGO=ring
config='../config/wxformer_6hr_cwa.yml'
TORCHRUN=$(which torchrun)
${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=8 \
	applications/train_cwa.py -c ${config} 
