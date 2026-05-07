#!/bin/bash
#SBATCH --job-name=wxformer               # Job name
#SBATCH --partition=normal                 # GPU partition
#SBATCH --nodes=4                          # Number of nodes
#SBATCH --ntasks-per-node=1                # MPI tasks per node
#SBATCH --cpus-per-task=28                 # Number of CPU cores per task
#SBATCH --gres=gpu:8                       # Number of GPUs per node
#SBATCH --time=720:00:00                    # Runtime (hh:mm:ss)
#SBATCH --output=/nwpr/ait/p088/work/creditlog/%x_%j.out     # 佐~H併 stdout 佒~L stderr 佈° job name + job id
#SBATCH --error=/nwpr/ait/p088/work/creditlog/%x_%j.err

#set -x

export NODES=4
export PPN=1
export GPU_PER_NODE=8
export OMP_NUM_THREADS=2


#Anaconda
export PYTHONPATH=
source ~/.bashmini
conda activate credit
#module use /package/x86_64/modulefiles
#module load miniforge3

# Python
#conda activate /nwpr/ait/p088/miniforge3/envs/physicnemomain
#conda activate /users/xb80/.conda/envs/corrdiff
#export PYTHONPATH=$PYTHONPATH:/users/xb80/source/modulus/build/lib:/users/xb80/.local/lib/python3.10/site-packages

# OpenMPI
module use /package/x86_64/nvidia/hpc_sdk/modulefiles
module load nvhpc-hpcx-cuda12/24.11



#torchrun \
#	--standalone \
#	--nnodes=1 \
#	--nproc-per-node=8 \
#	applications/train_cwa.py -c ../config/wxformer_6hr_cwa.yml

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($(nvidia-smi -L | wc -l) - 1)))
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_ALGO=ring
export NCCL_SOCKET_FAMILY=AF_INET
MASTER_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=29511
echo "Master Node: $MASTER_NODE"
echo "Running on Nodes: $SLURM_NODELIST"

config="${SLURM_SUBMIT_DIR:-$(pwd)}/../config/wxformer_6hr_cwa_30y.yml"
APP="${SLURM_SUBMIT_DIR:-$(pwd)}/applications/train_cwa.py -c ${config}"
TORCHRUN=$(which torchrun)

echo GO `date +%Y%m%d%H`
echo GO `date +%Y%m%d%H`
st=`date +%s`

srun --mpi=pmix -N ${NODES} \
        ${TORCHRUN} --nnodes=${NODES} --nproc_per_node=${GPU_PER_NODE} \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_NODE}:${MASTER_PORT} \
        ${APP}

echo Done `date +%Y%m%d%H`
echo Done `date +%Y%m%d%H`
diff=$((`date +%s` - ${st}))
cost=$(echo "scale=5; ${diff}/3600" | bc)

echo '***********************************'
echo Training time is ${cost} hours
echo Training time is ${cost} hours
echo Training time is ${cost} hours
echo '***********************************'

#config='../config/wxformer_6hr_cwa.yml'
#TORCHRUN=$(which torchrun)
#${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=8 \
#	applications/train_cwa.py -c ${config} 
