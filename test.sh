#!/bin/bash
#PJM -N wxformer6hr
#PJM -L rscunit=rscunit_pg01    # Resource unit name
#PJM -L rscgrp=gpu-rd-large       # Resource group name
#PJM -L vnode=1                 # number of Nodes
#PJM -L vnode-core=32           # CPU cores
#PJM --mpi proc=32              # MPI processes
#PJM -L gpu=8                   # GPU Cards
#PJM -L elapse=240:00:00          # elapse time
#PJM -o '../log/run_1Node8GPU.%j.out'
#PJM -e '../log/run_1Node8GPU.%j.err'

source /nwpr/ait/p088/.bashforge
export PYTHONPATH=

conda activate credit

. /usr/share/Modules/init/profile.sh
module use /package/x86_64/nvidia/hpc_sdk/modulefiles
module load nvhpc-hpcx-cuda12/24.11

#gccpath=/users/xb80/opt/x86_64/gcc-13.3.0
#export PATH=${gccpath}/bin:$PATH
#export LD_LIBRARY_PATH=${gccpath}/lib64:${gccpath}/lib:$LD_LIBRARY_PATH

nvcc --version
nvidia-smi
which python

export GPU_PER_NODE=8
#export OMP_NUM_THREADS=2
GPUS=`nvidia-smi -L | wc -l`

export TORCHRUN=$(which torchrun)

#python -u train.py --config-name=${configname}
export TORCHRUN=$(which torchrun)

config="${PJM_SUBMIT_DIR:-$(pwd)}/../creditwks/config/diffusion_test.yml"
APP="${PJM_SUBMIT_DIR:-$(pwd)}/../creditwks/applications/train.py -c ${config}"

${TORCHRUN} --standalone --nproc_per_node=${GPU_PER_NODE} ${APP}
