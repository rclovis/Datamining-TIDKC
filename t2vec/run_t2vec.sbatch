#!/bin/bash
#SBATCH --job-name=runt2vec          # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --gpus=1                     # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=14:00:00              # total run time limit (HH:MM:SS)
#SBATCH --partition=normal           # partition(large/normal/cpu) where you submit
#SBATCH --account=msccsit2024        # account code
#SBATCH --chdir=/home/zzhangjf/t2vec # directory to execute


module purge                     # clear environment modules inherited from submission
module load Anaconda3/2023.09-0  # load the exact modules required
source activate t2vec_env        # activate


CUDA_LAUNCH_BLOCKING=1 /home/zzhangjf/.conda/envs/t2vec_env/bin/python t2vec.py -vocab_size 18866 -criterion_name "KLDIV" -knearestvocabs "data/porto-vocab-dist-cell100.h5"
# python t2vec.py -mode 2 -vocab_size 18866 -checkpoint "/home/zzhangjf/t2vec/data/best_model.pt" -prefix "exp1"
