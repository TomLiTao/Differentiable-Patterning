#! /bin/sh
#$ -N gpu_test
#$ -cwd
#$ -l h_rt=1:00:00
#$ -l rl9=true

#$ -pe gpu-a100 1
#$ -l h_vmem=80G


. /etc/profile.d/modules.sh

module load anaconda
module load cuda
source activate jax_gpu
python ./jax_eddie_test.py
source deactivate