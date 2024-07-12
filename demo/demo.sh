#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N NCA_DEMO              
#$ -cwd
#$ -e ../Eddie_OP
#$ -o ../Eddie_OP                  
#$ -l h_rt=00:05:00 
#$ -l h_vmem=4G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Anaconda
module load anaconda/2024.02

#Activate the environment
source activate jax

# Run the program
python ./train_nca_to_pde_demo.py
conda deactivate
