#! /bin/sh
#$ -N image_kaNCA_demo
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 1 -l h_vmem=200G



bash kaNCA_demo.sh $SGE_TASK_ID