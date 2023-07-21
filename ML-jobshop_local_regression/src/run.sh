#!/bin/bash
#SBATCH --job-name="JSSP"
#SBATCH --constraint=Xeon-Gold-6150
#SBATCH --partition=comp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-2:00:00
#SBATCH --array=11-19
##SBATCH --array=[20,30,40,50,60,70,80,90,100]
##SBATCH --mem-per-cpu=4G

module load python/3.8.5

#python3 testing_regression.py -ml_method MLP_128 -time_limit 3600 -obj twt -nmachine ${SLURM_ARRAY_TASK_ID} -njob ${SLURM_ARRAY_TASK_ID}

python3 testing_regression.py -ml_method MLP_128 -time_limit 60 -obj twt -nmachine ${SLURM_ARRAY_TASK_ID} -njob ${SLURM_ARRAY_TASK_ID}
