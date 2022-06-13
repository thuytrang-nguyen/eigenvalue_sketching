#!/bin/bash
#
#SBATCH --job-name=rand_fewbig2
#SBATCH --mail-type=END,FAIL                    		# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=thuytrangngu@umass.edu      		# Where to send mail 
#SBATCH --output=./outputs/res_%j.txt 					# output file
#SBATCH -e ./outputs/res_%j.err        					# File to which STDERR will be written
#SBATCH --partition=longq         						# Partition to submit to
#SBATCH --mem=32000                     				# Memory required in MB
#SBATCH --cpus-per-task=2                  				# No. of required CPUs
#SBATCH --nodes=2										# No. of compute nodes
#SBATCH --ntasks-per-node=8

echo "SLURM_JOBID: " $SLURM_JOBID
source /ven/bin/activate
echo "Starting to run sample script..."

python3 compare_unified.py
wait

echo "Done running the sample script!"
exit
