#!/bin/bash

# To ensure limited /home space is not taken up, change directory to /scratch filesystem by running
#	cd /scratch/$USER
# or
#	cd $SCRATCH

# Submit the job to queue by running
#	sbatch nvidia-gpu.sbatch <commmand-line-arguments>

# Check status of job on queue by running
#	squeue -u $USER
# or
#	squeue --me

#SBATCH --nodes=1												# Number of nodes
#SBATCH --ntasks-per-node=1										# Number of tasks per node
#SBATCH --cpus-per-task=1										# Number of compute cores per task
#SBATCH --time=1:00:00											# Max time limit for the job
#SBATCH --mem=2GB												# Memory allocated for the job
#SBATCH --job-name=nvidia-gpu
#SBATCH --mail-type=ALL											# E-mail alerts sent whenever jobs begin; end; fail; have unsatisfied dependencies; are re-queued; or once a job has been completed/cancelled (stage-out) and the job's resources have been released (teardown)
#SBATCH --mail-user=bc2611@nyu.edu
#SBATCH --gres=gpu:a100:1

# %j is the job ID

#SBATCH --output=slurm_%j--nvidia-gpu.out						# Places standard outpout in ./slurm_<job_id>.out, where . is the directory whence the job is submitted; if --error option is not specified, also places standard error in ./slurm_<job_id>.out
#SBATCH --error=slurm_%j--nvidia-gpu.err						# Places standard error in ./slurm_<job_id>.err


module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo Running:
# Double-quoting $@ ensures that each element of the array is quoted, so that if any argument is originally double-quoted, it preserves this property (useful and potentially necessary for when arguments contain spaces)
# Double-quoting $@ is valid no matter whether there is another character alongside it within the double quotes or not
# -e option enables interpretation of backslash escapes
echo -e "\t$@"
echo
"$@"
