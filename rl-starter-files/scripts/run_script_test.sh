# this line tells the scheduler to interpret the rest of the script as a bash script
#$ -S /bin/bash

# set a task increment
#$ -t 1-{}

# amount of memory
#$ -l tmem= 100G

# Limit of time
#$ -l h_rt=00:20:00

# Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=true

# GPU
#$ -pe gpu 2

# reserves requested resources:
#$ -R y

# Set the name of the job.
#$ -N TestEmpty

#activate the virtual environment
#source /home/rmapkay/new-env/bin/activate
# source /shared/ucl/apps/miniconda/4.10.3/etc/profile.d/conda.sh
# conda activate minigrid


# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
#$ -wd cd /cluster/project7/diversity_rl/diversity_study/rl-starter-files

# # Checks which copy of Python is being run
# command -v python3

# # Checks which libraries that version of Python is using
# ldd `command -v python3`


# #read parameter values and run
# paramfile=/home/rmapkay/Scratch/DIAYN_corrected_params.txt
# number=$SGE_TASK_ID

# env="`sed -n ${number}p $paramfile | awk '{print $1}'`"
# model="`sed -n ${number}p $paramfile | awk '{print $2}'`"
# frames="`sed -n ${number}p $paramfile | awk '{print $3}'`"
# algo="`sed -n ${number}p $paramfile | awk '{print $4}'`"


# Run the application
# echo "$algo" "$env" "$folder_name" "$frames" "$entropy_coef" "$ir_coef" "$disc_lr" "$num_skills" "$seed"
# python3 -m scripts.train --algo "$algo" --env "$env" --folder-name "$folder_name" --frames "$frames" --entropy-coef "$entropy_coef" --ir-coef "$ir_coef" --disc-lr "$disc_lr" --num-skills "$num_skills" --seed "$seed"
python3 -m train --algo ppo --env MiniGrid-DoorKey-5x5-v0 --model DoorKey --save-interval 10 --frames 80 --intrinsic-reward-model TrajectoryCount
