# this line tells the scheduler to interpret the rest of the script as a bash script
#$ -S /bin/bash

# set a task increment
#$ -t 1-80

# amount of memory
#$ -l tmem= 100G

# Limit of time
#$ -l h_rt=12:00:00

# Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=true

# GPU
#$ -pe gpu 2

# reserves requested resources:
#$ -R y

# Set the name of the job.
#$ -N TestTry1

#activate the virtual environment
#source /home/rmapkay/new-env/bin/activate
# source /shared/ucl/apps/miniconda/4.10.3/etc/profile.d/conda.sh
# conda activate minigrid


# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
#$ -cwd

# # Checks which copy of Python is being run
# command -v python3

# # Checks which libraries that version of Python is using
# ldd `command -v python3`


# #read parameter values and run
paramfile=/cluster/project7/diversity_rl/diversity_study/rl-starter-files/params_test.txt
number=$SGE_TASK_ID

env ="`sed -n ${SGE_TASK_ID}'{p;q}' $paramfile | awk '{print $1}'`"
folder_name="`sed -n ${SGE_TASK_ID}'{p;q}' $paramfile | awk '{print $2}'`"
frames="`sed -n ${SGE_TASK_ID}'{p;q}' $paramfile | awk '{print $3}'`"
intrinsic_reward_model="`sed -n ${SGE_TASK_ID}'{p;q}' $paramfile | awk '{print $4}'`"
beta_coeff="`sed -n ${SGE_TASK_ID}'{p;q}' $paramfile | awk '{print $5}'`"
no_skills="`sed -n ${SGE_TASK_ID}'{p;q}' $paramfile | awk '{print $6}'`"
window_size="`sed -n ${SGE_TASK_ID}'{p;q}' $paramfile | awk '{print $7}'`"


echo "$env" "$folder_name" "$frames" "$intrinsic_reward_model" "$beta_coeff" "$no_skills" "$window_size"
python3 -m scripts.train --algo ppo --env "$env" --model "$folder_name" --save-interval 10 --frames "$frames" --intrinsic-reward-model "$intrinsic_reward_model" --intrinsic-coef "$beta_coeff" --number-skills "$no_skills" --window-size "$window_size"