# this line tells the scheduler to interpret the rest of the script as a bash script
#$ -S /bin/bash

# set a task increment
#$ -t 1-{}

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

env ="`sed -n ${number}p $paramfile | awk '{print $1}'`"
folder_name="`sed -n ${number}p $paramfile | awk '{print $2}'`"
seed="`sed -n ${number}p $paramfile | awk '{print $3}'`"
frames="`sed -n ${number}p $paramfile | awk '{print $4}'`"
intrinsic_reward_model="`sed -n ${number}p $paramfile | awk '{print $5}'`"
beta_coeff="`sed -n ${number}p $paramfile | awk '{print $6}'`"
no_skills="`sed -n ${number}p $paramfile | awk '{print $7}'`"
window_size="`sed -n ${number}p $paramfile | awk '{print $8}'`"


# Run the application
# echo "$algo" "$env" "$folder_name" "$frames" "$entropy_coef" "$ir_coef" "$disc_lr" "$num_skills" "$seed"
# python3 -m scripts.train --algo "$algo" --env "$env" --folder-name "$folder_name" --frames "$frames" --entropy-coef "$entropy_coef" --ir-coef "$ir_coef" --disc-lr "$disc_lr" --num-skills "$num_skills" --seed "$seed"
# python3 -m scripts.train --algo ppo --env MiniGrid-DoorKey-5x5-v0 --model DoorKey --save-interval 10 --frames 80 --intrinsic-reward-model TrajectoryCount

echo "$env" "$folder_name" "$seed" "$frames" "$intrinsic_reward_model" "$beta_coeff" "$no_skills" "$window_size"
python3 -m scripts.train --algo ppo --env "$env" --model "$folder_name" --seed "$seed" --save-interval 10 --frames "$frames" --intrinsic-reward-model "$intrinsic_reward_model" --intrinsic-coef "$beta_coeff" --number-skills "$no_skills" --window-size "$window_size"