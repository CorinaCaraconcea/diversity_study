# diversity_study

This repository is used to benchmark state-of-the-art intrinsic motivation methods for exploration in RL such as State Count, Random Network Distillation (RND), and Diversity is all you Need (DIAYN). This repository introduces two novel trajectory-based intrinsic motivation methods, Trajectory Window Count and Trajectory RND.

## Installation

1. Clone this repository.

2. Install `minigrid` environments and `torch-ac` RL algorithms:

```
pip3 install -r requirements.txt
```

## Instructions

To train an agent you need to use the following bash command:

```
python3 -m rl_starter_files.scripts.train [arguments]
 ```

The main arguments are:

General Arguments:

--algo: (REQUIRED) Algorithm to use. Use ppo.

--env: (REQUIRED) Name of the Minigrid environment to train on.

--model: (Optional) Name of the model. Default: {ENV}_{ALGO}_{TIME}.

--seed: Random seed. Default: 1.

--log-interval: Number of updates between logs. Default: 1.

--save-interval: Number of updates between saves (set to 0 to disable). Default: 10.

--procs: Number of processes. Default: 16.

--frames: Total number of frames. Default: 10**7.

--pretrained-model: Specifies if to use a pretrained model. Default: False.

Intrinsic Motivation Arguments:

--intrinsic-reward-model: (REQUIRED) Specifies the intrinsic reward model. Choices include None(no IM), count, RND, DIAYN, and TrajectoryCount, TrajectoryRND, TrajectoryWindowCount

--intrinsic-coef: Specifies the beta coefficient for the intrinsic reward. Default: 0.005.

--number-skills: Specifies the number of skills for DIAYN. Default: 10.

--window-size: Specifies the window size for TrajectoryWindowCount. Default: 5.

Singleton Env Arguments:

--singleton-env: Specifies if the env is singleton (default env is procedurally-generated). If this flat is set to true, the code will additionally plot an state visitation heatmap and intrinsic reward heatmap. Default False

## Example of use

Train an agent on the `MiniGrid-MultiRoom-N7-S4-v0` environment for 100000 with a state count intrinsic reward model and an intrinsic coefficient of 0.005:

```
python3 -m rl_starter_files.scripts.train --algo ppo --env MiniGrid-MultiRoom-N7-S4-v0 --model MultiroomN7S4 --seed 1 --save-interval 10 --frames 100000 --intrinsic-reward-model count --intrinsic-coef 0.005
```

## Citations

The torc_ac folder uses the https://github.com/lcswillems/torch-ac.git repo.

The rl-starter-file folder uses the https://github.com/lcswillems/rl-starter-files.git repo.

The Minigrid environment uses the https://github.com/Farama-Foundation/Minigrid.git repo.
