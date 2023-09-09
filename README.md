# diversity_study

This repository is used to study benchmark state-of-the-art intrinsic motivation methods for exploration in RL. This repository also introduces two novel trajectory-based intrinsic motivation methods, Trajectory Window Count and Trajectory RND.

## Installation

1. Clone this repository.

2. Install `minigrid` environments and `torch-ac` RL algorithms:

```
pip3 install -r requirements.txt
```

## Example of use

Train an agent on the `MiniGrid-MultiRoom-N7-S4-v0` environment for 100000 with a state count intrinsic reward model and an intrinsic coefficient of 0.005:

```
python3 -m scripts.train --algo ppo --env MiniGrid-MultiRoom-N7-S4-v0 --model MultiroomN7S4 --seed 1 --save-interval 10 --frames 100000 --intrinsic-reward-model count --intrinsic-coef 0.005
```

## Citations

The torc_ac folder uses the https://github.com/lcswillems/torch-ac.git repo.

The rl-starter-file folder uses the https://github.com/lcswillems/rl-starter-files.git repo.
