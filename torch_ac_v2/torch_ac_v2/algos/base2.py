from abc import ABC, abstractmethod
import torch
import sys
import os
import math
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, '/cluster/project7/diversity_rl/diversity_study/torch_ac_v2/torch_ac_v2')

import format
from format import default_preprocess_obss

sys.path.insert(0, '/cluster/project7/diversity_rl/diversity_study/torch_ac_v2/torch_ac_v2/utils')

from dictlist import DictList
from penv import ParallelEnv
from count_module import CountModule,TrajectoryCountModule, DIAYN_reward, DIAYN_discriminator, RNDModel, WindowTrajectory, CNN_encoder, WindowEncoder, WindowDecoder,RNDTrajectoryModel

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs,obs_space,action_space, acmodel, state_action_model, intrinsic_reward_model, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss,intrinsic_coef, no_skills, window_size, reshape_reward):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
            the number of frames per process refers to the number of steps each environment (or process)
            takes before the agent updates its parameters; this parameter is important since it controls
            the trade-off between the sample efficiency and the computational efficiency (more time steps might
            mean better updates but the the training might be slow)
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
            adding an entropy to the loss function is a form of regularization as 
            it prevents premature convergence to a deterministic policy
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value to avoid vanishing or exploding
            gradients which is a common problem with neural nets; if the gradients vector gets 
            to the maximum treshold it'll get re-scaled
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        intrinsic_coef : float
            coefficient of the intrinsic reward for diversity
        """

        # Store parameters

        self.beta = intrinsic_coef
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.intrinsic_coef=intrinsic_coef
        self.state_action_model = state_action_model
        self.intrinsic_reward_model = intrinsic_reward_model
        self.no_skills = no_skills
        self.obs_space = obs_space
        self.action_space = action_space
        self.window_size = window_size

        self.env = ParallelEnv(envs)

        
        # initialize the state count module
        self.count_module = CountModule()

        # initialize the trajectory count module
        self.trajectory_count_module = TrajectoryCountModule()
        self.window_trajectory_count_module = WindowTrajectory()


        if self.intrinsic_reward_model == "TrajectoryAutoencoder":        
            # initialize the CNN encoder + one-hot action encoder for the state-action pair
            self.cnn_state_action_encoder = CNN_encoder(self.obs_space,self.action_space)
            self.cnn_state_action_encoder.to(self.device)
            # initialize trajectory window encoder
            window_encoder_size = 71 * self.window_size
            self.trajectory_window_encoder = WindowEncoder(window_encoder_size, hidden_size = 10, output_size = 64)
            self.trajectory_window_encoder.to(self.device)
            # initialize trajectory window decoder
            self.trajectory_window_decoder = WindowDecoder(64, 10, window_encoder_size)
            self.trajectory_window_decoder.to(self.device)

        if self.intrinsic_reward_model == "DIAYN":        

            # initialize the DIAYN discriminator
            self.diayn_discriminator = DIAYN_discriminator(self.obs_space, self.no_skills)
            self.diayn_discriminator.to(self.device)

            # DIAYN reward class
            self.diayn_reward = DIAYN_reward(self.no_skills, self.diayn_discriminator,self.intrinsic_coef)

        # initialize the RND module for the RND networks (predictor + target)
        self.rnd_model = RNDModel(self.obs_space)
        self.rnd_model.to(self.device)

        # initialize the RND module for the RND networks (predictor + target)
        self.rnd_trajectory_model = RNDTrajectoryModel()
        self.rnd_trajectory_model.to(self.device)     

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1

        # we need the number of frames per processs to be a multiple of the recurrence attribute
        # because of how an rnn works it will backprop a certain number of steps so we want to
        # effectively use all the training data

        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        # calculate the number of parallel environments
        self.num_procs = len(envs)
        # total number of steps across all environments
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        # the shape of experience; each column is the experience in one of the parallel environments
        shape = (self.num_frames_per_proc, self.num_procs)

        lstm_shape = (self.num_frames_per_proc, self.num_procs,64)

        self.obs = self.env.reset()
        # empty list with None for each step until the process ends
        self.obss = [None] * (shape[0])

        # initiate memory
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
            print("recurrent")
        # mask has a 1 for each environment
        self.mask = torch.ones(shape[1], device=self.device)
        # masks is a zero matrix with no of frames x no of parallel environments
        self.masks = torch.zeros(*shape, device=self.device)
        
        # actions/values/rewards are a zero matrix with no of frames x no of parallel environments
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        # the rewards tensor will include both intrinsic and extrinsic rewards
        self.rewards = torch.zeros(*shape, device=self.device)
        self.total_rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        if self.intrinsic_reward_model == "RND":
            self.rnd_loss = torch.zeros(*shape, device=self.device)
        
        if self.intrinsic_reward_model == "TrajectoryAutoencoder":
            self.trajectory_autoencoder_rewards = torch.zeros(*shape, device=self.device)
            self.trajectory_windows = [None] * (shape[0])
            self.frame_trajectory = [None] * (shape[1])


        # Initialize log values with a zero entry for each parallel environment
        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_ext_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        # the above log values are updated during the episode and then appended to the final values and reset to 0
        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_ext_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

        # # add avg 100 episodes return
        # self.last_100return = deque([0],maxlen=100)

        # Initialize list of empty trajectories for all parallel environments
        self.trajectories = [[] for _ in range(self.num_procs)]

        # Initialize the list of embedded trajectories
        self.embedded_trajectories = torch.ones((self.num_procs, 65), dtype=torch.float32)

        if self.intrinsic_reward_model == "DIAYN":
            # sample a skill for each parallel environment
            self.skills = self.diayn_reward.sample_skills(self.num_procs).tolist()
        
            # One-hot encode skills for each environment to be fed to the actor-critic networks
            self.one_hot_skills = self.diayn_reward.hot_encode_skills(torch.tensor(self.skills))
        
        else:
            # if the skills are None then the forward pass of the actor-critic will ignore the skills
            self.skills = None
            self.one_hot_skills = None

        # placeholder to keep track of the skills of each environment at each frame
        self.skills_tracker = torch.zeros(*shape, device = self.device)


    # the experience collector runs in all the environments at the same time and 
    # the next actions are computed in a batch mode for all environments at the same time
    # the exps returns the experience details
    # logs consist of average reward so far, policy loss, value loss
    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        # the length of the loop is the number of actions until the gradient update
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            # take the observation and preprocess it
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            with torch.no_grad():
                # if you have a recurrent model you have a memory and you try to return a distribution
                # over the action space and an approx of state value which is the expected cumulative return
                # under the current policy
                if self.acmodel.recurrent:
                    dist, value, memory, lstm_embedding = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1),self.one_hot_skills)
                else:
                    dist, value = self.acmodel(preprocessed_obs, False, self.one_hot_skills)
                    
            # sample from the distribution over actions
            action = dist.sample()


            # take a step using the sampled action and get the extrinsic reward
            obs, reward, terminated, truncated,agent_loc,_= self.env.step(action.cpu().numpy())

            input_next_obs = self.preprocess_obss(obs, device=self.device)

            # keep track of the extrinsic rewards
            self.log_episode_ext_return += torch.tensor(reward, device=self.device, dtype=torch.float)

            # check the model for intrinsic reward
            if self.intrinsic_reward_model == "count":
                # print("Using state-count")

                # calculate the count-based intrinsic reward:
                count_intrinsic_reward = [self.count_module.count_based_ir(ob, act) for ob,act in zip(input_next_obs.image, action)]

                # update the state or state-action count
                state_counts = self.count_module.update_count(input_next_obs.image, action)

                total_reward = np.array(reward,dtype=np.float64)
                total_reward += np.array(count_intrinsic_reward, dtype=np.float64)
                total_reward = tuple(total_reward)

            elif self.intrinsic_reward_model == "RND":
                print("Using RND")
                # only add the rnd intrinsic reward if the model is not None
                # if self.rnd_model.recurrent:
                #     # shape is no of parallel envs x latent dim (512)
                #     predictor_output, target_output = self.rnd_model(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                # else:
                predictor_output, target_output= self.rnd_model(input_next_obs)

                # calculate the intrinsic reward as the MSE between the output of the target and predictor nets for each parallel environment

                # Calculate the element-wise square of the difference
                square_diff = torch.pow(target_output - predictor_output, 2)

                # Calculate the mean over the first dimension
                intrinsic_reward = torch.mean(square_diff, dim=1)

                # Keep track of the MSE loss to update the params of the predictor network
                self.rnd_loss[i]=intrinsic_reward

                # Add the intrinsic reward to the the extrinsic/envs reward
                total_reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True)  # Ensure reward is float and requires grad
                total_reward = total_reward.clone() + self.intrinsic_coef * intrinsic_reward
                # print("reward is", reward)
                total_reward = tuple(total_reward)

            elif self.intrinsic_reward_model == "TrajectoryCount":
                # print("Using Trajectory Count")
                
                trajectory_intrinsic_reward = []

                # encode the state-action for each environment and add it to the trajectory if it wasn't there already
                for idx,(ob, act) in enumerate(zip(input_next_obs.image, action)):
                    state_act = self.trajectory_count_module.state_action_encoder(ob,act)
                    if state_act not in self.trajectories[idx]:
                        self.trajectories[idx].append(state_act)
                        intrinsic_reward, trajectory_counts = self.trajectory_count_module.update_trajectory_count(self.trajectories[idx])
                        trajectory_intrinsic_reward.append(intrinsic_reward)

                    else:
                        trajectory_intrinsic_reward.append(0)

                # Add the intrinsic reward to the the extrinsic/envs reward
                total_reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True)  # Ensure reward is float and requires grad
                trajectory_intrinsic_reward = torch.tensor(trajectory_intrinsic_reward, dtype=torch.float32, requires_grad=True)
                total_reward = total_reward.clone() + trajectory_intrinsic_reward
                # print("reward is", reward)
                total_reward = tuple(total_reward)

            elif self.intrinsic_reward_model == "DIAYN":
                # print("Using DYAIN")

                diayn_rewards = []

                # Gets the probability that the discriminator gives to the correct skill
                discriminator_predicted_probabilities_unnormalised = self.diayn_discriminator(input_next_obs)
                probability_of_correct_skill = F.softmax(discriminator_predicted_probabilities_unnormalised)

                for idx,ob in enumerate(input_next_obs.image):
                    diayn_reward = self.diayn_reward.diayn_reward(probability_of_correct_skill[idx,self.skills[idx]])
                    diayn_rewards.append(diayn_reward)
                
                # print("diayn rewards ", diayn_rewards)

                # Add the intrinsic reward to the the extrinsic/envs reward
                total_reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True)  # Ensure reward is float and requires grad
                diayn_rewards = torch.tensor(diayn_rewards, dtype=torch.float32, requires_grad=True)
                total_reward = total_reward.clone() + diayn_rewards
                total_reward = tuple(reward)   

            elif self.intrinsic_reward_model == "TrajectoryWindowCount":

                window_count_intrinsic_reward = []

                for idx,(ob, act) in enumerate(zip(input_next_obs.image, action)):
                    # concatenate the state-action pair
                    state_act = self.window_trajectory_count_module.state_action_encoder(ob,act)
                    # add it to the trajectory
                    self.trajectories[idx].append(state_act)
                    if len(self.trajectories[idx]) >= self.window_size:
                        last_window = self.trajectories[idx][-self.window_size:]
                        window_int_reward, window_counts = self.window_trajectory_count_module.update_window_count(last_window)
                        window_count_intrinsic_reward.append(window_int_reward)
                    else:
                        window_count_intrinsic_reward.append(0)
                
                # Add the intrinsic reward to the the extrinsic/envs reward
                total_reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True)  # Ensure reward is float and requires grad
                window_count_intrinsic_reward = torch.tensor(window_count_intrinsic_reward, dtype=torch.float32, requires_grad=True)
                total_reward = total_reward.clone() + window_count_intrinsic_reward
                # print("reward is", reward)
                total_reward = tuple(total_reward)

            elif self.intrinsic_reward_model == "TrajectoryAutoencoder":

                state_actions_encodings = self.cnn_state_action_encoder(preprocessed_obs.image,action)

                for idx,state_action in enumerate(state_actions_encodings):
                    self.trajectories[idx].append(state_action)
                    if len(self.trajectories[idx]) >= self.window_size:
                        last_window = self.trajectories[idx][-self.window_size:]
                        last_window = torch.cat(last_window, dim = 0)
                    else:
                        last_window = self.trajectories[idx]
                        pad  = torch.zeros(self.window_size * 71 - len(last_window)*71)
                        if len(last_window) == 0:
                            last_window = pad
                        else:
                            last_window = torch.cat(last_window, dim = 0)
                            last_window = torch.cat((pad,last_window), dim = 0)
                    
                    self.frame_trajectory[idx] = last_window

                    last_window = last_window.unsqueeze(0)
                    encoded_window, input_hidden = self.trajectory_window_encoder(last_window.unsqueeze(0))
                    decoded_window, _ = self.trajectory_window_decoder(encoded_window,input_hidden)

                    # Calculate the element-wise square of the difference
                    square_diff = torch.pow(decoded_window - last_window, 2)

                    # Calculate the mean over the first dimension
                    traj_intrinsic_reward = torch.mean(square_diff, dim=1)

                    self.trajectory_autoencoder_rewards[i] = traj_intrinsic_reward

                # Add the intrinsic reward to the the extrinsic/envs reward
                total_reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True)  # Ensure reward is float and requires grad
                total_reward = total_reward.clone() + self.intrinsic_coef * self.trajectory_autoencoder_rewards[i, :]
                # print("reward is", reward)
                total_reward = tuple(total_reward)

            elif self.intrinsic_reward_model == "TrajectoryRND":
 
                predictor_output, target_output= self.rnd_trajectory_model(lstm_embedding)

                # calculate the intrinsic reward as the MSE between the output of the target and predictor nets for each parallel environment

                # Calculate the element-wise square of the difference
                square_diff = torch.pow(target_output - predictor_output, 2)

                # Calculate the mean over the first dimension
                traj_rnd_intrinsic_reward = torch.mean(square_diff, dim=1)

                # Add the intrinsic reward to the the extrinsic/envs reward
                total_reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True)  # Ensure reward is float and requires grad
                total_reward = total_reward.clone() + self.intrinsic_coef * traj_rnd_intrinsic_reward
                # print("reward is", reward)
                total_reward = tuple(total_reward)
                print(reward)
            
            # for no intrinsic model
            elif self.intrinsic_reward_model == None:

                total_reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True)
                total_reward = tuple(total_reward) 


            # essentially the logical OR operator -> check if the episode is done
            # which is either ended in a natural way (terminal state) or somehow cutoff before
            done = tuple(a | b for a, b in zip(terminated, truncated))

            # Update experiences values
            # current obs
            self.obss[i] = self.obs
            # update obs to the next obs
            self.obs = obs
            # sane for memory
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            # masks are used in the case where an episode ends before the total number of
            # frames per process to indicate that the extra steps are not part of the experience 
            # and they should not be used whe computing the gradients updates
            self.masks[i] = self.mask
            # the mask is set to 0 once the episode ends and 1 otherwise
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            # the agent keeps track of the action taken
            # each row is one step/frame and each column is an environment
            # tensor of length number of environments
            self.actions[i] = action
            # print(action.shape)
            # the agent keeps track of the state values
            self.values[i] = value
            # keep track of rewards
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)

            if self.reshape_reward is not None:
                self.total_rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, total_reward, done)
                ], device=self.device)
            else:
                self.total_rewards[i] = torch.tensor(total_reward, device=self.device)

            # keep track of the skills
            if self.intrinsic_reward_model == "DIAYN":
                self.skills_tracker[i] = torch.tensor(self.skills, device=self.device)
            
            # keep track of the log prob of the action under the distribution over the action space 
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            # current reward received after taking the action
            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            # reshaped return
            self.log_episode_reshaped_return += self.rewards[i]
            # add one to each environment as one action was taken
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            # done includes both terminated and truncated
            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_ext_return.append(self.log_episode_ext_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

                    # if the episode is done then reset the trajectory
                    if self.intrinsic_reward_model == "TrajectoryCount":
                        self.trajectories[i] = []
                    
                    # if the episode is done, re-sample a skill
                    if self.intrinsic_reward_model == "DIAYN":
                        self.skills[i] = self.diayn_reward.sample_skills(1).item()
                        self.one_hot_skills = self.diayn_reward.hot_encode_skills(torch.tensor(self.skills))

            # if the episode ends then the mask is used to reset the return to 0
            self.log_episode_return *= self.mask
            self.log_episode_ext_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1),self.one_hot_skills)
            else:
                _, next_value = self.acmodel(preprocessed_obs,None,self.one_hot_skills)


        # calculate the advantage using the GAE framework
        # look at the states in reverse order
        # the mask is used to handle the boundaries between episodes i.e. it is set to 
        # 1 if the current step is not the last episode and 0 when it is
        for i in reversed(range(self.num_frames_per_proc)):
            # take the next mask
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            # take the next value
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            # nect advantage
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            # reward + discount * next_state_value*next_mask - current value
            delta = self.total_rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        # dictionary of lists of the same length
        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]

        
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T

        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        exps.skill = self.skills_tracker.transpose(0, 1).reshape(-1)

        if self.intrinsic_reward_model == "RND":
            exps.rnd_loss = self.rnd_loss.transpose(0, 1).reshape(-1)

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        # keep only the max of done episodes and the total number of environments
        keep = max(self.log_done_counter, self.num_procs)
        print("log_done_counter",self.log_done_counter)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "ext_return_per_episode": self.log_ext_return[-keep:]
        }

        # reset the counters
        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_ext_return = self.log_ext_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        # return experience and logs
        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
