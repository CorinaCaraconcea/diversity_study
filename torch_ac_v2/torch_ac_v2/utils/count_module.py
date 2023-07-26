import math
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_ac

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    # takes as input a pytorch class m and checks if the class is linear, if that's the case then the weights are sampled from a normal 0,1 distribution
    # and rescaled by their squared root mean
    # if the bias is none then it is initialized as 0
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CountModule():
    def __init__(self, state_action = False,beta=0.005):
        self.state_action = state_action # define if depends also to the state
        self.counts = {}
        self.beta = beta

    def state_encoder(self, state, action):
        """
        Function to flatten the observed image state and encode the state or the state-action pair.
        """

        # this function should only take one observation
        flat_obs = state.flatten().tolist()

        if self.state_action == True:
            return (tuple(flat_obs), action)
        else:
            return (tuple(flat_obs))
        
    def count_based_ir(self, state, action):
        """
        Function to return the count-based IR
        """
        tup = self.state_encoder(state,action)
        if tup in self.counts:
            return self.beta/math.sqrt(self.counts[tup])
        else:
            return self.beta

    def update_count(self,states,actions):
        """
        Function to update the state/state-action counts and to return the intrinsic reward for each parallel env.
        """

        int_count_rewards = np.ones(len(actions))

        for idx,(o,a) in enumerate(zip(states,actions)):
            tup = self.state_encoder(o,a)
            if tup in self.counts:
                self.counts[tup] += 1
                v = self.beta/math.sqrt(self.counts[tup])
            else:
                self.counts[tup] = 1
                v = self.beta

        
            int_count_rewards[idx] = v

        return self.counts


class TrajectoryCountModule():
    def __init__(self,beta=0.0005):
        self.trajectory_counts = {}
        self.beta = beta

    def state_action_encoder(self, state, action):
        """
        Function to flatten the observed image state and encode the state or the state-action pair.
        """

        # this function should only take one observation
        flat_obs = state.flatten().tolist()

        return (tuple(flat_obs), action.item())    
    
    def update_trajectory_count(self,trajectory):
        """
        Function to update the state/state-action counts and to return the intrinsic reward for each parallel env.
        """

        if tuple(trajectory) in self.trajectory_counts.keys():
            self.trajectory_counts[tuple(trajectory)] += 1
            intrinsic_reward = self.beta/math.sqrt(self.trajectory_counts[tuple(trajectory)])
        else:
            self.trajectory_counts[tuple(trajectory)] = 1
            intrinsic_reward = self.beta

        return intrinsic_reward,self.trajectory_counts
    
class TrajectoryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TrajectoryEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # We only want the final timestep output, to have a single fixed-size vector per sequence
        out = out[:, -1, :]

        return out
    

class DIAYN_reward():
    def __init__(self,no_skills,discriminator,beta):
        self.discriminator = discriminator
        self.no_skills = no_skills
        # Each skill equally likely to be chosen
        self.prior_probability_of_skill = torch.tensor(1/no_skills)

        self.beta = beta

    def get_predicted_probability_of_skill(self, skill, next_state):
        """
        Gets the probability that the discriminator gives to the correct skill and also returns the full
        unnormalised probabilities vector which is the output of the discriminator network.
        """
        predicted_probabilities_unnormalised = self.discriminator(next_state)
        probability_of_correct_skill = F.softmax(predicted_probabilities_unnormalised)[:, skill]

        return  probability_of_correct_skill.item(), predicted_probabilities_unnormalised

    def diayn_reward(self, probability_correct_skill):
        """
        Calculates an intrinsic reward that encourages maximum exploration. It also keeps track of the discriminator
        outputs so they can be used for training
        """

        # calculate rewards as log q - log p
        intrinsic_reward = self.beta * (torch.log(probability_correct_skill) - torch.log(self.prior_probability_of_skill+ 1e-6))

        return intrinsic_reward.item()
    
    def sample_skills(self,no_envs):
        """
        Sample a skill for each environment (uniform sampling)
        """
        skills = np.random.randint(0, self.no_skills-1, no_envs)

        return skills

    
    def hot_encode_skills(self,skills):
        """
        Hot encode skills to be fed to actor-critic networks
        """
        # Create a zero tensor of shape [batch_size, num_skills]
        skills = skills.to(torch.int64)
        one_hot_skills = torch.zeros((skills.shape[0], self.no_skills), device=skills.device)
        # Set the indices of the skills to 1
        one_hot_skills.scatter_(1, skills.unsqueeze(1), 1)
        return one_hot_skills



class DIAYN_discriminator(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, no_skills, use_memory=False, use_text=False):
        super().__init__()

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Resize image embedding
        self.embedding_size = self.semi_memory_size

        # Define the discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, no_skills)
        )        
        
        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size
    
    def forward(self,next_state):
        """
        The forward takes as input the next_state and returns an unnormalized prob over skills latent space.
        """

        # apply the 3D CNN to the image 
        embedding = next_state.image.transpose(1, 3).transpose(2, 3)
        embedding = self.image_conv(embedding)
        embedding = embedding.reshape(embedding.shape[0], -1)

        # return an unnormalized distribution over the skills
        x = self.discriminator(embedding)

        return x

class RNDModel(nn.Module,torch_ac.RecurrentACModel):

    """
    RNDModel is an implementation of the Random Network Distillation (RND) 
    technique, which is used for generating intrinsic rewards in reinforcement learning.
    This technique encourages exploration by measuring the error of predicting the output of a
    fixed randomly initialized neural network (the "target" network) using another neural network 
    (the "predictor" network). The predictor network is trained to minimize the MSE loss between 
    its predictions and the output of the target network.
    """

    def __init__(self, obs_space,use_memory=False, use_text=False):
        super(RNDModel, self).__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define the embedding size
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # The predictor network, which is trained to predict the output of the target network
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # The target network, which is randomly initialized and then frozen
        self.target = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Initialize parameters correctly
        self.apply(init_params)

        # The target network is not trained, so we freeze its parameters
        for param in self.target.parameters():
            param.requires_grad = False

        for param in self.predictor.parameters():
            param.requires_grad = True

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    # The forward pass calculates the output of the target and predictor networks
    def forward(self, next_obs):
        x = next_obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        embedding = x

        target_feature = self.target(embedding)
        predict_feature = self.predictor(embedding)

        return predict_feature, target_feature

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
    


class WindowTrajectory():
    """
    WindowTrajectory is an implementation of trajectory windowing i.e. keeping track of windows of trajectories.
    """
    def __init__(self, state_action = False,beta=0.001):
        self.state_action = state_action # define if depends also to the state
        self.window_counts = {}
        self.beta = beta

    def state_action_encoder(self, state, action):
        """
        Function to flatten the observed image state and encode the state or the state-action pair.
        """

        # this function should only take one observation
        flat_obs = state.flatten().tolist()

        return (tuple(flat_obs), action.item())    
    
    def update_window_count(self,window):
        """
        Function to update the state/state-action counts and to return the intrinsic reward for each parallel env.
        """

        if tuple(window) in self.window_counts.keys():
            self.window_counts[tuple(window)] += 1
            intrinsic_reward = self.beta/math.sqrt(self.window_counts[tuple(window)])
        else:
            self.window_counts[tuple(window)] = 1
            intrinsic_reward = self.beta

        return intrinsic_reward,self.window_counts



class WindowEncoder(nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - fixed to a lower dimensionality

    The input to this class should be [batch_size, sequence_length, no_features]

    """
    def __init__(self, input_size, hidden_size, output_size):
        super(WindowEncoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size)
        
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        if hidden==None:
            self.hidden = (torch.zeros(1,1,self.hidden_size),
                           torch.zeros(1,1,self.hidden_size))
        else:
            self.hidden = hidden
            
        """
        inputs need to be in the right shape as defined in documentation
        - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        
        lstm_out - will contain the hidden states from all times in the sequence
        self.hidden - will contain the current hidden state and cell state
        """
        lstm_out, self.hidden = self.lstm(x.view(len(x),1,-1), 
                                          self.hidden)
        
        predictions = self.linear(lstm_out.view(len(x), -1))
        
        return predictions, self.hidden


class WindowDecoder(nn.Module):
    """
    input_size - will be the same as the output_size of the encoder
    hidden_size - Should be the same as the hidden_size of the encoder
    output_size - This will be the prediction size you desire
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(WindowDecoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size)
        
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        """
        For the decoder, the hidden state is required, it can't be None.
        """
        assert hidden is not None, "Hidden state must be provided to the decoder."

        lstm_out, hidden = self.lstm(x.view(x.shape[0],1,-1), hidden)
                
        predictions = self.linear(lstm_out.view(x.shape[0], -1))
        
        return predictions, hidden


    

class CNN_encoder(nn.Module):
    def __init__(self, obs_space,action_space):
        super().__init__()

        self.no_actions = action_space.n
        # print("number of actions:", self.no_actions)

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

    def hot_encode_action(self,actions):
        """
        Hot encode skills to be fed to actor-critic networks
        """
        # Create a zero tensor of shape [batch_size, num_skills]
        actions = actions.to(torch.int64)
        one_hot_actions = torch.zeros((actions.shape[0], self.no_actions), device=actions.device)
        # Set the indices of the skills to 1
        one_hot_actions.scatter_(1, actions.unsqueeze(1), 1)
        return one_hot_actions

    def forward(self,obs,action):
        # transpose your image and put it through your image embedding net
        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        action_encoded = self.hot_encode_action(action)
        # print("actions encoded",action_encoded)

        state_action_encoding = torch.cat((x,action_encoded),dim = -1)

        return state_action_encoding


class RNDTrajectoryModel(nn.Module,torch_ac.RecurrentACModel):

    """
    RNDModel is an implementation of the Random Network Distillation (RND) 
    technique, which is used for generating intrinsic rewards in reinforcement learning.
    This technique encourages exploration by measuring the error of predicting the output of a
    fixed randomly initialized neural network (the "target" network) using another neural network 
    (the "predictor" network). The predictor network is trained to minimize the MSE loss between 
    its predictions and the output of the target network.
    """

    def __init__(self):
        super(RNDTrajectoryModel, self).__init__()

        # The predictor network, which is trained to predict the output of the target network
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # The target network, which is randomly initialized and then frozen
        self.target = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Initialize parameters correctly
        self.apply(init_params)

        # The target network is not trained, so we freeze its parameters
        for param in self.target.parameters():
            param.requires_grad = False

        for param in self.predictor.parameters():
            param.requires_grad = True


    # The forward pass calculates the output of the target and predictor networks
    def forward(self, embedding):

        target_feature = self.target(embedding)
        predict_feature = self.predictor(embedding)

        return predict_feature, target_feature

