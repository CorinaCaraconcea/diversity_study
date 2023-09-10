import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
from model import init_params

import torch_ac


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

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

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding/ keep it here but don't use text
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

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
    def forward(self, next_obs, memory):
        x = next_obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        # use memory?
        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        # use text?
        if self.use_text:
            embed_text = self._get_embed_text(next_obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        target_feature = self.target(embedding)
        predict_feature = self.predictor(embedding)

        return predict_feature, target_feature

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
    
    # def reset_memory(self, batch_size):
    #     # Reset the LSTM hidden state. You can modify this code based on your requirements.
    #     return torch.zeros((batch_size, self.memory_size)).to(self.memory_rnn.weight.device)



class StateActionNet(nn.Module):
    def __init__(self, obs_space,use_memory=False):
        super(StateActionNet, self).__init__()
        
        # Decide which components are enabled
        self.use_memory = use_memory

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
        
        #         # Define the feed-forward network for the action
        # self.action_net = nn.Sequential(
        #     nn.Linear(1, 3),
        #     nn.ReLU()
        # )
        
        # # Define the LSTM that will process the combined state-action embeddings
        # self.lstm = nn.LSTM(input_size=self.image_embedding_size + 3, 
        #                    hidden_size=hidden_size, 
        #                    num_layers=num_layers)

    def forward(self, obs, action):
        # Pass each state and action through their respective networks
        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        # action_embeddings = self.action_net(action)

        # Combine the state and action embeddings
        # combined = torch.cat((x, action_embeddings), dim=-1)

        # # Pass the combined embeddings through the RNN
        # output, _ = self.lstm(combined)

        return x

