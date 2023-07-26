import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
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


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=True, use_text=False, use_diayn=False, skill_size=0):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Add a flag for DIAYN since in that method, the actor-critic has to take as input the skill as well
        # the agent's policy (the actor) is rewarded for making the discriminator's job as difficult as possible
        self.use_diayn = use_diayn
        self.skill_size = skill_size

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

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size
    
        # If we add the skill as input, the embedding size increases by the skill embedding size
        if self.use_diayn == True:
            self.embedding_size  += self.skill_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory, skills = None):
        # transpose your image and put it through your image embedding net
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            lstm_embedding = embedding
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x
            lstm_embedding = None

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        # print("embedding size", embedding.shape )
        # print("skill size", skills.shape)

        if self.use_diayn == True:
            # Expand dimensions to match the batch size of the embeddings
            # skills = skills.unsqueeze(0).expand(embedding.size(0), -1)
            embedding = torch.cat((embedding, skills), dim=1)

        # print("embedding size", embedding.shape )

        # The output x is typically a tensor where each element represents the raw (unnormalized) 
        # log-probability of taking a particular action. These raw log-probabilities are often called "logits".
        x = self.actor(embedding)

        # log_softmax will exponentiate and normalize the tensor to get an actual
        # tensor of probabilities over the space of actions then Categorical creates a Pytorch categorical distribution
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        # the critic component will output the expected value of the state under the current policy
        x = self.critic(embedding)
        value = x.squeeze(1)

        # return the true distribution over states, the value obtained from the critic network
        return dist, value, memory, lstm_embedding

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
