import torch

import utils
from .other import device
from model import ACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_name,model_dir,
                 argmax=False, num_envs=1, use_memory=False, use_text=False):
        # preprocess the observation space
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        # initialize the model as A2C and check if the agent's model uses memory and if it uses a text input
        self.acmodel = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text)
        # selects actions using argmax
        self.argmax = argmax
        # the number of environments in which the actor acts
        self.num_envs = num_envs

        # is the model recurrent, a memory tensor is initiated 
        # the shape is no of environments x memory size
        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)

        self.acmodel.load_state_dict(utils.get_model_state(model_name,model_dir))
        # cpu/gpu
        self.acmodel.to(device)
        # evaluate function
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir,model_name))

    def get_actions(self, obss):
        # takes the actions from the preprocessed environment
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            # do you keep the memory of previous experiences?
            # feeds them to the actor-critic model to get a distribution over actions
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        # either take argmax and pick the action with the highest prob or sample from the action space
        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    # simplified function for the case when the agent only acts in one environment
    def get_action(self, obs):
        return self.get_actions([obs])[0]

    # This method is used when the agent is acting in multiple environments at once. 
    # It takes a list of rewards and a list of boolean flags indicating whether each environment is done. 
    # If the model is recurrent, it uses these to update the agent's memories, masking out 
    # the memories of environments that are done.
    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    # simplified version for when the agent only acts in one environment
    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
