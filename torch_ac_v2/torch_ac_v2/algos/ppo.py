import numpy
import torch
import torch.nn.functional as F
from torch import optim, nn

from .base2 import BaseAlgo

from torch_ac_v2.torch_ac_v2.utils.clip_grads import global_grad_norm_
from torch_ac_v2.torch_ac_v2.utils.count_module import CountModule,TrajectoryCountModule, DIAYN_reward, DIAYN_discriminator

class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs,obs_space,action_space, acmodel, state_action_model = None ,intrinsic_reward_model = None, device=None, num_frames_per_proc=None, discount=0.99, lr=0.0001, gae_lambda=0.95,
                 entropy_coef=0.0005, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None, intrinsic_coef = 0.005, no_skills = 10, window_size = 10,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs,obs_space, action_space , acmodel, state_action_model, intrinsic_reward_model, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss ,intrinsic_coef, no_skills, window_size, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.no_skills = no_skills
        self.obs_space = obs_space
        self.intrinsic_coef = intrinsic_coef
        self.window_size = window_size

        assert self.batch_size % self.recurrence == 0
    
        # define the optimizer
        if self.intrinsic_reward_model == "RND":
            combined_params = list(self.acmodel.parameters())
            self.optimizer = torch.optim.Adam(combined_params, lr, eps=adam_eps)
        elif self.intrinsic_reward_model == "DIAYN":
            combined_params = list(self.acmodel.parameters()) + list(self.diayn_discriminator.parameters())
            self.optimizer = torch.optim.Adam(combined_params, lr, eps=adam_eps)                 
        else:
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
    
        self.batch_num = 0

        # self.diayn_discriminator = DIAYN_discriminator(self.obs_space, self.no_skills)
        # self.diayn_discriminator.to(self.device)
        # self.diayn_reward = DIAYN_reward(self.no_skills, self.diayn_discriminator,self.intrinsic_coef)


    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_discriminator_loss = []

            for inds in self._get_batches_starting_indexes():

                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0
                batch_rnd_loss = 0
                batch_discriminator_loss = 0
                batch_trajectory_rnd_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss
                    
                    if self.intrinsic_reward_model == "DIAYN":
                        update_skills = self.diayn_reward.hot_encode_skills(sb.skill)
                    else:
                        update_skills = None

                    # call the AC model to get the prob distribution over states
                    if self.acmodel.recurrent:
                        dist, value, memory,lstm_embeddings = self.acmodel(sb.obs, memory * sb.mask,update_skills)
                    else:
                        dist, value, _ = self.acmodel(sb.obs,None, update_skills)


                    lstm_embeddings_copy = lstm_embeddings.clone()
                    lstm_embeddings_copy = lstm_embeddings_copy.detach()

                    # compute the entropy of the policy
                    print("no_policies", dist)
                    entropy = dist.entropy().mean()

                    # ratio between the old policy and the new policy 
                    # this is the exponential of the difference in log probabilities
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    # the TRPO is the product between the advantage (computed in the BaseAlgo using the GAE framework)
                    surr1 = ratio * sb.advantage
                    # clipping objective
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    # take minumum and take the mean cause it's an expectation
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # clip the value (i.e. past value function + clipped difference between current value and previous)
                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    # the two surrogate objectives are MSE
                    surr1 = (value - sb.returnn).pow(2)
                    # using value_clipped brings everything closer to the previous value loss
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()


                    # loss is the policy loss - entropy loss + value loss
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    if self.intrinsic_reward_model == "RND":

                        self.rnd_model.update(sb.obs)

                    if self.intrinsic_reward_model == "DIAYN":
                        discriminator_outputs = self.diayn_discriminator(sb.obs)
                        skills = sb.skill.long()
                        discriminator_loss = nn.CrossEntropyLoss()(discriminator_outputs, skills)


                    if self.intrinsic_reward_model == "TrajectoryRND":

                        # this need to take the LSTM embedding as input

                        self.rnd_trajectory_model.update(lstm_embeddings_copy)

                        
                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss
                    # if self.intrinsic_reward_model == "RND":
                    #     batch_rnd_loss += rnd_model_loss
                    if self.intrinsic_reward_model == "DIAYN":
                        batch_discriminator_loss += discriminator_loss
                    
                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence
                batch_discriminator_loss /= self.recurrence


                # Update actor-critic

                self.optimizer.zero_grad()

                loss = batch_loss  + batch_discriminator_loss 

                loss.backward()

                if self.intrinsic_reward_model == "DIAYN":
                    global_grad_norm_(list(self.acmodel.parameters())+list(self.diayn_discriminator.parameters()))
                    grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
                    # print("discriminator network update")                   
                else:
                    grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
                    # clip the grad to make sure you don't update it past a certain point that would create instability
                    torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)

                self.optimizer.step()               

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)
                # log_discriminator_loss.append(batch_discriminator_loss)


        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
            # "discriminator_loss": numpy.mean(log_discriminator_loss)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
