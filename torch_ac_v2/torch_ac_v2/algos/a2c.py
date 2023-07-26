import numpy
import torch
import sys
import os
import torch.nn.functional as F

from base2 import BaseAlgo

sys.path.insert(0, os.path.abspath('/Users/corinacaraconcea/Documents/UCL DSML/MSc projects/torch_ac_v2/torch_ac_v2/utils'))

from clip_grads import global_grad_norm_

class A2CAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel,rndmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None,wrapper = None, intrinsic_coef=0.005, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, rndmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss,wrapper,intrinsic_coef, reshape_reward)

            
        if self.rnd_model is not None:
            combined_params = list(self.acmodel.parameters()) + list(self.rnd_model.predictor.parameters())
            self.optimizer = torch.optim.RMSprop(combined_params, lr, alpha=rmsprop_alpha, eps=rmsprop_eps)
        else:
            self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr, alpha=rmsprop_alpha, eps=rmsprop_eps)
                                             

    def update_parameters(self, exps):
        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        # update the entropy in the loss function
        update_entropy = 0
        # update the expected value of the state under the current policy
        update_value = 0
        # update the policy loss(actor)
        update_policy_loss = 0
        # update the value func loss (critic)
        update_value_loss = 0
        update_loss = 0

        update_rnd_loss = 0

        # Initialize memory

        if self.acmodel.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # print("i is", i)
            # Create a sub-batch of experience

            # create a batch of experience starting from the starting indices for each experience batch
            sb = exps[inds + i]

            print("batch of experience length:", len(sb.obs))

            # Compute loss
            # call the model and get the distribution over actions and the value of the state and the memory
            if self.acmodel.recurrent:
                dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
            else:
                dist, value = self.acmodel(sb.obs)

            
            # compute the entropy of the distribution over states
            # dist is a Categorical pytorch object and .entropy is a built-in function to calculate the entropy
            entropy = dist.entropy().mean()

            # convert the prob to log_prob and multiply by the advantage 
            policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

            # calculate the value loss as the difference between the value and the current value+advantage
            value_loss = (value - sb.returnn).pow(2).mean()

            # the loss is the policy loss - entropy + value loss
            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

            if self.rnd_model is not None:
                if self.rnd_model.recurrent:
                    predict_feature, target_feature = self.rnd_model(sb.obs, memory * sb.mask)
                else:
                    predict_feature, target_feature = self.rnd_model(sb.obs)

                # Calculate the element-wise square of the difference
                square_diff = torch.pow(target_feature - predict_feature, 2)

                # Calculate the mean over the first dimension
                rnd_model_loss = torch.mean(square_diff, dim=1)
                rnd_model_loss = rnd_model_loss.mean()                

            # Update batch values
            # update the losses for the batch
            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss
            if self.rnd_model is not None:
                update_rnd_loss += rnd_model_loss

        # Update update values

        # divide by the recurrence
        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        if self.rnd_model is not None:
            update_rnd_loss /= self.recurrence

        # Update actor-critic

        # update the actor-critic

# 
        self.optimizer.zero_grad()

        loss = update_rnd_loss + update_loss

        loss.backward()

        if self.rnd_model is not None:
            global_grad_norm_(list(self.acmodel.parameters())+list(self.rnd_model.predictor.parameters()))
            update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        else:
            update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
            # clip the grad to make sure you don't update it past a certain point that would create instability
            torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
