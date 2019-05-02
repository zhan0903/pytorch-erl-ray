from ray.rllib.evaluation import PolicyGraph
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import to_numpy
from copy import deepcopy
import pysnooper
from threading import Lock
import logging
from ray.rllib.utils.annotations import override
from ray.rllib.agents.dqn.dqn_policy_graph import (
    _huber_loss, _minimize_and_clip, _scope_vars, _postprocess_dqn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        # self.cuda()

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def set_grads(self, grads):
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.grad = torch.from_numpy(
                    grads[cpt:cpt + tmp]).view(param.size()).cuda()
            else:
                param.grad = torch.from_numpy(
                    grads[cpt:cpt + tmp]).view(param.size())
            cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)
        # self.cuda()

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def set_grads(self, grads):
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.grad = torch.from_numpy(
                    grads[cpt:cpt + tmp]).view(param.size()).cuda()
            else:
                param.grad = torch.from_numpy(
                    grads[cpt:cpt + tmp]).view(param.size())
            cpt += tmp

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.leaky_relu(self.l1(xu))
        x1 = F.leaky_relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.leaky_relu(self.l4(xu))
        x2 = F.leaky_relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.leaky_relu(self.l1(xu))
        x1 = F.leaky_relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class TD3Postprocessing(object):
    @override(PolicyGraph)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        if False:#self.config["parameter_noise"]:
            # adjust the sigma of parameter space noise
            states, noisy_actions = [
                list(x) for x in sample_batch.columns(
                    [SampleBatch.CUR_OBS, SampleBatch.ACTIONS])
            ]
            self.sess.run(self.remove_noise_op)
            clean_actions = self.sess.run(
                self.output_actions,
                feed_dict={
                    self.cur_observations: states,
                    self.stochastic: False,
                    self.eps: .0
                })
            distance_in_action_space = np.sqrt(
                np.mean(np.square(clean_actions - noisy_actions)))
            self.pi_distance = distance_in_action_space
            if distance_in_action_space < self.config["exploration_sigma"]:
                self.parameter_noise_sigma_val *= 1.01
            else:
                self.parameter_noise_sigma_val /= 1.01
            self.parameter_noise_sigma.load(
                self.parameter_noise_sigma_val, session=self.sess)

        return _postprocess_dqn(self, sample_batch)


class TD3PolicyGraph(TD3Postprocessing,PolicyGraph):
    # @pysnooper.snoop()
    def __init__(self, state_dim, action_dim, config):
        PolicyGraph.__init__(self, state_dim, action_dim, config)
        self.config = config
        self.max_action = config["max_action"]
        self.actor = Actor(state_dim.shape[0], action_dim.shape[0], self.max_action).to(device)
        self.actor_target = Actor(state_dim.shape[0], action_dim.shape[0], self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim.shape[0], action_dim.shape[0]).to(device)
        self.critic_target = Critic(state_dim.shape[0], action_dim.shape[0]).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.lock = Lock()

        # self.max_action = config["max_action"]

    # @pysnooper.snoop()
    def compute_single_action(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        return self.actor(obs).cpu().data.numpy().flatten()

    # @pysnooper.snoop()
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # logger.debug("len of obs_batch:{}".format(len(obs_batch)))
        with self.lock:
            with torch.no_grad():
                ob = torch.from_numpy(np.array(obs_batch)) \
                    .float().to(device)
                # actions_
                model_out = self.actor(ob)
                actions = model_out
                # action_dist = self._action_dist_cls(logits)
                # actions = action_dist.sample()
                return (actions.cpu().numpy(),
                        [],
                        self.extra_action_out(model_out))

    def extra_action_out(self, model_out):
        """Returns dict of extra info to include in experience batch.

        Arguments:
            model_out (list): Outputs of the policy model module."""
        return {}
    # def compute_td_error(self):
    #     pass

    def learn_on_batch(self, samples):
        # Sample replay buffer
        logger.info("learn on batch in td3 graph, do nothing")
        exit(0)
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(x).to(device)
        action = torch.FloatTensor(u).to(device)
        next_state = torch.FloatTensor(y).to(device)
        done = torch.FloatTensor(1 - d).to(device)
        reward = torch.FloatTensor(r).to(device)

        # Select action according to policy and add clipped noise
        noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (done * discount * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if it % policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # def compute_apply(self):
    #     print("compute_apply")
    #     pass
        
    def get_weights(self):
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.actor.parameters()]))

    def set_weights(self, params):
        self.actor.set_params(params)

    # def train(self,params):
    #     pass

