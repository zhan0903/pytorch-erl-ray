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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class TD3PolicyGraph(PolicyGraph):
    # @pysnooper.snoop()
    def __init__(self, state_dim, action_dim, config):
        PolicyGraph.__init__(self, state_dim, action_dim, config)
        # self.config = config
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

    @pysnooper.snoop()
    def compute_single_action(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        return self.actor(obs).cpu().data.numpy().flatten()

    @pysnooper.snoop()
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        print("in graph,obs_batch,", obs_batch)
        with self.lock:
            with torch.no_grad():
                ob = torch.from_numpy(np.array(obs_batch)) \
                    .float().to(device)
                # actions_
                model_out = self.actor(ob)
                actions = model_out
                # action_dist = self._action_dist_cls(logits)
                # actions = action_dist.sample()
                return actions.cpu().data.numpy().flatten()

    # def compute_td_error(self):
    #     pass

    # def learn_on_batch(self, samples):
    #     print("learn_on_batch")
    #     pass

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

