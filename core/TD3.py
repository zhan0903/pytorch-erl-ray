import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import to_numpy
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.cuda()

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
        self.cuda()

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


class PERL(object):
    def __init__(self, state_dim, action_dim, max_action, pop_size):
        self.pop_size = pop_size
        self.actors = [Actor(state_dim, action_dim, max_action) for _ in range(pop_size)]
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def evolve(self):
        pass

    def select_action(self, state, actor_id):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actors[actor_id](state).cpu().data.numpy().flatten()

    def process_gradients(self, gradients, steps):
        steps.sort()
        # print("steps,", steps)
        import collections
        counter = collections.Counter(steps)
        # print("counter,", counter)
        gradients_new = []
        for key, value in counter.items():
            gradients_temp = []
            key_start = len(gradients_new)
            # print("key,key_start", key, key_start)
            for item in gradients:
                # print("len(item),", len(item))
                if len(item) > key_start:
                    gradients_temp.append(item[key_start:key])
            gradients_new.extend(np.sum(gradients_temp, axis=0) / value)
            key_start += key
        # print("len of gradients_new,", len(gradients_new))

        return np.array(gradients_new)

    def apply_grads(self, actor_critc, gradient_critic, steps, logger):
        # gradients_new = self.process_gradients(gradient_critic)

        # logger.info("shape of gradient_critic:{}".format(gradient_critic.shape))
        # logger.info("gradient_critic[1][1][:5]:{}".format(gradient_critic[1][1][:5]))

        critic_grad = self.process_gradients(gradient_critic, steps)
        # logger.info("shape of critic_grad:{}".format(critic_grad.shape))
        # logger.info("critic_grad[1][:5]:{}".format(critic_grad[1][:5]))
        # logger.info("gradient:{}".format(critic_grad[-1][:5]))

        for grad in critic_grad:
            self.critic_optimizer.zero_grad()
            self.critic.set_grads(grad)
            self.critic_optimizer.step()

        logger.info("after gradient update, self.critic.l6.bias:{}".format(self.critic.state_dict()["l6.bias"]))


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action

        self.grads_critic = []
        self.grads_actor = []

    # def set_weights(self, params_critic, tau=0.005):
    #     # self.actor.load_state_dict(params_actor)
    #     self.critic.load_state_dict(params_critic)
    #
    #     # Update the frozen target models
    #     for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
    #         target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # def get_weights(self):
    #     return self.critic.state_dict()

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return self.actor.get_params(), self.critic.get_params()

    def set_params(self, params_actor, params_critic, tau=0.005):
        """
        Set the params of the network to the given parameters
        """
        if params_actor is not None:
            self.actor.set_params(params_actor)
        self.critic.set_params(params_critic)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def apply_gradients(self, gradient_actor, gradient_critic):
        for actor_grad, critic_grad in zip(gradient_actor, gradient_critic):
            self.actor_optimizer.zero_grad()
            self.actor.set_grads(actor_grad)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            self.critic.set_grads(critic_grad)
            self.critic_optimizer.step()

        # for grad in gradient_critic:
        #     self.critic_optimizer.zero_grad()
        #     self.critic.set_grads(grad)
        #     self.critic_optimizer.step()
        #
        # for grad in gradient_critic:
        #     self.critic_optimizer.zero_grad()
        #     for g, p in zip(grad, self.critic.parameters()):
        #         if g is not None:
        #             p.grad = torch.from_numpy(g).to(device)
        #     self.critic_optimizer.step()

    def append_grads(self):
        # grads_critic = [param_critic.grad.data.cpu().numpy() if param_critic.grad is not None else None
        #                 for param_critic in self.critic.parameters()]

        grads_critic = self.critic.get_grads()
        grads_actor = self.actor.get_grads()

        self.grads_critic.append(grads_critic)
        self.grads_actor.append(grads_actor)

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.grads_critic = []
        self.grads_actor = []

        for it in range(iterations):

            # Sample replay buffer
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
            # self.append_grads()

            # Delayed policy updates
            # if it % policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # actor_grads = []
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # self.append_grads_actor()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            self.append_grads()


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
