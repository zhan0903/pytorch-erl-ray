import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

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

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
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


    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2


    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
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

    def apply_grads(self, gradient_critic, logger):
        # with self.lock:

        # logger.debug("shape grads[0] size:{}".format(grads[0].shape))
        # logger.debug("shape grads[1] size:{}".format(grads[1].shape))
        #
        # logger.debug("shape grads[2] size:{}".format(grads[2].shape))
        # logger.debug("shape grads[3] size:{}".format(grads[3].shape))

        # logger.info("shape grads[0] size:{}".format((grads[0])))

        critic_grad = np.sum(gradient_critic, axis=0)/self.pop_size

        for grad in critic_grad:
            self.critic_optimizer.zero_grad()
            for g, p in zip(grad, self.critic.parameters()):
                if g is not None:
                    p.grad = torch.from_numpy(g).to(device)
            self.critic_optimizer.step()

            # for grad in gradient_actor:
            #     self.critic_optimizer.zero_grad()
            #     for g, p in zip(grad, self.critic.parameters()):
            #         if g is not None:
            #             p.grad = torch.from_numpy(g).to(device)
            #     self.critic_optimizer.step()


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
        # self.grads_actor = []

    def set_weights(self, params_critic, tau=0.005):
        # self.actor.load_state_dict(params_actor)
        self.critic.load_state_dict(params_critic)

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def get_weights(self):
        return self.critic.state_dict()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def apply_gradients(self, gradient_critic):
        for grad in gradient_critic:
            self.critic_optimizer.zero_grad()
            for g, p in zip(grad, self.critic.parameters()):
                if g is not None:
                    p.grad = torch.from_numpy(g).to(device)
            self.critic_optimizer.step()

        # for grad in gradient_actor:
        #     self.actor_optimizer.zero_grad()
        #     for g, p in zip(grad, self.actor.parameters()):
        #         if g is not None:
        #             p.grad = torch.from_numpy(g).to(device)
        #     self.actor_optimizer.step()

    def append_grads_critic(self):
        grads_critic = [param_critic.grad.data.cpu().numpy() if param_critic.grad is not None else None
                        for param_critic in self.critic.parameters()]

        self.grads_critic.append(grads_critic)

    # def append_grads_actor(self):
    #     grads_actor = [param_actor.grad.data.cpu().numpy() if param_actor.grad is not None else None
    #                     for param_actor in self.actor.parameters()]
    #
    #     self.grads_actor.append(grads_actor)

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.grads_critic = []
        # self.grads_actor = []

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
            self.append_grads_critic()

            # Delayed policy updates
            if it % policy_freq == 0:

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

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
