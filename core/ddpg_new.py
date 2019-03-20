import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


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

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.cuda()

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class PERL(object):
    def __init__(self, state_dim, action_dim, max_action, pop_size):
        self.pop_size = pop_size
        self.actors = [Actor(state_dim, action_dim, max_action).state_dict() for _ in range(pop_size)]
        self.critic = Critic(state_dim, action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def evolve(self):
        pass

    def apply_grads(self, grads):
        self.critic_optimizer.zero_grad()
        # for worker_grad in critic_grad:
        critic_grad = np.sum(grads, axis=0)/self.pop_size

        print(critic_grad[-1][-1])
        print(grads[0][-1][-1])

        for grad in critic_grad:
            self.critic_optimizer.zero_grad()
            for g, p in zip(grad, self.critic.parameters()):
                if g is not None:
                    p.grad = torch.from_numpy(g).to(device)
            self.critic_optimizer.step()


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.001, momentum=0.8)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.grads_critic = [] #[] # []
        # self.grads_actor = []

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def sum_grads(self):
        grads_critic = [param.grad.data.cpu().numpy() if param.grad is not None else None
                        for param in self.critic.parameters()]

        # grads_actor = [param.grad.data.cpu().numpy() if param.grad is not None else None
        #                for param in self.actor.parameters()]

        if self.grads_critic is None:
            self.grads_critic = grads_critic
        else:
            for t_grad, grad in zip(self.grads_critic, grads_critic):
                t_grad += grad

    def append_grads(self):
        # grads_critic = [param.grad.data.cpu().numpy() if param.grad is not None else None
        #                 for param in self.critic.parameters()]

        # grads_actor = [param_actor.grad.data.cpu().numpy() if param_actor.grad is not None else None
        #                for param_actor in self.actor.parameters()]

        grads_critic = [param_critic.grad.data.cpu().numpy() if param_critic.grad is not None else None
                        for param_critic in self.critic.parameters()]

        # print(grads_critic)
        # print(grads_actor)

        # self.grads_actor.append(grads_actor)
        self.grads_critic.append(grads_critic)

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        self.grads_critic = [] # []

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
            self.critic_optimizer.step()

            self.append_grads()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # self.append_grads()

            #Update the frozen target models
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
