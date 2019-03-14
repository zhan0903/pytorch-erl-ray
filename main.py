import numpy as np
import gym,os, time, sys, random
import argparse
import logging
import ray
import copy
from core import ddpg_new as ddpg
from core import replay_memory
import torch
from torch.optim import Adam
import torch.nn as nn
from core import mod_utils as utils
from core import mod_neuro_evo as utils_ne
import utils
import time



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@ray.remote(num_gpus=0.1)
class Worker(object):
    def __init__(self, args):
        # self.env = utils.NormalizedActions(gym.make(env_tag))
        self.env = gym.make(args.env_name)
        self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.policy = ddpg.DDPG(state_dim, action_dim, max_action)
        self.replay_buffer = utils.ReplayBuffer()

        self.args = args
        self.total_timesteps = 0
        self.episode_num = 0
        self.timesteps_since_eval = 0

    def set_weights(self,actor_weights,critic_weights):
        self.policy.actor.load_state_dict(actor_weights)
        self.policy.critic.load_state_dict(critic_weights)

        for param, target_param in zip(self.policy.critic.parameters(), self.policy.critic_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

        for param, target_param in zip(self.policy.actor.parameters(), self.policy.actor_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    # Runs policy for X episodes and returns average reward
    def evaluate_policy(self, actor_weights, critic_weights, eval_episodes=10):
        self.set_weights(actor_weights,critic_weights)
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action = self.policy.select_action(np.array(obs))
                obs, reward, done, _ = self.env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
        print("---------------------------------------")
        return avg_reward

    def train(self,actor_weights, critic_weights):
        self.set_weights(actor_weights, critic_weights)

        done = False
        episode_timesteps = 0
        episode_reward = 0
        obs = self.env.reset()
        while True:
            if done:
                self.episode_num += 1
                if self.total_timesteps != 0:
                    print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (self.total_timesteps, self.episode_num, episode_timesteps, episode_reward))
                    self.policy.train(self.replay_buffer, episode_timesteps, self.args.batch_size, self.args.discount, self.args.tau)
                # Reset environment
                # obs = env.reset()
                # done = False
                # episode_reward = 0
                # episode_timesteps = 0
                break
            # Select action randomly or according to policy
            if self.total_timesteps < args.start_timesteps:
                action = self.env.action_space.sample()
            else:
                action = self.policy.select_action(np.array(obs))
                if args.expl_noise != 0:
                    action = (action + np.random.normal(0, args.expl_noise, size=self.env.action_space.shape[0])).clip(
                        self.env.action_space.low, self.env.action_space.high)
            # Perform action
            new_obs, reward, done, _ = self.env.step(action)
            done_bool = 0 if episode_timesteps + 1 == self.env._max_episode_steps else float(done)
            episode_reward += reward

            # Store data in replay buffer
            self.replay_buffer.add((obs, new_obs, action, reward, done_bool))
            obs = new_obs

            episode_timesteps += 1
            self.total_timesteps += 1
            self.timesteps_since_eval += 1

        grads_critic = [param.grad.data.cpu().numpy() if param.grad is not None else None
                 for param in self.policy.critic.parameters()]
        grads_actor = [param.grad.data.cpu().numpy() if param.grad is not None else None
                        for param in self.policy.actor.parameters()]

        return self.total_timesteps, grads_actor, grads_critic


def process_results(results):
    total_timesteps = []
    grads_critic = []
    grads_actor = []
    for result in results:
        grads_critic.append(result[2])
        grads_actor.append(result[1])
        total_timesteps.append(result[0])
    return sum(total_timesteps), grads_actor, grads_critic


def apply_grads(net,grads_actor,grads_critic):
    # update actor
    net.actor_optimizer.zero_grad()
    grads_sum_actor = copy.deepcopy(grads_actor[-1])
    for grad in grads_actor[:-1]:
        for temp_itme, grad_item in zip(grads_sum_actor, grad):
            if grad_item is not None:
                temp_itme += grad_item

    for g, p in zip(grads_sum_actor, net.actor.parameters()):
        if g is not None:
            p.grad = torch.from_numpy(g).to(device)
    net.actor_optimizer.step()

    # update critic
    net.critic_optimizer.zero_grad()
    grads_sum_critic = copy.deepcopy(grads_critic[-1])
    for grad in grads_critic[:-1]:
        for temp_itme, grad_item in zip(grads_sum_critic, grad):
            if grad_item is not None:
                temp_itme += grad_item

    for g, p in zip(grads_sum_critic, net.critic.parameters()):
        if g is not None:
            p.grad = torch.from_numpy(g).to(device)
    net.critic_optimizer.step()


if __name__ == "__main__":
    num_workers = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="OurDDPG")
    parser.add_argument("--env_name", default="HalfCheetah-v1")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=1e4, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=float)
    parser.add_argument("--max_timesteps", default=1e6, type=float)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise

    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % file_name)
    print("---------------------------------------")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")


    # Create Env
    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = ddpg.DDPG(state_dim, action_dim, max_action)


    ray.init(include_webui=False, ignore_reinit_error=True)
    workers = [Worker.remote(args)
               for _ in range(num_workers+1)]

    evaluations = [ray.get(workers[-1].evaluate_policy.remote(policy.actor.state_dict(),policy.critic.state_dict()))]
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_timesteps = 0
    done = True
    time_start = time.time()


    while total_timesteps < args.max_timesteps:
        train_id = [worker.train.remote(policy.actor.state_dict(),policy.critic.state_dict()) for worker in workers[:-1]]
        results = ray.get(train_id)
        total_timesteps,grads_actor,grads_critic = process_results(results)
        apply_grads(policy, grads_actor, grads_critic)

    # Final evaluation
    evaluations.append(ray.get(workers[-1].evaluate_policy.remote(policy.actor.state_dict(),policy.critic.state_dict())))
    if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)












