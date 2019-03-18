import numpy as np
import gym,os, time, sys, random
import argparse
import logging
import ray
import copy
from core import ddpg_new as ddpg
import torch
import utils
import time
from core import mod_neuro_evo as utils_ne



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Parameters:
    def __init__(self):
        self.input_size = None
        self.hidden_size = 36
        self.num_actions = None
        self.learning_rate = 0.1

        #Number of Frames to Run
        # if env_tag == 'Hopper-v2': self.num_frames = 4000000
        # elif env_tag == 'Ant-v2': self.num_frames = 6000000
        # elif env_tag == 'Walker2d-v2': self.num_frames = 8000000
        # else: self.num_frames = 2000000

        #USE CUDA
        self.is_cuda = True; self.is_memory_cuda = True

        #Sunchronization Period
        # if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.synch_period = 1
        # else: self.synch_period = 10

        #DDPG params
        self.use_ln = True  # True
        self.gamma = 0.99; self.tau = 0.001
        self.seed = 7
        self.batch_size = 128
        self.buffer_size = 1000000
        self.frac_frames_train = 1.0
        self.use_done_mask = True

        ###### NeuroEvolution Params ########
        #Num of trials
        # if env_tag == 'Hopper-v2' or env_tag == 'Reacher-v2': self.num_evals = 5
        # elif env_tag == 'Walker2d-v2': self.num_evals = 3
        # else: self.num_evals = 1

        #Elitism Rate
        # if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.elite_fraction = 0.3
        # elif env_tag == 'Reacher-v2' or env_tag == 'Walker2d-v2': self.elite_fraction = 0.2
        # else: self.elite_fraction = 0.1
        self.elite_fraction = 0.1

        self.pop_size = 10
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9

        #Save Results
        self.state_dim = None; self.action_dim = None #Simply instantiate them here, will be initialized later
        # self.save_foldername = 'test3-debug/%s/' % env_tag
        # if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)


@ray.remote(num_gpus=0.2)
class Worker(object):
    def __init__(self, args):
        # self.env = utils.NormalizedActions(gym.make(env_tag))
        self.env = gym.make(args.env_name)
        self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.policy = ddpg.DDPG(state_dim, action_dim, max_action)
        print("in worker init critic,", self.policy.critic.state_dict()["l3.bias"])

        self.replay_buffer = utils.ReplayBuffer()

        self.args = args
        self.total_timesteps = 0
        self.episode_num = 0
        self.timesteps_since_eval = 0

    def init_nets(self, actor_weight_init, critic_weight_init):
        self.policy.critic.load_state_dict(critic_weight_init)
        self.policy.critic_target.load_state_dict(self.policy.critic.state_dict())

        self.policy.actor.load_state_dict(actor_weight_init)
        self.policy.actor_target.load_state_dict(self.policy.actor.state_dict())

        return 1

    def set_weights(self,actor_weights, critic_weights):
        if actor_weights is not None:
            # print("come here 1")
            self.policy.actor.load_state_dict(actor_weights)
        self.policy.critic.load_state_dict(critic_weights)

        for param, target_param in zip(self.policy.critic.parameters(), self.policy.critic_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        if actor_weights is not None:
            # print("come here 2")
            for param, target_param in zip(self.policy.actor.parameters(), self.policy.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    # Runs policy for X episodes and returns average reward
    def evaluate_policy(self, actor_weights, critic_weights, eval_episodes=10):
        # self.set_weights(actor_weights,critic_weights)
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

    def train(self, actor_weights, critic_weights):
        # print("into 0 self.policy.actor,", self.policy.actor.state_dict()["l3.bias"])
        self.set_weights(actor_weights, critic_weights)
        print("set_weight self.policy.critic,", self.policy.critic.state_dict()["l3.bias"])

        # self.policy_debug.actor.load_state_dict(self.policy.actor.state_dict())
        # self.policy_debug.critic.load_state_dict(self.policy.critic.state_dict())

        # print("into 1 self.policy.actor,", self.policy.actor.state_dict()["l3.bias"])
        # print("into 1 self.policy_debug.actor,", self.policy_debug.actor.state_dict()["l3.bias"])

        # grads_critic = [param.grad.data.cpu().numpy() if param.grad is not None else None
        #                 for param in self.policy.critic.parameters()]
        # print("grads_critic before,",grads_critic)
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
                # Reset environment test on child process
                # obs = self.env.reset()
                # done = False
                # episode_reward = 0
                # episode_timesteps = 0
                break
            # Select action randomly or according to policy
            if self.total_timesteps < args.start_timesteps:
                action = self.env.action_space.sample()
            else:
                action = self.policy.select_action(np.array(obs))
                # if args.expl_noise != 0:
                #     action = (action + np.random.normal(0, args.expl_noise, size=self.env.action_space.shape[0])).clip(self.env.action_space.low, self.env.action_space.high)

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

        print("before self.policy.critic,", self.policy.critic.state_dict()["l3.bias"])
        # return self.policy.critic.cpu().state_dict()["l3.bias"], self.policy_debug.critic.cpu().state_dict()["l3.bias"]

        return self.total_timesteps, self.policy.grads_critic, episode_reward


def process_results(r):
    total_t = []
    grads_c = []
    all_f = []
    for result in r:
        all_f.append(result[2])
        grads_c.append(result[1])
        total_t.append(result[0])
    return sum(total_t), grads_c, all_f


def apply_grads(policy_net, critic_grad):
    policy_net.critic_optimizer.zero_grad()
    for worker_grad in critic_grad:
        for grad in worker_grad:
            for g, p in zip(grad, policy_net.critic.parameters()):
                if g is not None:
                    p.grad = torch.from_numpy(g).to(device)
            policy_net.critic_optimizer.step()


if __name__ == "__main__":
    num_workers = 1
    parameters = Parameters()
    evolver = utils_ne.SSNE(parameters)

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
    print("in main policy,", policy.critic.state_dict()["l3.bias"])

    ray.init(include_webui=False, ignore_reinit_error=True,object_store_memory=30000000000)

    # g_critic = ddpg.Critic(state_dim, action_dim)
    # g_critic_optimizer = torch.optim.Adam(g_critic.parameters())
    # print("in main g_critic,", g_critic.state_dict()["l3.bias"])

    actors = []
    for _ in range(num_workers):
        actors.append(ddpg.Actor(state_dim, action_dim, max_action))

    workers = [Worker.remote(args)
               for _ in range(num_workers+1)]

    # init_result_id = [worker.init_nets.remote(actor.state_dict(), g_critic.state_dict()) for worker, actor in zip(workers[:-1], actors)]
    #
    # print(ray.get(init_result_id))

    # evaluations = [ray.get(workers[-1].evaluate_policy.remote(policy.actor.state_dict(),policy.critic.state_dict()))]
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_timesteps = 0
    done = True
    time_start = time.time()
    debug = True

    while total_timesteps < args.max_timesteps:
        # if debug:
        #     actor_weight = actors[0].state_dict()
        # else:
        #     actor_weight = None
        train_id = [worker.train.remote(actor.state_dict(), policy.critic.state_dict()) for worker, actor in zip(workers[:-1],actors)]
        results = ray.get(train_id)
        total_timesteps, grads_critic, all_fitness = process_results(results)
        apply_grads(policy, grads_critic)
        print(time.time()-time_start)
        # debug = False
        print("after apply_grads self.policy.critic,", policy.critic.state_dict()["l3.bias"])
        # elite_index = evolver.epoch(actors, all_fitness)
        # exit(0)
    # Final evaluation
    # evaluations.append(ray.get(workers[-1].evaluate_policy.remote(policy.actor.state_dict(),policy.critic.state_dict())))
    # print("done")
    # exit(0)
    # if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
    # np.save("./results/%s" % (file_name), evaluations)












