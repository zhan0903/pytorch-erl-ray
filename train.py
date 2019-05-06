import numpy as np
import gym,os, time, sys, random
import argparse
import logging
import ray
import copy
from core import TD3 as ddpg
import torch
from utils import *
import time
from core import mod_neuro_evo as utils_ne
import math
from copy import deepcopy
import pyarrow as pa
# from ray.rllib.optimizers.async_replay_optimizer import AsyncReplayOptimizer

# from ray.rllib.optimizers.async_replay_optimizer import AsyncReplayOptimizer
from core.async_replay_optimizer import AsyncReplayOptimizer

from ray.rllib.evaluation import PolicyGraph, SampleBatch
from core.td3_policy_graph import TD3PolicyGraph
from ray import tune
# import pysnooper
# from ray.rllib.evaluation.policy_evaluator import PolicyEvaluator
from core.policy_evaluator import PolicyEvaluator



#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#


# def train(config,reporter):
#     env = gym.make(config["env_name"])
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     max_action = float(env.action_space.high[0])
#     policy = TD3PolicyGraph(state_dim,action_dim,max_action)
#     workers = [
#         PolicyEvaluator.as_remote().remote(lambda c: gym.make("CartPole-v0"),
#                                            CustomPolicy)
#         for _ in range(config["num_workers"])
#     ]


    
# def make_local_evaluator(env_creator, policy_graph):
#     pass
#
# def make_remote_evaluators(env_creator, policy_graph, size):
#     pass
# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=5):
	avg_reward = 0.
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			action = policy.compute_single_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print("---------------------------------------")
	return avg_reward




if __name__ == "__main__":
    ray.init(include_webui=False, ignore_reinit_error=True, object_store_memory=20000000000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=2e3, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=float)
    parser.add_argument("--max_timesteps", default=2e5, type=float)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--pop_size", default=2, type=int)
    parser.add_argument("--crossover_prob", default=0.0, type=float)
    parser.add_argument("--mutation_prob", default=0.9, type=float)
    parser.add_argument("--elite_fraction", default=0.1, type=float)
    parser.add_argument("--node_name", default="qcis5")
    parser.add_argument("--output")
    # parser.add_argument("--max_action")

    args = parser.parse_args()
    # evaluator to produce experiences
    # local_evaluator is learner
    # local_evaluator = make_local_evaluator(env_creator=lambda _: gym.make(args["env_name"]),, policy_graph=TD3PolicyGraph)
    # optimizer to update policy
    env = gym.make(args.env_name)
    # state_dim = env.observation_space
    # action_dim = env.action_space
    max_action = float(env.action_space.high[0])
    # args.max_action = max_action
    config = {"parameter_noise": False,"prioritized_replay_eps":1e-6}


    policy = TD3PolicyGraph(env.observation_space, env.action_space, max_action, config)
    local_evaluator = PolicyEvaluator(env_creator=lambda _: gym.make(args.env_name),
                                      policy_graph=TD3PolicyGraph)

    remote_evaluators = [PolicyEvaluator.as_remote().remote(env_creator=lambda _: gym.make(args.env_name),
                         policy_graph=TD3PolicyGraph)
                         for _ in range(args.pop_size)]

    # optimizer = AsyncReplayOptimizer.make(
    # evaluator_cls = PolicyEvaluator,
    # evaluator_args = {
    # "env_creator": lambda _: gym.make(args["env_name"]),
    # "policy_graph": TD3PolicyGraph,},
    # num_workers = 10)
    # @pysnooper.snoop()
    print("len of remote_evaluators,", len(remote_evaluators))
    optimizer = AsyncReplayOptimizer(local_evaluator, remote_evaluators, buffer_size=2000000, debug=True, train_batch_size=100)
    evaluate_steps = 0

    while optimizer.num_steps_sampled < args.max_timesteps:
        optimizer.step()
        #evalute the agent's score every 5000 steps
        if optimizer.num_steps_sampled // 5000  > evaluate_steps:
            evaluate_steps += 1
            evaluate_policy(policy,eval_episodes=5)
            print("all sample steps,",optimizer.num_steps_sampled)
        




    #
    # tune.run(
    #     train,
    #     resources_per_trial={
    #         "gpu": 1 if args.gpu else 0,
    #         "cpu": 1,
    #         "extra_cpu": args.pop_size,
    #     },
    #     config={
    #         "num_workers": args.pop_size,
    #         "num_iters": args.max_timesteps,
    #         "env_name": args.env_name
    #     },
    # )

