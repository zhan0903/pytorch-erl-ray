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
from ray.rllib.optimizers.async_replay_optimizer import AsyncReplayOptimizer

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#




def train():
    start_timestep = 0
    
def make_local_evaluator():
    pass
    
def make_remote_evaluators():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=2e3, type=int)
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
    parser.add_argument("--pop_size", default=4, type=int)
    parser.add_argument("--crossover_prob", default=0.0, type=float)
    parser.add_argument("--mutation_prob", default=0.9, type=float)
    parser.add_argument("--elite_fraction", default=0.1, type=float)
    parser.add_argument("--node_name", default="qcis5")
    parser.add_argument("--output")
    args = parser.parse_args()
    
    local_evaluator = make_local_evaluator(env_creator, policy_graph)
    remote_evaluator = make_remote_evaluators(env_creator, policy_graph, args["pop_size"])
    
    optimizer = AsyncReplayOptimizer(local_evaluator, remote_evaluator, train_batch_size=500)
    
    while True:
        optimizer.step()