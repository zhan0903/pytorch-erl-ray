import numpy as np
import gym,os, time, sys, random
import argparse
import logging
import ray
import copy
from core import ddpg as ddpg
from core import replay_memory
import torch
from torch.optim import Adam
import torch.nn as nn
from core import mod_utils as utils
from core import mod_neuro_evo as utils_ne




render = False
parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) (Swimmer-v2) (Hopper-v2)', required=True)
env_tag = vars(parser.parse_args())['env']


logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(level=logging.DEBUG)


# # yapf: disable
# # __sphinx_doc_begin__
# DEFAULT_CONFIG = {
#     # No remote workers by default
#     "num_workers": 0,
#     # Learning rate
#     "lr": 0.0004,
#     # Use PyTorch as backend
#     "use_pytorch": False,
# }
# # __sphinx_doc_end__
# # yapf: enable


# def make_session(single_threaded):
#     if not single_threaded:
#         return tf.Session()
#     config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
#     config.gpu_options.allow_growth = True
#     return tf.Session(config=config)


class Parameters:
    def __init__(self):
        self.input_size = None
        self.hidden_size = 36
        self.num_actions = None
        self.learning_rate = 0.1

        #Number of Frames to Run
        if env_tag == 'Hopper-v2': self.num_frames = 4000000
        elif env_tag == 'Ant-v2': self.num_frames = 6000000
        elif env_tag == 'Walker2d-v2': self.num_frames = 8000000
        else: self.num_frames = 2000000

        #USE CUDA
        self.is_cuda = True; self.is_memory_cuda = True

        #Sunchronization Period
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.synch_period = 1
        else: self.synch_period = 10

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
        if env_tag == 'Hopper-v2' or env_tag == 'Reacher-v2': self.num_evals = 5
        elif env_tag == 'Walker2d-v2': self.num_evals = 3
        else: self.num_evals = 1

        #Elitism Rate
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.elite_fraction = 0.3
        elif env_tag == 'Reacher-v2' or env_tag == 'Walker2d-v2': self.elite_fraction = 0.2
        else: self.elite_fraction = 0.1

        self.pop_size = 10
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9

        #Save Results
        self.state_dim = None; self.action_dim = None #Simply instantiate them here, will be initialized later
        self.save_foldername = 'test3-debug/%s/' % env_tag
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)


class OUNoise:
    def __init__(self, action_dimension, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


@ray.remote(num_gpus=0.2)
class Worker(object):
    def __init__(self, args):
        self.env = utils.NormalizedActions(gym.make(env_tag))
        self.args = args
        self.ounoise = OUNoise(args.action_dim)
        # self.sess = make_session(single_threaded=True)
        self.policy = ddpg.Actor(args)
        self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size//args.pop_size)
        self.num_games = 0; self.num_frames = 0; self.gen_frames = 0

    def add_experience(self, state, action, next_state, reward, done):
        reward = utils.to_tensor(np.array([reward])).unsqueeze(0)
        if self.args.is_cuda: reward = reward.cuda()
        if self.args.use_done_mask:
            done = utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
            if self.args.is_cuda: done = done.cuda()
        action = utils.to_tensor(action)
        if self.args.is_cuda: action = action.cuda()
        self.replay_buffer.push(state, action, next_state, reward, done)

    def do_rollout(self, params, store_transition=True):
        fitness = 0
        if params:
            self.policy.load_state_dict(params)
            # self.policy.set_weights(params)

        # todo: rollout in remote functions
        for _ in range(self.args.num_evals):
            fitness += self._rollout()

        # print("evaluate fitness,", fitness/self.args.num_evals)

        # self.policy.learn()
        # fitness_pg = self._rollout()
        # print("evalute, pg fitness,", fitness, fitness_pg)

        return fitness/self.args.num_evals, self.policy.cpu().state_dict(), self.num_frames

    def _rollout(self, is_action_noise=False, store_transition=True):
        total_reward = 0.0
        state = self.env.reset()
        state = utils.to_tensor(state).unsqueeze(0)
        if self.args.is_cuda:
            state = state.cuda()
        done = False

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            action = self.policy.forward(state)
            action.clamp(-1, 1)
            action = utils.to_numpy(action.cpu())
            if is_action_noise: action += self.ounoise.noise()
            # print("come there in evaluate")
            next_state, reward, done, info = self.env.step(action.flatten())  # Simulate one step in environment
            # print("come there in evaluate")
            next_state = utils.to_tensor(next_state).unsqueeze(0)
            if self.args.is_cuda:
                next_state = next_state.cuda()
            total_reward += reward

            if store_transition: self.add_experience(state, action, next_state, reward, done)
            state = next_state
        if store_transition: self.num_games += 1
        # print("come here,total_reward:",total_reward)
        return total_reward

def process_results(results):
    pops = []
    fitness = []
    num_frames = []
    for result in results:
        num_frames.append(result[2])
        pops.append(result[1])
        fitness.append(result[0])
    return fitness, pops, num_frames


if __name__ == "__main__":
    # time_start = time.time()
    num_workers = 10
    parameters = Parameters()
    # tf.enable_eager_execution()

    # Create Env
    env = utils.NormalizedActions(gym.make(env_tag))
    parameters.action_dim = env.action_space.shape[0]
    parameters.num_actions = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]
    parameters.input_size = env.observation_space.shape[0]

    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    ray.init(include_webui=False, ignore_reinit_error=True)
    workers = [Worker.remote(parameters)
               for _ in range(num_workers)]
    pops_new = [None for _ in range(num_workers)]
    print("num_evals,", parameters.num_evals)
    # print(pops_new)
    time_start = time.time()
    while True:
        # parallel pg process
        rollout_ids = [worker.do_rollout.remote(pop_params) for worker, pop_params in zip(workers,pops_new)]
        results = ray.get(rollout_ids)
        all_fitness, pops, num_frames = process_results(results)
        # print("maximum score,", max(all_fitness))
        # print("all num_frames,", sum(num_frames))
        time_evaluate = time.time()-time_start
        time_middle = time.time()
        print("time for evalutation,",time_evaluate)
        pops_new = copy.deepcopy(pops)

        # evolver process
        evolver = utils_ne.SSNE(parameters)
        new_pops = []
        for pop in pops_new:
            new_pop = ddpg.Actor(parameters)
            new_pop.load_state_dict(pop)
            new_pops.append(new_pop)

        elite_index = evolver.epoch(new_pops, all_fitness)
        # print("elite_index,", elite_index)
        time_evolve = time.time()-time_middle
        print("time for evolve,", time_evolve)

        if sum(num_frames) % 44000 == 0:
            print("maximum score,", max(all_fitness))
            print("all num_frames,", sum(num_frames))
            print("time,",time.time()-time_start)
        exit(0)












