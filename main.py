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


def test_value_rollout():
    pass


@ray.remote(num_gpus=0.1)
class Worker(object):
    def __init__(self, args):
        self.env = utils.NormalizedActions(gym.make(env_tag))
        self.env.seed(args.seed)
        self.args = args
        self.ounoise = OUNoise(args.action_dim)
        # self.sess = make_session(single_threaded=True)
        self.actor = ddpg.Actor(args, init=True)
        self.actor_target = ddpg.Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr=0.5e-4)

        self.critic = ddpg.Critic(args)
        self.critic_target = ddpg.Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.5e-3)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        ddpg.hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        ddpg.hard_update(self.critic_target, self.critic)

        self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size//10)
        self.num_games = 0; self.num_frames = 0; self.gen_frames = 0

    def compute_gradients(self, actor_params, gcritic_params):
        self.actor.load_state_dict(actor_params)
        self.critic.load_state_dict(gcritic_params)
        # ddpg.soft_update(self.actor_target, self.actor, self.tau)-failed
        ddpg.hard_update(self.actor_target, self.actor)
        ddpg.hard_update(self.critic_target, self.critic)

        self.gen_frames = 0
        avg_fitness = self.do_rollout()
        for _ in range(int(2*self.gen_frames*self.args.frac_frames_train)):
            # print("gen_frames,", self.gen_frames)
            # print("size of replay_buff,",len(self.replay_buffer))
            transitions = self.replay_buffer.sample(self.args.batch_size)
            batch = replay_memory.Transition(*zip(*transitions))
            self.update_params(batch)

        grads = [param.grad.data.cpu().numpy() if param.grad is not None else None
                 for param in self.critic.parameters()]

        # grads = 0

        value_after_gradient = self.do_rollout()
        print("(avg_fitness, value_after_gradient),", avg_fitness, value_after_gradient)

        if value_after_gradient < avg_fitness:
            return grads, self.actor_target.state_dict(), avg_fitness, self.num_frames #, (avg_fitness, value_after_gradient)

        return grads, self.actor.state_dict(), value_after_gradient, self.num_frames #, (avg_fitness, value_after_gradient)

    def update_params(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        if self.args.use_done_mask: done_batch = torch.cat(batch.done)
        # state_batch.volatile = False
        # next_state_batch.volatile = True
        # action_batch.volatile = False

        # Load everything to GPU if not already
        if self.args.is_memory_cuda and not self.args.is_cuda:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()
            self.critic.cuda()
            state_batch = state_batch.cuda()
            next_state_batch = next_state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            if self.args.use_done_mask:
                done_batch = done_batch.cuda()

        #Critic Update
        next_action_batch = self.actor_target.forward(next_state_batch)
        next_q = self.critic_target.forward(next_state_batch, next_action_batch)
        if self.args.use_done_mask: next_q = next_q * (1 - done_batch.float()) #Done mask
        target_q = reward_batch + (self.gamma * next_q)

        self.critic_optim.zero_grad()
        current_q = self.critic.forward((state_batch), (action_batch))
        dt = self.loss(current_q, target_q)
        dt.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        # Actor Update
        self.actor_optim.zero_grad()
        policy_loss = -self.critic.forward((state_batch), self.actor.forward((state_batch)))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optim.step()

        # ddpg.soft_update(self.actor_target, self.actor, self.tau)
        # ddpg.soft_update(self.critic_target, self.critic, self.tau)

    def add_experience(self, state, action, next_state, reward, done):
        reward = utils.to_tensor(np.array([reward])).unsqueeze(0)
        if self.args.is_cuda: reward = reward.cuda()
        if self.args.use_done_mask:
            done = utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
            if self.args.is_cuda: done = done.cuda()
        action = utils.to_tensor(action)
        if self.args.is_cuda: action = action.cuda()
        self.replay_buffer.push(state, action, next_state, reward, done)

    def do_rollout(self, store_transition=True):
        fitness = 0
        # todo: rollout in remote functions
        for _ in range(self.args.num_evals):
            fitness += self._rollout(store_transition=store_transition)

        return fitness/self.args.num_evals

    def do_test(self, params, store_transition=False):
        fitness = 0
        self.actor.load_state_dict(params)
        for _ in range(5):
            fitness += self._rollout(store_transition=store_transition)
        return fitness/5.0

    def _rollout(self, is_action_noise=False, store_transition=True):
        total_reward = 0.0
        state = self.env.reset()
        state = utils.to_tensor(state).unsqueeze(0)
        if self.args.is_cuda:
            state = state.cuda()
        done = False

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            action = self.actor.forward(state)
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
            # print("action,",action)

            state = next_state
        if store_transition: self.num_games += 1
        # print("come here,total_reward:",total_reward)
        return total_reward


def process_results(results):
    pops = []
    fitness = []
    num_frames = []
    grads = []
    # fitness_after_gradient = []
    for result in results:
        # fitness_after_gradient.append([result[4]])
        num_frames.append(result[3])
        fitness.append(result[2])
        pops.append(result[1])
        grads.append(result[0])
    return grads, pops, fitness, num_frames


if __name__ == "__main__":
    # time_start = time.time()
    num_workers = 10
    parameters = Parameters()
    device = "cuda" # if args.cuda else "cpu"
    # tf.enable_eager_execution()

    # Create Env
    env = utils.NormalizedActions(gym.make(env_tag))
    parameters.action_dim = env.action_space.shape[0]
    parameters.num_actions = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]
    parameters.input_size = env.observation_space.shape[0]

    # env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)
    evolver = utils_ne.SSNE(parameters)
    # print("random,",random.randint(0,10))

    # pops_new = []
    # for _ in range(parameters.pop_size):
    #     pops_new.append(ddpg.Actor(parameters))

    gcritic = ddpg.Critic(parameters)
    gcritic_target = ddpg.Critic(parameters)
    gcritic_optim = Adam(gcritic.parameters(), lr=0.5e-3)

    pops_new = []
    for _ in range(parameters.pop_size):
        pops_new.append(ddpg.Actor(parameters))

    ray.init(include_webui=False, ignore_reinit_error=True)
    workers = [Worker.remote(parameters)
               for _ in range(num_workers+1)]

    time_start = time.time()
    grads_sum = None
    frames_sum = 0

    while frames_sum <= 1e6:
        # time_start = time.time()
        rollout_ids = [worker.compute_gradients.remote(pop_params.state_dict(), gcritic.state_dict()) for worker, pop_params in zip(workers[:-1], pops_new)]
        results = ray.get(rollout_ids)
        grads, actors, avg_fitness, num_frames = process_results(results)
        best_train_fitness = max(avg_fitness)
        champ_index = avg_fitness.index(max(avg_fitness))
        frames_sum = sum(num_frames)
        print("best_train_fitness,", best_train_fitness)

        # grads_sum = copy.deepcopy(grads[-1])
        gcritic_optim.zero_grad()
        for grad in grads:
            # for temp_itme, grad_item in zip(grads_sum, grad):
            #      if grad_item is not None:
            #           temp_itme += grad_item
            for g, p in zip(grad, gcritic.parameters()):
                if g is not None:
                    p.grad = torch.from_numpy(g).to(device)
            nn.utils.clip_grad_norm_(gcritic.parameters(), 10)
            gcritic_optim.step()

        # for param, grad in zip(gcritic.parameters(), grads_sum):
        #     param.grad = torch.FloatTensor(grad).to(device)
        #
        # nn.utils.clip_grad_norm_(gcritic.parameters(), 10)
        # gcritic_optim.step()

        pops_new = []
        for pop in actors:
            new_pop = ddpg.Actor(parameters)
            new_pop.load_state_dict(pop)
            pops_new.append(new_pop)

        elite_index = evolver.epoch(pops_new, avg_fitness)

        if sum(num_frames) % 40000 == 0:
            test_score_id = workers[-1].do_test.remote(pops_new[champ_index].state_dict())
            test_score = ray.get(test_score_id)
            print("#Max score:", best_train_fitness,"#Test score,",test_score,"#Frames:",frames_sum, "Time:",(time.time()-time_start))
        # # exit(0)
        # if test:
        #     test = False
        #     continue
        # else:
        #     break













