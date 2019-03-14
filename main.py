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




# render = False
# parser = argparse.ArgumentParser()
# parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) (Swimmer-v2) (Hopper-v2)', required=True)
# env_tag = vars(parser.parse_args())['env']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.save_models = False
        self.eval_freq = 5e3

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
        self.gamma = 0.99; self.tau = 0.005
        self.seed = 7
        self.batch_size = 100
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
        # self.env = utils.NormalizedActions(gym.make(env_tag))
        self.env = gym.make(args.env_name)
        self.env.seed(args.seed)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.policy = ddpg.DDPG(state_dim, action_dim, max_action)
        self.replay_buffer = utils.ReplayBuffer()

        self.args = args
        self.total_timesteps = 0
        self.episode_num = 0
        self.timesteps_since_eval = 0

        self.num_games = 0; self.num_frames = 0; self.gen_frames = 0

    def set_weights(self,actor_weights,critic_weights):
        self.policy.actor.load_state_dict(actor_weights)
        self.policy.critic.load_state_dict(critic_weights)

        for param, target_param in zip(self.policy.critic.parameters(), self.policy.critic_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

        for param, target_param in zip(self.policy.actor.parameters(), self.policy.actor_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)



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


    def train(self,actor_weights,critic_weights):
        self.set_weights(actor_weights,critic_weights)
        done = False
        episode_timesteps = 0
        episode_reward = 0
        obs = self.env.reset()
        while True:
            if done:
                # if self.total_timesteps != 0:
                print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (self.total_timesteps, self.episode_num, episode_timesteps, episode_reward))
                self.policy.train(self.replay_buffer, episode_timesteps, self.args.batch_size, self.args.discount, self.args.tau)
                self.episode_num += 1
                break
            # Select action randomly or according to policy
            if self.total_timesteps < args.start_timesteps:
                action = self.env.action_space.sample()
            else:
                action = self.policy.select_action(np.array(obs))
                if args.expl_noise != 0:
                    action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                        env.action_space.low, env.action_space.high)
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
    total_timesteps = []
    grads_critic = []
    grads_actor = []
    for result in results:
        grads_critic.append(result[2])
        grads_actor.append(result[1])
        total_timesteps.append(result[0])
    return sum(total_timesteps), grads_actor, grads_critic


def apply_grads(net,grads_actor,grads_critic):
    net.critic_optimizer.zero_grad()
    grads_sum_actor = copy.deepcopy(grads_actor[-1])
    for grad in grads_actor[:-1]:
        for temp_itme, grad_item in zip(grads_sum_actor, grad):
             if grad_item is not None:
                  temp_itme += grad_item
    for g, p in zip(grads_sum_actor, net.actor.parameters()):
        if g is not None:
            p.grad = torch.from_numpy(g).to(device)

    net.critic_optimizer.step()

    net.actor_optimizer.zero_grad()
    grads_sum_critic = copy.deepcopy(grads_actor[-1])
    for grad in grads_critic[:-1]:
        for temp_itme, grad_item in zip(grads_sum_critic, grad):
             if grad_item is not None:
                  temp_itme += grad_item

    for g, p in zip(grads_sum_critic, net.critic.parameters()):
        if g is not None:
            p.grad = torch.from_numpy(g).to(device)
    net.actor_optimizer.step()


if __name__ == "__main__":
    # time_start = time.time()
    num_workers = 1
    # parameters = Parameters()
    # # device = "cuda" # if args.cuda else "cpu"
    # # tf.enable_eager_execution()
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

    # pops_new = []
    # for _ in range(parameters.pop_size):
    #     pops_new.append(ddpg.Actor(parameters))


    # create ray workers
    ray.init(include_webui=False, ignore_reinit_error=True)
    workers = [Worker.remote(args)
               for _ in range(num_workers+1)]
    # time_start = time.time()
    # grads_sum = None
    #
    # total_timesteps = 0
    # timesteps_since_eval = 0
    # episode_num = 0
    # done = True

    # Evaluate untrained policy
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
        apply_grads(policy,grads_actor, grads_critic)

    # Final evaluation
    evaluations.append(ray.get(workers[-1].evaluate_policy.remote(policy.actor.state_dict(),policy.critic.state_dict())))
    if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)












