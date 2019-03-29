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
import math

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M',
#                     filename='./debug/4_swimmer_debug_logger.log',
#                     filemode='w')
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# console.setFormatter(formatter)
# logging.getLogger('').addHandler(console)
#
# logger_worker = logging.getLogger('Worker')
# logger_main = logging.getLogger('Main')


def select_action(state, actor):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    return actor(state).cpu().data.numpy().flatten()


def evaluate_policy(env, policy, eval_episodes=5):
    # self.set_weights(actor_weights,critic_weights)
    avg_reward = 0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = select_action(np.array(obs), policy)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    logger_main.info("---------------------------------------")
    logger_main.info("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    logger_main.info("---------------------------------------")
    # print("Evaluation over after gradient %f, id %d" % (avg_reward,self.id))
    return avg_reward


@ray.remote(num_gpus=0.5)
class Worker(object):
    def __init__(self, args, id):
        self.env = gym.make(args.env_name)
        self.id = id
        self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename='./debug/%s_%s_%s_%s' % (args.version_name, args.node_name, args.env_name, args.pop_size),
                            filemode='a+')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-4s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        self.logger_worker = logging.getLogger('Worker')

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.policy = ddpg.DDPG(state_dim, action_dim, max_action)
        self.actor_evovlved = ddpg.ActorErl(state_dim, action_dim)
        self.better_actor = ddpg.ActorErl(state_dim, action_dim)
        self.replay_buffer = utils.ReplayBuffer()

        self.args = args
        self.total_timesteps = 0
        self.episode_num = 0
        self.timesteps_since_eval = 0
        self.episode_timesteps = 0
        self.training_times = 0
        self.better_reward = -math.inf

    def set_weights(self, actor_weights, critic_weights):
        if actor_weights is not None:
            self.policy.actor.load_state_dict(actor_weights)
        self.policy.critic.load_state_dict(critic_weights)

        for param, target_param in zip(self.policy.critic.parameters(), self.policy.critic_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

        if actor_weights is not None:
            for param, target_param in zip(self.policy.actor.parameters(), self.policy.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    # Runs policy for X episodes and returns average reward
    def evaluate_policy_temp(self, eval_episodes=1):
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
        # print("Evaluation over after gradient %f, id %d" % (avg_reward,self.id))
        return avg_reward

    # def select_action(self, state, actor):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    #     return actor(state).cpu().data.numpy().flatten()

    def evaluate_policy(self, actor):
        obs = self.env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = select_action(np.array(obs), actor)

            # Perform action
            new_obs, reward, done, _ = self.env.step(action)
            done_bool = 0 if self.episode_timesteps + 1 == self.env._max_episode_steps else float(done)
            episode_reward += reward

            # Store data in replay buffer
            self.replay_buffer.add((obs, new_obs, action, reward, done_bool))
            obs = new_obs

            self.episode_timesteps += 1
            self.total_timesteps += 1

        return episode_reward

    def train(self, actor_weights, critic_weights, evolve=True, train=True):
        self.episode_timesteps = 0
        # done = False
        if not evolve:
            self.set_weights(actor_weights, critic_weights)
        else:
            self.actor_evovlved.load_state_dict(actor_weights)

        self.logger_worker.info("After Evolved: ID: {0},net_weight:{1}".
                                format(self.id, self.policy.actor.state_dict()["w_out.weight"][-1][:5]))

        if evolve:
            reward_evolved = self.evaluate_policy(self.actor_evovlved)
            self.episode_num += 1
        else:
            reward_evolved = -math.inf

        if train:
            if len(self.replay_buffer.storage) == 0:
                iteration = 0
            else:
                iteration = 1000
            self.policy.train(self.replay_buffer, iteration, self.args.batch_size, self.args.discount, self.args.tau)
            if self.episode_timesteps:
                self.training_times += 1
            reward_learned = self.evaluate_policy(self.policy.actor)
            self.episode_num += 1
        else:
            reward_learned = -math.inf
        # reward_learned = self.evaluate_policy(self.policy.actor)
        # self.episode_num += 1

        self.logger_worker.info("After Learned: ID: {0},net_weight:{1}".
                                format(self.id, self.policy.actor.state_dict()["w_out.weight"][-1][:5]))

        self.logger_worker.info("ID: %d Total T: %d  Training_times: %d  Episode_Num: %d Episode T: "
                                "%d reward_evolved: %f  reward_learned: %f" %
                                (self.id, self.total_timesteps, self.training_times, self.episode_num,
                                 self.episode_timesteps, reward_evolved, reward_learned))

        if reward_evolved > reward_learned:
            return self.total_timesteps, self.policy.grads_critic, reward_evolved, reward_learned, reward_evolved, None
        else:
            return self.total_timesteps, self.policy.grads_critic, reward_evolved, reward_learned, \
                   reward_learned, self.policy.actor.state_dict()


def process_results(r):
    total_t = []
    grads_c = []
    all_f = []
    all_f_a = []
    all_rewards = []
    all_new_pop = []

    for result in r:
        all_new_pop.append(result[5])
        all_rewards.append(result[4])
        all_f_a.append(result[3])
        all_f.append(result[2])
        grads_c.append(result[1])
        total_t.append(result[0])
    return sum(total_t), grads_c, all_f, all_f_a, all_rewards, all_new_pop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="OurDDPG")
    parser.add_argument("--env_name", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=1e5, type=int)
    parser.add_argument("--eval_freq", default=4e4, type=float)
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
    parser.add_argument("--version_name")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='./debug/%s_%s_%s_%s' % (args.version_name, args.node_name, args.env_name, args.pop_size),
                        filemode='a+')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-4s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logger_main = logging.getLogger('Main')

    evolver = utils_ne.SSNE(args)
    file_name = "%s_%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed), args.node_name)
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

    # policy = ddpg.DDPG(state_dim, action_dim, max_action)
    agent = ddpg.PERL(state_dim, action_dim, max_action, args.pop_size)
    ray.init(include_webui=False, ignore_reinit_error=True, object_store_memory=30000000000)

    workers = [Worker.remote(args, i)
               for i in range(args.pop_size)]

    all_timesteps = 0
    timesteps_since_eval = 0

    evaluations = []
    time_start = time.time()

    episode = 0
    evolve = False
    train = True
    actors = [actor.state_dict() for actor in agent.actors]
    average = None
    get_value = True
    value = 0
    MaxValue = None
    maxvalue = None
    evolve_count = 0
    gradient_count = 0
    generation = 0

    logger_main.info("************************************************************************************")
    logger_main.info("3276, choose better one, evolve for exploration+every 5 generation, ga for exploitation")
    logger_main.info("************************************************************************************")

    while all_timesteps < args.max_timesteps:
        critic_id = ray.put(agent.critic.state_dict())
        evolve_id = ray.put(evolve)
        train_id = ray.put(train)
        train_id = [worker.train.remote(actor, critic_id, evolve_id, train_id) for worker, actor in zip(workers, actors)] # actor.state_dict()
        results = ray.get(train_id)
        all_timesteps, grads_critic, all_reward_evolved, all_reward_learned, rewards, new_pop = process_results(results)
        # logger_main.info("grads_critic:{}".format(grads_critic))
        if grads_critic[0]:
            agent.apply_grads(grads_critic, logger_main)
        generation += 1

        logger_main.info("generation:{}".format(generation))
        for new_actor, actor in zip(new_pop, agent.actors):
            if new_actor is not None:
                actor.load_state_dict(new_actor)

        # logger_main.info("#Max:{0}, #All_TimeSteps:{1}, #Time:{2},".
        #                  format(max(rewards), all_timesteps, (time.time()-time_start)))
        # logger_main.info("#rewards:{}".format(rewards))

        average_evolved = sum(all_reward_evolved)/args.pop_size
        average_learned = sum(all_reward_learned)/args.pop_size

        logger_main.info("#Max:{0}, #All_TimeSteps:{1}, #Average_evolved:{2},#Average_learned:{3} ##Time:{4},".
                         format(max(rewards), all_timesteps, average_evolved, average_learned, (time.time() - time_start)))
        logger_main.info("#rewards:{}".format(rewards))

        # if 8e4 <= all_timesteps <= 1.2e5:
        #     if average_evolved > average_learned:
        #         evolve_count += 1
        #     else:
        #         gradient_count += 1

        logger_main.info("evolve_count:{0}, gradient_count:{1}".format(evolve_count, gradient_count))

        # if all_timesteps > 1.2e5:
        #     if evolve_count > gradient_count:
        #         evolve = True
        #         train = False
        #     else:
        #         evolve = False
        #         train = True

        if get_value:
            value = results[0][0]
            get_value = False

        timesteps_since_eval += value * args.pop_size

        if MaxValue is None:
            MaxValue = max(rewards)
        else:
            if MaxValue < max(rewards):
                MaxValue = max(rewards)

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            champ_index = rewards.index(max(rewards))
            logger_main.debug("champ_index in evaluate:{}".format(champ_index))
            actor_input = ddpg.ActorErl(state_dim, action_dim)

            if new_pop[champ_index] is None:
                actor_input.load_state_dict(agent.actors[champ_index].state_dict())
            else:
                actor_input.load_state_dict(new_pop[champ_index])

            evaluations.append(evaluate_policy(env, actor_input, eval_episodes=5))
            np.save("./results/%s" % file_name, evaluations)

        # if evolve:
        #     if all_timesteps > 1.2e5:
        #         evolver.epoch(agent.actors, rewards)
        #     else:

        if generation >= 5:
            evolver.explore(agent.actors, rewards)
            generation %= 5
            evolve = True
            actors = [actor.state_dict() for actor in agent.actors]
        else:
            if evolve:
                actors = [actor.state_dict() for actor in agent.actors]
                evolve = False
            else:
                actors = [None for _ in range(args.pop_size)]
    logger_main.info("Finish! MaxValue:{}".format(MaxValue))










