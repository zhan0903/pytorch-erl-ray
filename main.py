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


@ray.remote(num_gpus=0.2)
class Worker(object):
    def __init__(self, args, id):
        # self.env = utils.NormalizedActions(gym.make(env_tag))
        self.env = gym.make(args.env_name)
        self.id = id
        self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.policy = ddpg.DDPG(state_dim, action_dim, max_action)
        # print("in worker init critic,", self.policy.critic.state_dict()["l3.bias"])

        self.replay_buffer = utils.ReplayBuffer()

        self.args = args
        self.total_timesteps = 0
        self.episode_num = 0
        self.timesteps_since_eval = 0

    # def init_nets(self, actor_weight_init, critic_weight_init):
    #     self.policy.critic.load_state_dict(critic_weight_init)
    #     self.policy.critic_target.load_state_dict(self.policy.critic.state_dict())
    #
    #     self.policy.actor.load_state_dict(actor_weight_init)
    #     self.policy.actor_target.load_state_dict(self.policy.actor.state_dict())
    #
    #     return 1

    def set_weights(self,actor_weights, critic_weights):
        if actor_weights is not None:
            # print("come here 1")
            self.policy.actor.load_state_dict(actor_weights)
        self.policy.critic.load_state_dict(critic_weights)
        # self.policy.critic.zero_grad()

        for param, target_param in zip(self.policy.critic.parameters(), self.policy.critic_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

        if actor_weights is not None:
            # print("come here 2")
            for param, target_param in zip(self.policy.actor.parameters(), self.policy.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    # Runs policy for X episodes and returns average reward
    def evaluate_policy(self, eval_episodes=1):
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
        print("Evaluation over after gradient %f, id %d" % (avg_reward,self.id))
        return avg_reward

    def train(self, actor_weights, critic_weights):
        self.set_weights(actor_weights, critic_weights)
        # print("set_weight self.policy.critic,id", self.policy.critic.state_dict()["l3.bias"],self.id)
        print("set_weight self.policy.actor,id", self.policy.actor.state_dict()["l3.bias"],self.id)


        done = False
        episode_timesteps = 0
        episode_reward = 0
        obs = self.env.reset()

        # pop_reward = self.evaluate_policy(self.actor)

        while True:
            if done:
                self.episode_num += 1
                if self.total_timesteps != 0:
                    print("ID: %d Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (self.id, self.total_timesteps, self.episode_num, episode_timesteps, episode_reward))
                    self.policy.train(self.replay_buffer, episode_timesteps, self.args.batch_size, self.args.discount, self.args.tau)
                    pop_reward_after = self.evaluate_policy()

                    # print("before self.policy.critic,id,", self.policy.critic.state_dict()["l3.bias"], self.id)
                    print("before self.policy.actor,id,", self.policy.actor.state_dict()["l3.bias"], self.id)

                    if pop_reward_after > episode_reward:
                        return self.total_timesteps, self.policy.grads_critic, pop_reward_after, self.id, self.policy.actor.state_dict()
                    else:
                        return self.total_timesteps, self.policy.grads_critic, episode_reward, self.id, None

                # Reset environment test on child process
                # obs = self.env.reset()
                # done = False
                # episode_reward = 0
                # episode_timesteps = 0
                # self.episode_num += 1

            # Select action randomly or according to policy
            action = self.policy.select_action(np.array(obs))

            # if self.total_timesteps < args.start_timesteps:
            #     action = self.env.action_space.sample()
            # else:
            #     action = self.policy.select_action(np.array(obs))
            #     if args.expl_noise != 0:
            #         action = (action + np.random.normal(0, args.expl_noise, size=self.env.action_space.shape[0])).clip(self.env.action_space.low, self.env.action_space.high)
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

        # print("before self.policy.critic,id,", self.policy.critic.state_dict()["l3.bias"],self.id)
        # print("before self.policy.actor,id,", self.policy.actor.state_dict()["l3.bias"],self.id)
        #
        # # return self.policy.critic.cpu().state_dict()["l3.bias"], self.policy_debug.critic.cpu().state_dict()["l3.bias"]
        # return self.total_timesteps, self.policy.grads_critic, episode_reward, self.id,


def process_results(r):
    total_t = []
    grads_c = []
    all_f = []
    all_id = []
    new_pop = []
    for result in r:
        new_pop.append(result[4])
        all_id.append(result[3])
        all_f.append(result[2])
        grads_c.append(result[1])
        total_t.append(result[0])
    return sum(total_t), grads_c, all_f,all_id, new_pop


if __name__ == "__main__":
    # num_workers = 3
    # parameters = Parameters()
    # evolver = utils_ne.SSNE(parameters)

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
    parser.add_argument("--pop_size", default=5, type=int)
    parser.add_argument("--crossover_prob", default=0.0, type=float)
    parser.add_argument("--mutation_prob", default=0.9, type=float)
    parser.add_argument("--elite_fraction", default=0.1, type=float)

    args = parser.parse_args()

    evolver = utils_ne.SSNE(args)


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

    # policy = ddpg.DDPG(state_dim, action_dim, max_action)
    agent = ddpg.PERL(state_dim, action_dim, max_action, args.pop_size)

    print("in main policy,", agent.critic.state_dict()["l3.bias"])

    ray.init(include_webui=False, ignore_reinit_error=True,object_store_memory=30000000000)

    workers = [Worker.remote(args, i)
               for i in range(args.pop_size+1)]

    # evaluations = [ray.get(workers[-1].evaluate_policy.remote(policy.actor.state_dict(),policy.critic.state_dict()))]
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_timesteps = 0
    done = True
    time_start = time.time()
    debug = True
    episode = 0

    while total_timesteps < args.max_timesteps:
        # if debug:
        #     actor_weight = actors[0].state_dict()
        # else:
        #     actor_weight = None
        critic_id = ray.put(agent.critic.state_dict())
        train_id = [worker.train.remote(actor.state_dict(), critic_id) for worker, actor in zip(workers[:-1], agent.actors)] # actor.state_dict()
        results = ray.get(train_id)
        total_timesteps, grads_critic, all_fitness, all_id, new_pop = process_results(results)
        agent.apply_grads(grads_critic)
        print(time.time()-time_start)
        print("max value,", max(all_fitness))
        print("ids,",all_id)
        episode += 1
        # debug = False
        # print("after apply_grads self.policy.critic,", agent.critic.state_dict()["l3.bias"])
        # if episode // 3 == 0:
        for actor, pop in zip(agent.actors, new_pop):
            if pop is not None:
                actor.load_state_dict(pop)

        print("before evolve actor 0,", agent.actors[0].state_dict()["l3.weights"][:10])
        print("before evolve actor 1,", agent.actors[1].state_dict()["l3.weights"][:10])
        print("before evolve actor 2,", agent.actors[2].state_dict()["l3.weights"][:10])
        print("before evolve actor 3,", agent.actors[3].state_dict()["l3.weights"][:10])
        print("before evolve actor 4,", agent.actors[4].state_dict()["l3.weights"][:10])

        elite_index = evolver.epoch(agent.actors, all_fitness)
        print("actor 0,",agent.actors[0].state_dict()["l3.weights"][:10])
        print("actor 1,", agent.actors[1].state_dict()["l3.weights"][:10])
        print("actor 2,", agent.actors[2].state_dict()["l3.weights"][:10])
        print("actor 3,", agent.actors[3].state_dict()["l3.weights"][:10])
        print("actor 4,", agent.actors[4].state_dict()["l3.weights"][:10])
        #     print("elite_index,",elite_index)
        # else:

        # exit(0)
    # Final evaluation
    # evaluations.append(ray.get(workers[-1].evaluate_policy.remote(policy.actor.state_dict(),policy.critic.state_dict())))
    # print("done")
    # exit(0)
    # if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
    # np.save("./results/%s" % (file_name), evaluations)












