import gym
import numpy as np
import random


class FiniteBot:
    def __init__(self):
        # self.Q = np.random.random((3, 2, 2)) * 50
        self.visits = np.zeros((3, 2, 2))
        self.Q = np.zeros((3, 2, 2))

    def print_q(self):
        for pos, vel in np.ndindex(2, 2):
            print(f'pos: {pos}, vel: {vel}, action: {np.argmax(self.Q[:, pos, vel])}')

        for pos, vel, action in np.ndindex(2, 2, 3):
            print(f'pos: {pos}, vel: {vel}, action: {action}:', self.Q[action, pos, vel], self.visits[action, pos, vel])

    def take_action(self, pos, vel, eps=0.05, diagnostic=False):
        if diagnostic:
            self.print_q()
        if random.random() > eps:
            values = self.Q[:, pos, vel]
            return np.argmax(values)
        else:
            return random.randint(0, 2)

    def update_Q(self, sample, alpha=0.001):
        for rec in sample:
            ind = tuple(map(int, rec[:-1]))
            self.visits[ind] += 1
            rew = rec[-1]
            alpha = 0.001
            self.Q[ind] += alpha * (rew - self.Q[ind])
            #self.Q[ind] += 1 / self.visits[ind] * (rew - self.Q[ind])


def process_observation(obs):
    return int(obs[0] > -0.508), int(obs[1] > 0)


class StatCollector:
    def __init__(self, qspace_shape, levels, discount):
        self.qspace_shape = qspace_shape
        self.games = []
        self.game_data = []
        self.last_observation = 0

        self.LEVELS = levels
        self.level_beaten = [0 for _ in range(len(levels))]
        self.DISCOUNT = discount
        self.game_level_achieved = [0]
        self.level = 0
        self.game = 0

    def one_step(self, reward, action, obs_raw):
        # returns artificial 'done'
        observation = process_observation(obs_raw)
        self.game_data.append([reward, action, *self.last_observation])
        self.last_observation = observation
        pos_raw = obs_raw[0]
        return pos_raw - (-0.5) > self.LEVELS[self.level]

    def update_level(self):
        # game has been won
        self.level_beaten[self.level] += 1

        if self.level_beaten[self.level] == 10 and self.level < len(self.LEVELS) - 1:
            self.level += 1
            self.game_level_achieved.append(self.game)

    def print_level_stats(self):
        print('|' * 20)
        print('Current Level: ', self.level)
        print(self.level_beaten)
        print('|' * 20)

    def finish_episode(self, t):
        game_data = np.array(self.game_data)

        game_data[:, 0] /= 100
        if t != 199:
            # game has been won
            self.update_level()
            game_data[-1, 0] = 10 ** 2 * (self.level + 1)
        if t == 199:
            # assume it will never be beaten
            game_data[-1, 0] = -0.01 * (1 / (1 - self.DISCOUNT))

        # propagate rewards
        for i in reversed(range(t)):
            game_data[i, 0] += self.DISCOUNT * game_data[i + 1, 0]
        self.games.append(game_data)
        self.game_data = []
        self.game += 1
        return game_data

    def add_starting_observation(self, observation):
        self.last_observation = observation


def first_observation_rewards(qspace_shape, game_data):
    first_obs_rews = list()
    for q in np.ndindex(qspace_shape):
        q_inds = np.where((game_data[:, 1:] == q).all(axis=1, ))[0]
        if q_inds.shape[0] > 0:
            first_obs_ind = q_inds.min()
            first_obs_rew = game_data[first_obs_ind, 0]
            first_obs_rews.append([*q, first_obs_rew])
    return first_obs_rews


def every_observation_rewards(game_data):
    return game_data[:, np.roll(range(game_data.shape[1]), -1)]


qspace_shape = (3, 2, 2)
statCollector = StatCollector(qspace_shape,
                              [0.2, 0.4, 0.6, 0.65, 100],
                              0.99
                              )
env = gym.make('MountainCar-v0')
num_episodes = 30000

bot = FiniteBot()

starting_episodes = 300

first_obs_reward = []
for i_episode in range(num_episodes):
    observation = process_observation(env.reset())
    pos, vel = observation
    statCollector.add_starting_observation(observation)

    for t in range(200):
        diagnostic = False
        if i_episode % 200 == 0:
            env.render()
            pass
        if i_episode % 200 == 0 and t == 0:
            print(i_episode)
            statCollector.print_level_stats()
            diagnostic = True
        # select action to make
        if i_episode < starting_episodes:
            eps = 0.6
        elif i_episode < starting_episodes * 3:
            eps = 0.4
        elif i_episode < starting_episodes * 5:
            eps = 0.1
        else:
            eps = 0.01

        action = bot.take_action(pos, vel,
                                 eps=eps,
                                 diagnostic=diagnostic
                                 )

        # make action\
        observation, reward, done, info = env.step(action)
        pos, vel = process_observation(observation)
        done |= statCollector.one_step(reward, action, observation)

        # check if beaten current level
        if done:
            game_data = statCollector.finish_episode(t)
            if i_episode <= starting_episodes:
                first_obs_reward.append(first_observation_rewards(qspace_shape, game_data))
                if i_episode == starting_episodes:
                    bot.update_Q(np.vstack(first_obs_reward), 0.5)
            elif i_episode > starting_episodes:
                bot.update_Q(first_observation_rewards(qspace_shape, game_data), 1 / (i_episode * 20))
            break
