import gym
import numpy as np

from sklearn.exceptions import NotFittedError
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


class Bot:
    def __init__(self):
        self.num_features = 4
        self.model = self.nn(self.num_features)

    def nn(self, num_features):
        model = Sequential()
        model.add(Dense(12, input_dim=num_features, activation='relu'))
        model.add(Dropout(0.65))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.65))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1))
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])
        return model

    def train_batch(self, x_batch, y_batch):
        self.model.train_on_batch(x_batch, y_batch)

    def fit(self, x, y, epochs=15):
        self.model.fit(x, y, epochs=epochs, batch_size=100)

    def predict(self, x_batch):
        return self.model.predict(x_batch)


def take_action(bot, observation, action_space, eps=0.05, diagnostic=False):
    try:
        if random.random() > eps:

            x_batch = np.array([observation * (-1 if i == 0 else 1) for i in range(action_space.n)])
            values = bot.predict(x_batch)
            if diagnostic:
                print(values)
            return np.argmax(values)
        else:
            return action_space.sample()
    except NotFittedError:
        return action_space.sample()


# np.random.seed(1337)

env = gym.make('CartPole-v0')
DISCOUNT = 0.99
episode_reward = []
bot = Bot()
moves = []
num_episodes = 5

num_games_per_episode = 500
data = []
for i_episode in range(num_episodes):
    ep_data = np.empty(shape=(0, env.observation_space.shape[0] + 1))
    for i_game in range(num_games_per_episode):

        observation = env.reset()

        observations_indexes = [0]
        moves_in_game = []

        ep_samples = 80
        game_data = np.empty(shape=(0, env.observation_space.shape[0] + 1))
        # Start Episode
        if i_game % 1000 == 0:
            print(i_game)

        for t in range(500):
            if i_game % 1000 == 0:
                env.render()
                pass

            action = take_action(bot, observation, env.action_space,
                                 eps=max(0.5 - i_episode / num_episodes, 0),
                                 diagnostic=(i_game % 100 == 0) and (t % 25 == 0)
                                 )
            observation2, reward, done, info = env.step(action)
            action_encoded = -1 if action == 0 else 1
            observation *= action_encoded
            row = np.append((reward, ), observation)
            moves_in_game.append(action)
            game_data = np.vstack((game_data, [row]))

            observation = observation2
            if done:
                # claim the rewards
                for i in reversed(range(t)):
                    game_data[i, 0] = game_data[i, 0] + DISCOUNT * game_data[i + 1, 0]
                episode_reward.append(game_data.shape[0])
                moves.append(np.mean(moves_in_game))
                ep_data = np.vstack((ep_data, game_data))
                break
        """
        if i_episode % 100 == 0 and i_episode != 0:
            # move through last batch
            train_data = data[-100:]
            bot.train_batch(train_data[:, 1:], train_data[:, 0])
        
        """
    print(np.mean(episode_reward))
    data.append(ep_data)
    train_data = np.vstack(data[-3:])
    epoch_num = 10 if i_episode == 1 else 5
    bot.fit(train_data[:, 1:], train_data[:, 0], epochs=epoch_num)
env.close()

# for bot in bots:
#    bot.close()
import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.plot(episode_reward)

plt.subplot(2, 1, 2)
plt.plot(moves)
plt.show()
