import gym
import matplotlib.pyplot as plt
import time
import numpy as np
from skimage.measure import block_reduce
from collections import deque

import pandas as pd

import random

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import os

from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, value):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(value)
        else:
            self.memory[self.position] = value
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def whole_buffer(self):
        return self.memory[self.position:] + self.memory[:self.position]


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from DQN.DQN1 import DQN1
from DQN.DQN2 import DQN2

import timeit

import copy

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)

# create model
behavior_model = DQN2(learning_rate=0.0001, DISCOUNT=0.9)
target_model = DQN2(learning_rate=0.00000001, DISCOUNT=0.9)

# su_model = SingleUpdate(model)


MINIBATCH_SIZE = 32

EPS = 0.05
EXP_EPISODES = 3000
DISCOUNT = 0.9
ER_SIZE = 1000000
CUR_STATE_SIZE = 4

UPDATES_PER_EPOCH = 50000

EP_LENGTH = 100000

ER = ReplayMemory(ER_SIZE)

cur_state = ReplayMemory(CUR_STATE_SIZE)
for _ in range(CUR_STATE_SIZE):
    cur_state.push(np.zeros((42, 42)))


def preprocess_image(image):
    image = image[29:197, :, :]
    bw = np.pad(np.dot(image, [0.2989, 0.5870, 0.1140]), [(0, 0), (4, 4)], mode='minimum')
    res = block_reduce(bw, (4, 4), np.max)
    del image
    del bw
    return res


def get_state(observation):
    global cur_state
    cur_state.append(observation)
    return np.array(cur_state)


SUM_REWARD = []
SUM_Q = []
UPDATES = []

update_num = 0

times_total = np.zeros(4)
time_start_total = 0

env = gym.make('Pong-v0')

for i_episode in range(EP_LENGTH):
    if i_episode % 3 == 0:
        print(i_episode)

    Q_EP = []
    REWARD_EP = []

    observation = env.reset()
    observation_processed = preprocess_image(observation)
    state = get_state(observation_processed)

    EPS = max(1 - (i_episode / EXP_EPISODES), 0.05)
    for t in range(EP_LENGTH):
        time_start = time.time()

        DIAGNOSTIC = i_episode % 30 == 0 and t % 50 == 0
        if DIAGNOSTIC:
            print('=' * 10)
            print('DIAGNOSTIC')
            print(i_episode, t)

        if i_episode % 10 == 0:
            env.render()

        if np.random.random() > EPS:
            if DIAGNOSTIC:
                print('EPS:', EPS)
                print('GREEDY')
            values = behavior_model.predict(np.stack([state]))
            Q_EP.append(values.max())
            action = values.argmax()
        else:
            if DIAGNOSTIC:
                print('EPS:', EPS)
                print('EXPLORING')
            action = env.action_space.sample()
        # action = env.action_space.sample()

        observation_new, reward, done, info = env.step(action)
        observation_processed = preprocess_image(observation_new)
        state_new = get_state(observation_processed)

        ER.push([state, action, reward, state_new])
        REWARD_EP.append(reward)

        state = state_new

        # update weights

        if len(ER) > MINIBATCH_SIZE:
            minibatch = np.array(random.sample(ER, MINIBATCH_SIZE))

            if DIAGNOSTIC:
                target_weights_before = copy.deepcopy(target_model.get_weights()[2])
                behavior_weights_before = copy.deepcopy(behavior_model.get_weights()[2])
            target_model.train_on_batch(minibatch, behavior_model)
            if DIAGNOSTIC:
                target_weights_after = target_model.get_weights()[2]
                behavior_weights_after = behavior_model.get_weights()[2]
                print('Updated target model:', [
                    1 - (before[0] == after[0]).mean()
                    for before, after
                    in zip(target_weights_before, target_weights_after)
                ])
                print('Updated behavior model:', [
                    1 - (before[0] == after[0]).mean()
                    for before, after
                    in zip(behavior_weights_before, behavior_weights_after)
                ])

            if DIAGNOSTIC:
                print('WEIGHTS COMPARE')
                b_body, b_heads, b_models = behavior_model.get_weights()
                t_body, t_heads, t_models = target_model.get_weights()

                print('head_biases')
                print('behaviour', [(model[6].mean()) for model in b_models])
                print('target', [(model[6].mean()) for model in t_models])


                def compare_heads(head1, head2):
                    return (head1[0] == head2[0]).mean()


                print('body:', [(b_body_lay == t_body_lay).mean() for b_body_lay, t_body_lay in zip(b_body, t_body)])
                print('heads: ', [compare_heads(b_head, t_head) for b_head, t_head in zip(b_heads, t_heads)])
                for i in range(len(b_models)):
                    print(f'model {i} layers: ', list((b_model_lay == t_model_lay).mean()
                                                      for b_model_lay, t_model_lay
                                                      in zip(b_models[i], t_models[i])))
                # print('models: ', [(b_model_lay == t_model_lay).mean() for b_model_lay, t_model_lay in zip(
                # b_models, t_models)])

            if target_model.num_updates() % UPDATES_PER_EPOCH == 0:
                behavior_model.set_weights_from(target_model)
                # behavior_model = target_model

        if t == EP_LENGTH - 1 or done:  # done
            SUM_Q.append(np.sum(Q_EP))
            SUM_REWARD.append(np.sum(REWARD_EP))
            UPDATES.append(target_model.num_updates())
            break
    if i_episode % 5 == 0:
        df = pd.DataFrame(data={'AVG_Q': SUM_Q, 'AVG_REWARD': SUM_REWARD, 'N_UPDATED': UPDATES})
        df.to_csv('data/pong/stats_dense.csv', index=False)

        target_model.save_weights('models/pong_dense')
        plt.plot(range(5, len(SUM_Q)), np.clip(SUM_Q[5:], a_min=-50, a_max=50), label='Average Q')
        plt.plot(range(5, len(SUM_Q)), SUM_REWARD[5:], label='Average Reward')
        plt.legend()
        plt.savefig('data/pong/graph.jpg')
        plt.clf()
    # if i_episode % 10 == 0:

    #     #np.save(f'data/pong/ER_{i_episode}', np.array(ER))
