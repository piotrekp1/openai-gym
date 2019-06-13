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


from DQN.DQN1 import DQN1
from DQN.DQN2 import DQN2

import timeit

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)

# create model
behavior_model = DQN2(learning_rate=0.4, DISCOUNT=0.95)
target_model = DQN2(learning_rate=0.4, DISCOUNT=0.95)

# su_model = SingleUpdate(model)


MINIBATCH_SIZE = 32

EPS = 0.05
EXP_EPISODES = 1000
DISCOUNT = 0.9
ER_SIZE = 225000
CUR_STATE_SIZE = 4

UPDATES_PER_EPOCH = 20000

EP_LENGTH = 100000

ER = deque(maxlen=ER_SIZE)

cur_state = deque(maxlen=CUR_STATE_SIZE)
for _ in range(CUR_STATE_SIZE):
    cur_state.append(np.zeros((42, 42)))


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
        if i_episode % 5 == 0:
            env.render()

        if np.random.random() > EPS:
            values = behavior_model.predict(np.stack([state]))
            Q_EP.append(values.max())
            action = values.argmax()
        else:
            action = env.action_space.sample()
        # action = env.action_space.sample()

        observation_new, reward, done, info = env.step(action)
        observation_processed = preprocess_image(observation_new)
        state_new = get_state(observation_processed)

        ER.append([state, action, reward, state_new])
        REWARD_EP.append(reward)

        state = state_new

        # update weights

        if len(ER) > MINIBATCH_SIZE:
            minibatch = np.array(random.sample(ER, MINIBATCH_SIZE))
            target_model.train_on_batch(minibatch, behavior_model)

            if target_model.num_updates() % UPDATES_PER_EPOCH == 0:
                behavior_model.set_weights_from(target_model)
                #behavior_model = target_model

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
