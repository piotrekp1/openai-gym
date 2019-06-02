import gym
import matplotlib.pyplot as plt
import time
import numpy as np
from skimage.measure import block_reduce
from collections import deque

import pandas as pd

import random

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

from keras.optimizers import Adam
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import timeit

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)

# create model
model = Sequential()
# add model layers
model.add(
    Conv2D(16, kernel_size=8, strides=4, activation='relu', data_format='channels_first', input_shape=(4, 42, 42)))
model.add(Conv2D(32, kernel_size=4, strides=2, activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(6))

opt = Adam(lr=0.1,
           # beta_1=0.9, beta_2=0.999, epsilon=None,
           decay=0.01,
           amsgrad=False)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

MINIBATCH_SIZE = 250

EPS = 0.05
DISCOUNT = 0.9
ER_SIZE = 5000
CUR_STATE_SIZE = 4

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

env = gym.make('Pong-v0')
for i_episode in range(EP_LENGTH):
    if i_episode % 3 == 0:
        print(i_episode)

    Q_EP = []
    REWARD_EP = []

    observation = env.reset()
    observation_processed = preprocess_image(observation)
    state = get_state(observation_processed)

    EPS = 0.8 if i_episode < 20 else 0.05
    for t in range(EP_LENGTH):
        if i_episode % 10 == 0:
            env.render()
        # if t > 60 and t % 10 == 0:
        #    plt.figure()
        #    plt.imshow(observation_new)
        #    plt.show()
        #    plt.imshow(observation_processed, cmap=plt.get_cmap('gray'))
        #    plt.show()
        # time.sleep(0.01)
        if t % 4 == 0:
            if np.random.random() > EPS:
                values = model.predict(np.stack([state]))
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
        if len(ER) > MINIBATCH_SIZE and t % 4 == 0:
            times = []
            times.append(time.time())

            minibatch = np.array(random.sample(ER, MINIBATCH_SIZE))
            mb_states, mb_actions, mb_rewards, mb_states_new = [minibatch[:, i] for i in range(4)]
            mb_states, mb_states_new = np.stack(mb_states), np.stack(mb_states_new)
            # minibatch = np.array(random.sample(ER, MINIBATCH_SIZE))

            times.append(time.time())

            # preds_total = model.predict(np.concatenate([mb_states, mb_states_new]))

            state_preds = model.predict(mb_states)
            state_new_preds = model.predict(mb_states_new)
            # state_preds = preds_total[:MINIBATCH_SIZE]
            # state_new_preds = preds_total[MINIBATCH_SIZE:]

            times.append(time.time())
            for i in range(MINIBATCH_SIZE):
                state_preds[i, int(mb_actions[i])] = mb_rewards[i] + DISCOUNT * state_new_preds[i, :].max()  # TD update

            times.append(time.time())
            model.train_on_batch(mb_states, state_preds)

            times.append(time.time())

            for i in range(4):
                times_total[i] += times[i + 1] - times[i]

            update_num += 1

        if t == EP_LENGTH - 1 or done:  # done
            SUM_Q.append(np.sum(Q_EP))
            SUM_REWARD.append(np.sum(REWARD_EP))
            UPDATES.append(update_num)
            break
    if i_episode % 5 == 0:
        df = pd.DataFrame(data={'AVG_Q': SUM_Q, 'AVG_REWARD': SUM_REWARD, 'N_UPDATED': UPDATES})
        df.to_csv('data/pong/stats.csv', index=False)
        model.save_weights('models/pong.h5')
        plt.plot(range(5, len(SUM_Q)), np.clip(SUM_Q[5:], a_min=-50, a_max=50), label='Average Q')
        plt.plot(range(5, len(SUM_Q)), SUM_REWARD[5:], label='Average Reward')
        plt.legend()
        plt.savefig('data/pong/graph.jpg')
        plt.clf()
