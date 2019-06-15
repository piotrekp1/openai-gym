from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam, SGD

import copy
import numpy as np

from keras.layers import LeakyReLU

import time

import os


class DQN2:
    def __init__(self, learning_rate=0.1, DISCOUNT=0.95, decay=0):
        self.DISCOUNT = DISCOUNT
        body_layers = [
            Conv2D(16, kernel_size=8, strides=4, data_format='channels_first',
                   input_shape=(4, 42, 42)),# activation='tanh'),
            Conv2D(32, kernel_size=4, strides=2), #, activation='tanh'),
            Flatten(),
            Dense(256)#, activation='tanh')
        ]
        head_layers = [Dense(1, use_bias=False) for _ in range(6)]

        self.body = Sequential(body_layers)
        self.models = [Sequential(body_layers + [head_layer]) for head_layer in head_layers]
        self.heads = [Sequential([head]) for head in head_layers]

        for model in self.models + self.heads + [self.body]:
            #opt = Adam(lr=learning_rate,
            #           # decay=decay,
            #          amsgrad=False)
            opt = SGD(lr=learning_rate)
            model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

        self.update_num = 0
        self.time_total = 0

    def set_weights_from(self, dqn_to_set_weights):
        body_weights = copy.deepcopy(dqn_to_set_weights.body.get_weights())
        self.body.set_weights(body_weights)
        for i in range(len(self.models)):
            head_weights = copy.deepcopy(dqn_to_set_weights.heads[i].get_weights())
            self.heads[i].set_weights(head_weights)
            self.models[i].set_weights(body_weights + head_weights)

    def predict(self, x):
        body_pred = self.body.predict(x)
        return np.stack([head.predict(body_pred)[:, 0] for head in self.heads], axis=1)

    def train_on_batch_i(self, x, y, i):
        self.models[i].train_on_batch(x, y)

    def train_on_batch(self, batch, behaviour_network):
        time_start = time.time()
        preds_new = behaviour_network.predict(np.stack(batch[:, 3])).max(axis=1)
        preds_new = np.clip(preds_new, -1, 1)

        for i in range(6):
            cond = batch[:, 1] == i
            if cond.sum() == 0:
                continue
            new_y_cond = preds_new[cond]
            new_y_cond = new_y_cond * self.DISCOUNT + batch[cond, 2]

            new_x_cond = np.stack(batch[cond, 0])

            self.train_on_batch_i(new_x_cond, new_y_cond, i)

        time_end = time.time()
        self.time_total += time_end - time_start
        self.update_num += 1
        if self.update_num % 1000 == 0:
            print(f'update_num: {self.update_num}, total_time: {self.time_total}')

    def save_weights(self, dir_path):
        self.body.save_weights(os.path.join(dir_path, 'body.h5'))
        for i, head in enumerate(self.heads):
            head.save_weights(os.path.join(dir_path, f'head_{i}.h5'))

    def num_updates(self):
        return self.update_num

    def get_weights(self):
        return self.body.get_weights(), [head.get_weights() for head in self.heads], [model.get_weights() for model in
                                                                                      self.models]
