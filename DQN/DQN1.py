from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

import numpy as np

import time


class DQN1:
    def __init__(self):
        self.DISCOUNT = 0.9

        self.model = Sequential()
        # add model layers
        self.model.add(
            Conv2D(16, kernel_size=8, strides=4, activation='relu', data_format='channels_first',
                   input_shape=(4, 42, 42)))
        self.model.add(Conv2D(32, kernel_size=4, strides=2, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(6))

        opt = Adam(lr=0.3,
                   # beta_1=0.9, beta_2=0.999, epsilon=None,
                   # decay=0.01,
                   amsgrad=False)

        self.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

        self.update_num = 0

        self.times_total = np.zeros(4)
        self.time_total = 0

    def predict(self, X):
        return self.model.predict(X)

    def train_on_batch(self, batch):
        self.times = []
        time_start = time.time()
        mb_states, mb_actions, mb_rewards, mb_states_new = [batch[:, i] for i in range(4)]
        mb_states, mb_states_new = np.stack(mb_states), np.stack(mb_states_new)
        # minibatch = np.array(random.sample(ER, MINIBATCH_SIZE))

        self.times.append(time.time())

        state_preds = self.model.predict(mb_states)
        state_new_preds = self.model.predict(mb_states_new)

        self.times.append(time.time())
        for i in range(batch.shape[0]):
            state_preds[i, int(mb_actions[i])] = mb_rewards[i] + self.DISCOUNT * state_new_preds[i,
                                                                                 :].max()  # TD update

        self.times.append(time.time())
        self.model.train_on_batch(mb_states, state_preds)

        self.times.append(time.time())

        for i in range(3):
            self.times_total[i] += self.times[i + 1] - self.times[i]

        time_end = time.time()
        self.time_total += time_end - time_start
        self.update_num += 1
        if self.update_num % 10 == 0:
            # print(f'update_num: {self.update_num}, total_time: {self.time_total}')
            print('-' * 20)
            for i in range(3):
                print(self.times_total[i])
                self.times_total[i] += self.times[i + 1] - self.times[i]

    def num_updates(self):
        return self.update_num

    def save_weights(self, path):
        self.model.save_weights(path)
