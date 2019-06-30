from .AbstractAgent import AbstractAgent
from utils.RoundBuffer import RoundBuffer

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from networks.DQN_TensorFlow import DQN_TensorFlow

import time

import tensorflow as tf

import os


class DQN_Agent(AbstractAgent):
    def __init__(self, env):
        super().__init__(env)

        # RL training parameters
        self.MINIBATCH_SIZE = 32

        self.FINAL_EPS = 0.05
        self.START_EPS = 1
        self.EXP_EPISODES = 600
        self.DISCOUNT = 0.99
        ER_SIZE = 1000000

        self.UPDATES_PER_EPOCH = 10000

        self.no_update_moves = 50000

        # network parameters
        self.learning_rate = 0.00025
        #self.learning_rate = 10

        # initialization
        self.ER = RoundBuffer(ER_SIZE)

        self.num_updates = 0
        self.cur_episode_num = 0
        self.cur_eps = 1

        # networks
        self.target_network = DQN_TensorFlow(self.learning_rate, self.DISCOUNT, 'target')
        self.behavior_network = DQN_TensorFlow(self.learning_rate, self.DISCOUNT, 'behavior')

        # monitoring
        self.time_start = time.time()
        self.SUM_Q = []
        self.SUM_REWARD = []
        self.UPDATES = []
        self.Q_EP = 0
        self.Q_arr = []
        self.episode_greedy_actions = 0
        self.REWARD_EP = 0
        self.loss_list = []
        with tf.name_scope('Performance'):
            self.LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
            self.LOSS_SUMMARY = tf.summary.scalar('loss', self.LOSS_PH)
            self.REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
            self.REWARD_SUMMARY = tf.summary.scalar('reward', self.REWARD_PH)
            self.Q_PH = tf.placeholder(tf.float32, shape=None, name='q_summary')
            self.Q_SUMMARY = tf.summary.scalar('Q', self.Q_PH)

        self.PERFORMANCE_SUMMARIES = tf.summary.merge([self.LOSS_SUMMARY, self.REWARD_SUMMARY, self.Q_SUMMARY])

        SUMMARIES = "summaries"
        RUNID = 'run_84x84_2_fixed_preprocessing_fixedwriting'
        self.SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))

    def set_session(self, sess):
        self.target_network.set_session(sess)
        self.behavior_network.set_session(sess)

    def take_action(self, state):
        if np.random.random() < self.cur_eps:
            # exploration
            return self.env.action_space.sample()
        else:
            # exploitation
            self.episode_greedy_actions += 1

            action_values = self.behavior_network.predict([state])[0]
            q = action_values.max()
            self.Q_EP += q
            self.Q_arr.append(q)

            return action_values.argmax()

    def observe(self, state, action, reward, next_state, done):
        self.REWARD_EP += reward
        self.ER.push([state, action, reward, next_state, done])

    def reset_ep_stats(self):
        self.Q_EP = 0
        self.REWARD_EP = 0
        self.episode_greedy_actions = 0

    def next_episode(self):
        if len(self.ER) > self.no_update_moves:
            self.cur_eps = max(
                self.START_EPS + (self.FINAL_EPS - self.START_EPS) / self.EXP_EPISODES * self.cur_episode_num,
                self.FINAL_EPS
            )

        self.cur_episode_num += 1



        # update_stats
        AVG_Q = self.Q_EP / self.episode_greedy_actions if self.episode_greedy_actions > 0 else 0
        self.SUM_Q.append(AVG_Q)
        self.SUM_REWARD.append(self.REWARD_EP)
        self.UPDATES.append(self.num_updates)

        self.reset_ep_stats()

        if self.cur_episode_num % 5 == 0:
            #df = pd.DataFrame(data={'AVG_Q': self.SUM_Q, 'AVG_REWARD': self.SUM_REWARD, 'N_UPDATED': self.UPDATES})
            #df.to_csv('data/pong/stats_dense.csv', index=False)

            # target_model.save_weights('models/pong_dense')
            #plt.plot(range(5, len(self.SUM_Q)), np.clip(self.SUM_Q[5:], a_min=-50, a_max=50), label='SUM Q')
            #plt.plot(range(5, len(self.SUM_Q)), self.SUM_REWARD[5:], label='SUM Reward')
            #plt.legend()
            #plt.savefig('data/pong/graph.jpg')
            #plt.clf()

            if len(self.ER) > self.no_update_moves:
                with tf.Session() as sess:
                    loss_stat = np.mean(self.loss_list)
                    loss_stat = 0 if np.isnan(loss_stat) else loss_stat
                    summ = sess.run(self.PERFORMANCE_SUMMARIES,
                                    feed_dict={
                                        self.LOSS_PH: loss_stat,
                                        self.Q_PH: np.mean(self.SUM_Q[-10:]),
                                        self.REWARD_PH: np.mean(self.SUM_REWARD[-10:])
                                    }
                                    )
                    self.SUMM_WRITER.add_summary(summ, self.num_updates)
                    self.loss_list = []

    @property
    def is_diagnostic(self):
        return self.num_updates % 1000 == 0

    def step_train(self):
        if len(self.ER) > self.no_update_moves:
            minibatch = self.ER.sample(self.MINIBATCH_SIZE)
            loss = self.behavior_network.train_on_batch(minibatch, self.target_network)
            self.loss_list.append(loss)
            self.num_updates += 1

            if self.is_diagnostic:
                print('-' * 10)
                print('updates: ', self.num_updates, ' time: ', time.time() - self.time_start)
                print('eps: ', self.cur_eps)
                print('ep: ', self.cur_episode_num)

            if self.num_updates % self.UPDATES_PER_EPOCH == 0:
                self.target_network.set_params(self.behavior_network)
