import tensorflow as tf
import numpy as np


class RawDQN:
    def __init__(self, learning_rate):
        h, w = 42, 42
        num_moves = 6

        self.input = tf.placeholder(shape=[None, 4, h, w], dtype=tf.float32)
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

        self.conv1 = tf.layers.conv2d(
            inputs=self.input, filters=16,
            kernel_size=[5, 5], padding="same",
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            use_bias=False,
            data_format='channels_first', strides=2
        )
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1, filters=32,
            kernel_size=[5, 5], padding="same",
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            use_bias=False,
            data_format='channels_first', strides=2
        )
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2, filters=64,
            kernel_size=[5, 5], padding="same",
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            use_bias=False,
            data_format='channels_first', strides=2
        )
        self.conv4 = tf.layers.conv2d(
            inputs=self.conv3, filters=64,
            kernel_size=[5, 5], padding="same",
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            use_bias=False,
            data_format='channels_first', strides=2
        )

        self.conv4_flat = tf.layers.flatten(self.conv4)
        self.outputs = tf.layers.dense(
            inputs=self.conv3_flat, units=num_moves
        )

        self.Q = tf.reduce_sum(tf.multiply(
            self.outputs,
            tf.one_hot(self.actions, num_moves, dtype=tf.float32)
        ), axis=1)

        print(self.target_q.shape)
        print(self.Q.shape)

        self.loss = tf.reduce_mean(tf.losses.huber_loss(
            labels=self.target_q,
            predictions=self.Q
        ))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update = self.optimizer.minimize(self.loss)


class DQN_TensorFlow:
    def __init__(self, learninig_rate=0.00025, DISCOUNT=0.99, scope_name='global'):
        self.scope_name = scope_name
        with tf.variable_scope(scope_name):
            self.dqn = RawDQN(learninig_rate)
        self.trainable_variables = tf.trainable_variables(scope=scope_name)
        self.DISCOUNT = DISCOUNT
        self.sess = None

    def predict(self, x):
        return self.sess.run(self.dqn.outputs,
                             feed_dict={
                                 self.dqn.input: x
                             })

    def train_on_batch(self, minibatch, target_network):
        states, actions, rewards, next_states, dones = zip(*minibatch)
        not_dones = np.array(dones) * -1 + 1

        next_state_preds = target_network.predict(next_states).max(axis=1)
        expected_state_action_values = (next_state_preds * self.DISCOUNT * not_dones) + np.array(rewards)

        self.sess.run(self.dqn.update, feed_dict={
            self.dqn.input: states,
            self.dqn.actions: actions,
            self.dqn.target_q: expected_state_action_values
        })

    def set_session(self, sess):
        self.sess = sess

    def tf_copy_variables(self, from_network):
        update_ops = []
        for i, var in enumerate(self.trainable_variables):
            copy_op = from_network.trainable_variables[i].assign(var.value())
            update_ops.append(copy_op)
        return update_ops

    def set_params(self, from_network):
        update_ops = self.tf_copy_variables(from_network)
        for copy_op in update_ops:
            self.sess.run(copy_op)
