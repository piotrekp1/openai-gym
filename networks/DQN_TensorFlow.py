import tensorflow as tf
import numpy as np


class RawDQN:
    def __init__(self, learning_rate):
        h, w = 84, 84
        num_moves = 4

        self.input = tf.placeholder(shape=[None, 4, h, w], dtype=tf.float32)
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

        self.input_scaled = self.input / 255

        self.conv1 = tf.layers.conv2d(
            inputs=self.input_scaled, filters=32,
            kernel_size=[8, 8], padding="valid",
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            use_bias=False,
            data_format='channels_first', strides=4
        )
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1, filters=64,
            kernel_size=[4, 4], padding="valid",
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            use_bias=False,
            data_format='channels_first', strides=2
        )
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2, filters=64,
            kernel_size=[3, 3], padding="valid",
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            use_bias=False,
            data_format='channels_first', strides=1
        )
        self.conv4 = tf.layers.conv2d(
            inputs=self.conv3, filters=1024,
            kernel_size=[7, 7], padding="valid",
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            use_bias=False,
            data_format='channels_first', strides=1
        )

        self.conv4_flat = tf.layers.flatten(self.conv4)
        self.outputs = tf.layers.dense(
            inputs=self.conv4_flat, units=num_moves
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
    def __init__(self, learninig_rate=0.00025, DISCOUNT=0.99, scope_name='global', DoubleDQN=True):
        self.scope_name = scope_name
        with tf.variable_scope(scope_name):
            self.dqn = RawDQN(learninig_rate)
        self.trainable_variables = tf.trainable_variables(scope=scope_name)
        self.DISCOUNT = DISCOUNT
        self.sess = None
        self.DoubleDQN = DoubleDQN

    def predict(self, x):
        return self.sess.run(self.dqn.outputs,
                             feed_dict={
                                 self.dqn.input: x
                             })

    def train_on_batch(self, minibatch, target_network):
        states, actions, rewards, next_states, dones = zip(*minibatch)
        not_dones = np.array(dones) * -1 + 1

        next_state_target_preds = target_network.predict(next_states)

        # IF DOUBLE DQN USE BEHAVIOUR NETWORK, OTHERWISE TARGET NETWORK FOR ARGMAX
        network_to_argmax_from = self if self.DoubleDQN else target_network
        next_state_inds = np.argmax(network_to_argmax_from.predict(next_states), axis=1)

        next_state_preds = next_state_target_preds[range(len(minibatch)), next_state_inds]

        expected_state_action_values = (next_state_preds * self.DISCOUNT * not_dones) + np.array(rewards)

        _, loss = self.sess.run([self.dqn.update, self.dqn.loss], feed_dict={
            self.dqn.input: states,
            self.dqn.actions: actions,
            self.dqn.target_q: expected_state_action_values
        })
        return loss

    def set_session(self, sess):
        self.sess = sess

    def tf_copy_variables(self, from_network):
        update_ops = []
        for var_to, var_from in zip(self.trainable_variables, from_network.trainable_variables):
            copy_op = var_to.assign(var_from.value())
            update_ops.append(copy_op)
        return update_ops

    def set_params(self, from_network):
        update_ops = self.tf_copy_variables(from_network)
        for copy_op in update_ops:
            self.sess.run(copy_op)
