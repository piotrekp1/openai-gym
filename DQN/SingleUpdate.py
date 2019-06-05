import tensorflow as tf


class SingleUpdate:
    def __init__(self, model):
        self.model = model

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

        self.global_step = tf.Variable(0)

    def loss(self, x, y, i):
        y_ = self.model(x)[:, i]
        return tf.losses.mean_squared_error(labels=y, predictions=y_)

    def grad(self, inputs, targets, i):
        with tf.GradientTape() as tape:
            loss_value = self.loss(inputs, targets, i)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def train_on_batch(self, x, y, i):
        loss_value, grads = self.grad(x, y, i)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables),
                                       self.global_step)
        return loss_value

    def predict(self, x):
        return self.model.predict(x)
