import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

from keras.optimizers import Adam
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)

# create model
model = Sequential()
# add model layers
model.add(Conv2D(16, kernel_size=8, strides=4, activation='relu', input_shape=(84, 84, 4)))
model.add(Conv2D(32, kernel_size=4, strides=2, activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(6))

opt = Adam(lr=0.1,
           # beta_1=0.9, beta_2=0.999, epsilon=None,
           decay=0.01,
           amsgrad=False)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])



X = np.random.random((10, 84, 84, 4))

y = np.random.random((10, 6))

model.fit(X, y)