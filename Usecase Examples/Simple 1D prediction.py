

import tensorflow as tf
import numpy as np
from tensorflow import keras


#model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=10, input_shape=[1]))
model.add(tf.keras.layers.Dense(units=1))
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

xs = np.array([-1.0, 0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500, validation_split=0.2)
print(model.predict([10.0]))
print(model.predict([100.0]))
