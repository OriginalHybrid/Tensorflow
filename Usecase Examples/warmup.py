import tensorflow as tf
import numpy as np

# 4x - 3
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-7.0, -3.0, 1.0, 5.0, 9.0, 13.0], dtype=float)
tf.keras.backend.clear_session()


def ashish():
    model0 = tf.keras.Sequential([
                tf.keras.layers.Dense(1, input_shape=[1])
                ])
    model0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss='mean_squared_error', metrics=["mse", "mae"])
    model0.fit(xs, ys, epochs=1000, validation_split=0.2)
    print(model0.predict([1.0]))
    print(model0.predict([5.0]))
    print(model0.predict([10.0]))


def himanshu():
    tf.keras.backend.clear_session()
    model0 = tf.keras.Sequential([
        tf.keras.layers.Dense(units=10, input_shape=[1], activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    model0.compile(optimizer='sgd', loss='mean_squared_error', metrics=["mae", "mse"])
    history = model0.fit(xs, ys, epochs=1000, validation_split=0.2)
    print(model0.predict([1.0]))
    print(model0.predict([5.0]))
    print(model0.predict([10.0]))


if __name__ == "__main__":
    ashish()
    #himanshu()