# ==============================================================================
#
# TIME SERIES QUESTION
#
# Build and train a neural network to predict time indexed variables of
# the multivariate house hold electric power consumption time series dataset.
# Using a window of past 24 observations of the 7 variables, the model
# should be trained to predict the next 24 observations of the 7 variables.
#
# ==============================================================================
#
# ABOUT THE DATASET
#
# Original Source:
# https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
#
# The original Individual House Hold Electric Power Consumption Dataset
# has Measurements of electric power consumption in one household with
# a one-minute sampling rate over a period of almost 4 years.
#
# Different electrical quantities and some sub-metering values are available.
#
# For the purpose of the examination we have provided a subset containing
# the data for the first 60 days in the dataset. We have also cleaned the
# dataset beforehand to remove missing values. The dataset is provided as a
# CSV file in the project.
#
# The dataset has a total of 7 features ordered by time.
# ==============================================================================
#
# INSTRUCTIONS
#
# Complete the code in following functions:
# 1. solution_model()
#
# Your code will fail to be graded if the following criteria are not met:
#
# 1. Model input shape must be (BATCH_SIZE, N_PAST = 24, N_FEATURES = 7),
#    since the testing infrastructure expects a window of past N_PAST = 24
#    observations of the 7 features to predict the next N_FUTURE = 24
#    observations of the same features.
#
# 2. Model output shape must be (BATCH_SIZE, N_FUTURE = 24, N_FEATURES = 7)
#
# 3. The last layer of your model must be a Dense layer with 7 neurons since
#    the model is expected to predict observations of 7 features.
#
# 4. Don't change the values of the following constants:
#    SPLIT_TIME, N_FEATURES, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT, in
#    solution_model() (See code for additional note on BATCH_SIZE).
#
# 5. Code for normalizing the data is provided - don't change it.
#    Changing the normalizing code will affect your score.
#
# 6. Code for converting the dataset into windows is provided - don't change it.
#    Changing the windowing code will affect your score.
#
# 7. Code for setting the seed is provided - don't change it.
#
# HINT: If you follow all the rules mentioned above and throughout this
# question while training your neural network, there is a possibility that a
# validation MAE of approximately 0.055 or less on the normalized validation
# dataset may fetch you top marks.


import urllib
import zipfile

import pandas as pd
import tensorflow as tf
import numpy as np


# This function downloads and extracts the dataset to the directory that
# contains this file.
# DO NOT CHANGE THIS CODE
# (unless you need to change https to http)
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()


# This function normalizes the dataset using min max scaling.
# DO NOT CHANGE THIS CODE
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data


# This function is used to map the time series dataset into windows of
# features and respective targets, to prepare it for training and
# validation. First element of the first window will be the first element of
# the dataset. Consecutive windows are constructed by shifting
# the starting position of the first window forward, one at a time (indicated
# by shift=1). For a window of n_past number of observations of all the time
# indexed variables in the dataset, the target for the window
# is the next n_future number of observations of these variables, after the
# end of the window.

# DO NOT CHANGE THIS CODE
def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)



# This function loads the data from CSV file, normalizes the data and
# splits the dataset into train and validation data. It also uses
# windowed_dataset() to split the data into windows of observations and
# targets. Finally it defines, compiles and trains a neural network. This
# function returns the final trained model.

# COMPLETE THE CODE IN THIS FUNCTION
def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.
    download_and_extract_data()
    # Reads the dataset from the CSV.
    df = pd.read_csv('household_power_consumption.csv', sep=',',
                     infer_datetime_format=True, index_col='datetime', header=0)
    print(df.head(3))

    # Number of features in the dataset. We use all features as predictors to
    # predict all features at future time steps.
    N_FEATURES = len(df.columns) # DO NOT CHANGE THIS

    # Normalizes the data
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splits the data into training and validation sets.
    SPLIT_TIME = int(len(data) * 0.5) # DO NOT CHANGE THIS
    x_train = np.array(data[:SPLIT_TIME])
    x_valid = np.array(data[SPLIT_TIME:])
    print(x_train.shape, x_valid.shape)

    # DO NOT CHANGE THIS CODE
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    # DO NOT CHANGE BATCH_SIZE IF YOU ARE USING STATEFUL LSTM/RNN/GRU.
    # THE TEST WILL FAIL TO GRADE YOUR SCORE IN SUCH CASES.
    # In other cases, it is advised not to change the batch size since it
    # might affect your final scores. While setting it to a lower size
    # might not do any harm, higher sizes might affect your scores.
    BATCH_SIZE = 32  # ADVISED NOT TO CHANGE THIS

    # DO NOT CHANGE N_PAST, N_FUTURE, SHIFT. The tests will fail to run
    # on the server.
    # Number of past time steps based on which future observations should be
    # predicted
    N_PAST = 24  # DO NOT CHANGE THIS

    # Number of future time steps which are to be predicted.
    N_FUTURE = 24  # DO NOT CHANGE THIS

    # By how many positions the window slides to create a new window
    # of observations.
    SHIFT = 1  # DO NOT CHANGE THIS

    # Code to create windowed train and validation datasets.
    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    for item in train_set.take(1):
        # train shape, label shape
        print(item[0].shape, item[1].shape)
    # Code to define your model.

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=[N_PAST, N_FEATURES]),
        tf.keras.layers.LSTM(32),
        # tf.keras.layers.LSTM(128, input_shape=[N_PAST, N_FEATURES]),
        # tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(N_FUTURE*N_FEATURES),
        tf.keras.layers.Reshape(target_shape=(N_FUTURE, N_FEATURES)),
    ])

    print(model.summary())
    # Code to train and compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='huber',
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(train_set,
                        epochs=12 ,
                        validation_data=valid_set)

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.

if __name__ == '__main__':
    solution_model()
    #model = solution_model()
    #model.save("mymodel.h5")
