#! /usr/bin/env python
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.utils import np_utils
import tensorflow as tf


# Using TensorFlow 1.0.0; use tf.python_io in later versions
#tf.python.control_flow_ops = tf
tf.python_io = tf

# Set random seed
np.random.seed(42)

# Our data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# Building the model
xor = Sequential()

# Add required layers
xor.add(Dense(8, activation='tanh',input_dim=2))
xor.add(Dense(1, activation='sigmoid'))

# Specify loss as "binary_crossentropy", optimizer as "adam",
# and add the accuracy metric
# xor.compile()
xor.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

xor.summary()

# Fitting the model
history = xor.fit(X, y, epochs=500, verbose=0)

# Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# Checking the predictions
print("\nPredictions:")
print(xor.predict_proba(X))
