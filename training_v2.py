import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, Lambda, Flatten, multiply, maximum, add, Dense, dot, Flatten,MaxPooling2D
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import SGD,Adam
from tensorflow.python.keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import h5py
from dataset import Dataset

import keras

# Global Constants
IMG_W = 64;
IMG_H = 64;

# ====== HYPERPARAMS ======
batch_sz = 10;
num_steps = 10;


# ====== LOAD DATA ======
data = h5py.File('data/liberty.h5', 'r')
training_dset = Dataset(data, batch_size=batch_sz);


# ====== NETWORK ARCHITECTURE ======
convnet = Sequential()
convnet.add(Flatten(input_shape=(IMG_W, IMG_H)))
# convnet.add(keras.layers.BatchNormalization(axis=1, batch_input_shape=(IMG_W*IMG_H,)))
convnet.add(Dense(2, activation='relu', input_shape=(IMG_W*IMG_H,),      kernel_initializer='random_uniform', bias_initializer=Constant(value=0),
kernel_regularizer=tf.keras.regularizers.l2(0.1), bias_regularizer=tf.keras.regularizers.l2(0.1)))


# ====== SIAMESE NETWORK ======
siamese_input = Input((2,IMG_W,IMG_H))
norm_input = siamese_input
# norm_input = keras.layers.BatchNormalization()(siamese_input)
left_input = Lambda(lambda x: x[:, 0, :, :], output_shape=(batch_sz,IMG_W,IMG_H),)(norm_input);
right_input = Lambda(lambda x: x[:, 1, :, :], output_shape=(batch_sz,IMG_W,IMG_H),)(norm_input);
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
# calculate the loss
encoded_diff = tf.subtract(encoded_l, encoded_r)
dist_sq = tf.keras.layers.dot([encoded_diff, encoded_diff], axes=1)


# ====== LOSS FUNCTION ======
def loss_func(y_true, y_pred):
  # y_true - should be boolean (int)... 0 if non-corresponding, 1 if matched.
  # y_pred - is distance squared from siamese network
  # Here, we calculate:
  # (y_true * dist) + (1 - ytrue) * max(0, C - dist)

  # y_true == 1:
  dist = K.sqrt(y_pred);
  match_term = multiply([dist, y_true]); # y_true * dist

  # y_true == 0:
  C = K.constant([10.0]); # C = 10
  zero_const = K.constant([0.0]); # zero = 0
  one_const = K.constant([1.0]); # one = 1
  c_minus_dist = subtract([C, dist]); # C - dist
  temp1 = maximum([zero_const, c_minus_dist]); # max(0, C - dist)
  nonmatch_term_coeff = subtract([one_const, y_true]); # (1 - ytrue)
  nonmatch_term = multiply([nonmatch_term_coeff, temp1]); # (1 - ytrue) * max(0, C - dist)

  return add([match_term, nonmatch_term]);

class SiameseNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.splitLeft = Lambda(lambda x: x[:, 0, :, :], output_shape=(batch_sz,IMG_W,IMG_H),)
        self.splitRight = Lambda(lambda x: x[:, 1, :, :], output_shape=(batch_sz,IMG_W,IMG_H),)
    def call(self, x, training=None):
        xL = self.splitLeft(x)
        xR = self.splitRight(x)
        xL = convnet(xL)
        xR = convnet(xR)
        x = tf.subtract(xL, xR)
        x = tf.keras.layers.dot([x, x], axes=1)
        return x
        

# ====== TRAINING ======
siamese_net = Model(outputs=dist_sq, inputs=siamese_input)
optimizer = SGD(0.000001) # Adam(0.01) # default was 0.00006
siamese_net.compile(loss=loss_func,optimizer=optimizer)
weights = siamese_net.get_weights()
print "num weights: ", len(weights)
# print weights[0].shape
# print weights[4].shape, weights[5].shape
print "init weights: ", weights
step = 1;
for X_batch, y_batch in training_dset:

  # Run the graph on a batch of training data; recall that asking
  # TensorFlow to evaluate loss will cause an SGD step to happen.
  loss = siamese_net.train_on_batch(x=X_batch, y=y_batch)
  weights = siamese_net.get_weights()
  print "weights at step ", step, ": ", weights
  print 'Step #', step, ', loss=', loss
  step += 1
  if step > 10: # num_steps:
    break;
