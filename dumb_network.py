from keras.layers import Input, Conv2D, Lambda, Flatten, merge, multiply, maximum, subtract, add, Dense, dot, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.initializers import Constant
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import h5py
from training_v1 import Dataset

# Global Constants
IMG_W = 64;
IMG_H = 64;

# ====== HYPERPARAMS ======
batch_sz = 10;
num_epochs = 10;


# ====== LOAD DATA ======
data = h5py.File('data/liberty.h5', 'r')
training_dset = Dataset(data, batch_size=batch_sz);


# ====== NETWORK ARCHITECTURE ======
convnet = Sequential()
convnet.add(Flatten(input_shape=(IMG_W, IMG_H)))
convnet.add(Dense(3, activation='relu', input_shape=(2,IMG_W,IMG_H), kernel_initializer='random_uniform', bias_initializer=Constant(value=0)))


# ====== SIAMESE NETWORK ======
siamese_input = Input((2,IMG_W,IMG_H))
left_input = Lambda(lambda x: x[:, 0, :, :], output_shape=(batch_sz,IMG_W,IMG_H),)(siamese_input);
right_input = Lambda(lambda x: x[:, 1, :, :], output_shape=(batch_sz,IMG_W,IMG_H),)(siamese_input);
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
# calculate the loss
abs_diff_lambda = lambda x: K.abs(x[0]-x[1])
abs_diff = merge([encoded_l, encoded_r], mode = abs_diff_lambda, output_shape=lambda x: x[0])
dist_sq = dot([abs_diff, abs_diff], axes=1, normalize=False)


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


# ====== TRAINING ======
siamese_net = Model(output=dist_sq, inputs=siamese_input)
optimizer = Adam(0.01) # default was 0.00006
siamese_net.compile(loss=loss_func,optimizer=optimizer)
epoch = 1;
for X_batch, y_batch in training_dset:
  # Run the graph on a batch of training data; recall that asking
  # TensorFlow to evaluate loss will cause an SGD step to happen.
  loss = siamese_net.train_on_batch(x=X_batch, y=y_batch)
  print 'Epoch #', epoch, ', loss=', loss
  epoch += 1
  if epoch > num_epochs:
    break;


