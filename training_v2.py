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
batch_sz = 100;
num_steps = 100;


# ====== LOAD DATA ======
data = h5py.File('data/liberty.h5', 'r')
training_dset = Dataset(data, batch_size=batch_sz);

# ====== LOSS FUNCTION ======
def loss_func(y_true, y_pred):
    # y_true - should be boolean (int)... 0 if non-corresponding, 1 if matched.
    # y_pred - is distance squared from siamese network
    # Here, we calculate:
    # (y_true * dist) + (1 - ytrue) * max(0, C - dist)

    # y_true == 1:
    dist = tf.sqrt(y_pred);
    match_term = multiply([dist, y_true]); # y_true * dist
    match_term = tf.reshape(match_term, (batch_sz,));
    
    # y_true == 0:
    one_const = tf.constant(1.0, shape=(batch_sz,)); # one = 1
    nonmatch_term_coeff = tf.subtract(one_const, y_true); # (1 - ytrue)
    C = tf.constant(10.0); # C = 10
    zero_const = tf.constant(0.0, shape=(batch_sz,)); # zero = 0
    c_minus_dist = tf.subtract(C, dist); # C - dist
    c_minus_dist = tf.reshape(c_minus_dist, (batch_sz,));
    temp1 = tf.maximum(zero_const, c_minus_dist); # max(0, C - dist)
    nonmatch_term = tf.multiply(nonmatch_term_coeff, temp1); # (1 - ytrue) * max(0, C - dist)
    
    # add the two losses
    loss_val = tf.add(match_term, nonmatch_term);

    return loss_val

class SiameseNet(tf.keras.Model):

    def __init__(self):
        super(SiameseNet, self).__init__()
        initializer = tf.variance_scaling_initializer(scale=2.0)
        self.conv1 = tf.layers.Conv2D(filters = 1, kernel_size = (3, 3), strides = (2,2), padding = "SAME", activation = tf.nn.relu, use_bias = True, kernel_initializer = initializer)
    
    # apply convnet; operates only on one of the left/right channels at a time.
    def apply_convnet(self, x):

        # the actual network architecture:
        x_out = self.conv1(x)
        
        # flatten because at the end we want a single descriptor per input
        x_out = tf.layers.flatten(x_out);
        return x_out;
    
    # execute the siamese net
    def call(self, x, training=None):
        xL = tf.slice(x, [0, 0, 0, 0], [batch_sz, 1, IMG_W, IMG_H]);
        xR = tf.slice(x, [0, 1, 0, 0], [batch_sz, 1, IMG_W, IMG_H]);
        
        # for conv2d, we need the dims to be ordered (batch_sz, img_w, img_h, channel)
        xL = tf.transpose(xL, [0, 2, 3, 1])
        xR = tf.transpose(xR, [0, 2, 3, 1])

        # Convnet
        xL = self.apply_convnet(xL)
        xR = self.apply_convnet(xR)

        # compute distance squared; we will pass this to the loss function
        dist_sq = tf.subtract(xL, xR);
        dist_sq = tf.multiply(dist_sq, dist_sq);
        dist_sq = tf.reduce_sum(dist_sq, axis=1, keepdims=True);
        return dist_sq

# ====== TRAINING ======
# Construct computational graph
tf.reset_default_graph()
with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float32, [None, 2, IMG_W, IMG_H])
    y = tf.placeholder(tf.float32, [None])
    scores = SiameseNet()(x);
    loss_calc = loss_func(y, scores);
    optimizer = tf.train.GradientDescentOptimizer(1e-3);
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_calc)

# Run computational graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1;
    for X_batch, y_batch in training_dset:

        feed_dict = {x: X_batch, y: y_batch}
        
        loss_calc = tf.Print(loss_calc,
                             [tf.gradients(loss_calc, tf.trainable_variables()[0])],
                             summarize=10
                             )
        
        loss_output, _ = sess.run([loss_calc, train_op], feed_dict=feed_dict)
        print np.mean(loss_output)
        
        



        # next step
        num_steps = 5
        if step > num_steps:
            break;
        step += 1

