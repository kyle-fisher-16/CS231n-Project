import tensorflow as tf
import numpy.random as rng
import numpy as np
import os
import h5py
from dataset import Dataset
import matplotlib.pyplot as plt

# Global Constants
IMG_W = 64;
IMG_H = 64;
PLOT_BATCH = True; # whether or not to plot the batch distances

# ====== HYPERPARAMS ======
mining_ratio = 8;
batch_sz = 32;
num_steps = 1000;
learning_rate = 1e-3;
batches_per_epoch = 5;

# ====== LOAD DATA ======
data = h5py.File('data/liberty.h5', 'r')
training_dset = Dataset(data, batch_size=batch_sz);

# ====== SETUP ======
if PLOT_BATCH:
    plot_batch_fig, plot_batch_ax = plt.subplots(1,1)
    plt.ion()
    plt.show()

# ====== LOSS FUNCTION ======
def hinge_embed_loss_func(y_true, dist):
    # y_true - should be boolean (int)... 0 if non-corresponding, 1 if matched.
    # dists - is distance from siamese network
    # Here, we calculate:
    # (y_true * dist) + (1 - ytrue) * max(0, C - dist)
    # y_true == 1:
    
    # TODO: CLEAN VERSION
    #loss_val = tf.cond(y_true > 0.5, lambda: dist, lambda: tf.maximum(0.0, tf.subtract(10.0, dist)))
    
    match_term = tf.multiply(dist, y_true); # y_true * dist
    match_term = tf.reshape(match_term, (batch_sz,));
    
    # y_true == 0:
    one_const = tf.constant(1.0, shape=(batch_sz,)); # one = 1
    nonmatch_term_coeff = tf.subtract(one_const, y_true); # (1 - ytrue)
    C = tf.constant(1.0);
    zero_const = tf.constant(0.0, shape=(batch_sz,)); # zero = 0
    c_minus_dist = tf.subtract(C, dist); # C - dist
    c_minus_dist = tf.reshape(c_minus_dist, (batch_sz,));
    temp1 = tf.maximum(zero_const, c_minus_dist); # max(0, C - dist)
    nonmatch_term = tf.multiply(nonmatch_term_coeff, temp1); # (1 - ytrue) * max(0, C - dist)
    
    # add the two losses
    loss_val = tf.add(match_term, nonmatch_term);
#    loss_val = tf.reduce_mean(loss_val)

    return loss_val;

def plot_batch(d, y_true):
    dist_max = 3; # don't draw points after this distance.
    
    N = len(y_true);
    dists = np.asarray(d).reshape((-1))
    dists = np.minimum(dists, np.ones(dists.shape)*dist_max);
    
    x_data = np.zeros((N,));
    y_data = np.zeros((N,))
    
    color_data = []

    for i in range(N):
        x_data[i] = dists[i];
        y_data[i] = float(i) / float(N-1);
        if y_true[i] > 0.5: # matching
            color_data.append('g');
        else:
            color_data.append('r');

    # draw the plot
    global plot_batch_ax,plot_batch_fig;
    plot_batch_ax.clear()
    plot_batch_ax.scatter(x=x_data, y=y_data, c=color_data)
    plt.xlim(0, dist_max);
    plt.ylim(0, 1);
    plot_batch_fig.canvas.draw()
    plt.pause(0.0001)


def check_accuracy(dists_out_np, y_true):
    stats = {}
    
    dists = np.asarray(dists_out_np).reshape((-1))

    d = 0.5; # thresh distance
    match_correct = np.sum((dists<d)&(y_true==1))
    nomatch_correct = np.sum((dists>=d)&(y_true==0))
    acc = (match_correct + nomatch_correct)/(np.float(len(y_true)))

    if PLOT_BATCH:
        plot_batch(dists, y_true)
    
    stats['acc'] = acc;
    stats['avg_dist'] = np.mean(dists);
    return stats



class SiameseNet(tf.keras.Model):

    def __init__(self):
        super(SiameseNet, self).__init__()
        initializer = tf.variance_scaling_initializer(scale=2.0)
        self.conv1 = tf.layers.Conv2D(filters = 32, kernel_size = (7, 7), strides = (2, 2), padding = "SAME", activation = tf.nn.tanh, use_bias = True, kernel_initializer = initializer)
        self.pool1 = tf.layers.MaxPooling2D(pool_size = (2, 2), padding = "SAME", strides = (2, 2))
        self.norm1 = tf.layers.BatchNormalization(axis = 0, momentum = 0.99, epsilon = 1.0E-7)
        
        self.conv2 = tf.layers.Conv2D(filters = 64, kernel_size = (6, 6), strides = (3, 3), padding = "SAME", activation = tf.nn.tanh, use_bias = True, kernel_initializer = initializer)
        self.pool2 = tf.layers.MaxPooling2D(pool_size = (3, 3), padding = "SAME", strides = (3, 3))
        self.norm2 = tf.layers.BatchNormalization(axis = 0, momentum = 0.99, epsilon = 1.0E-7)
        
        self.conv3 = tf.layers.Conv2D(filters = 128, kernel_size = (5, 5), strides = (4, 4), padding = "SAME", activation = tf.nn.tanh, use_bias = True, kernel_initializer = initializer)
        self.pool3 = tf.layers.MaxPooling2D(pool_size = (4, 4), padding = "SAME", strides = (4, 4))
    
    # apply convnet; operates only on one of the left/right channels at a time.
    def apply_convnet(self, x):

        # CNN architecture
        
        x_out = self.conv1(x)
        x_out = self.pool1(x_out)
        x_out = self.norm1(x_out)
        
        x_out = self.conv2(x_out)
        x_out = self.pool2(x_out)
        x_out = self.norm2(x_out)
        
        x_out = self.conv3(x_out)
        x_out = self.pool3(x_out)
        
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
        
        # compute distance; we will pass this to the loss function
        dist_sq = tf.subtract(xL, xR);
        dist_sq = tf.multiply(dist_sq, dist_sq);
        dist_sq = tf.reduce_sum(dist_sq, axis=1)#, keepdims=True);
        dist = tf.sqrt(dist_sq);
        return dist

# ====== TRAINING ======
# Construct computational graph
tf.reset_default_graph()
with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float32, [None, 2, IMG_W, IMG_H])
    y = tf.placeholder(tf.float32, [None])
    dists_out = SiameseNet()(x);
    loss_vector_calc = hinge_embed_loss_func(y, dists_out);
    loss_scalar_calc = tf.reduce_mean(loss_vector_calc)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate);
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_scalar_calc)

def mine_one_batch(session_ref):
    
    X_unmined = np.zeros((0, 2, IMG_W, IMG_H))
    y_unmined = np.zeros((0,))
    loss_unmined = np.zeros((0,))
    for i in range(mining_ratio):
        X_batch, y_batch = training_dset.next()
        feed_dict = {x: X_batch, y: y_batch }
        
        # only forward prop
        loss_output = session_ref.run(loss_vector_calc, feed_dict=feed_dict)
        
        # save the results
        X_unmined = np.vstack((X_unmined, X_batch));
        y_unmined = np.concatenate((y_unmined, y_batch));
        loss_unmined = np.concatenate((loss_unmined, loss_output));

    evens = range(0,X_unmined.shape[0],2)
    odds = range(1,X_unmined.shape[0],2)

    X_unmined_p = X_unmined[evens]
    y_unmined_p = y_unmined[evens]
    loss_unmined_p = loss_unmined[evens]
    X_unmined_n = X_unmined[odds]
    y_unmined_n = y_unmined[odds]
    loss_unmined_n = loss_unmined[odds]

    idx_p = np.argsort(-loss_unmined_p)
    idx_n = np.argsort(-loss_unmined_n)
    X_mined_p = X_unmined_p[idx_p[0:batch_sz/2]]
    X_mined_n = X_unmined_n[idx_n[0:batch_sz/2]]
    y_mined_p = y_unmined_p[idx_p[0:batch_sz/2]]
    y_mined_n = y_unmined_n[idx_n[0:batch_sz/2]]

    X_mined = np.vstack((X_mined_p, X_mined_n))
    y_mined = np.concatenate((y_mined_p, y_mined_n))
    
    
    return X_mined, y_mined


# TODO: look into is_training flag, usually passed into feed_dict
# Run computational graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1;
    plt.ion()
    ct = 0;
    
    # ======= MINING =======
    for i in range(num_steps):
        X_batch, y_batch = mine_one_batch(sess)
    
        # check accuracy
        feed_dict = {x: X_batch, y: y_batch }
        dists_out_np = sess.run(dists_out, feed_dict=feed_dict)
        
        train_stats = check_accuracy(dists_out_np, y_batch)

        # do training step
        loss_output, _ = sess.run([loss_scalar_calc, train_op], feed_dict=feed_dict)

        # training progress log
        print 'Step', ('%6s' % step), '  |  ', \
                'Loss', ('%6s' % str(np.around(loss_output, 3))), '  |  ', \
                'Training Acc', (('%6s' % np.around(100.0*train_stats['acc'], 1)) + '%'), '  |  ', \
                'Avg Dist', ('%6s' % np.around(train_stats['avg_dist'], 3))


        if step >= num_steps or np.isnan(loss_output):
            break;
        step += 1

