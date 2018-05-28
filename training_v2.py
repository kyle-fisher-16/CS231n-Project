import tensorflow as tf
import numpy.random as rng
import numpy as np
import os
import h5py
from dataset import Dataset
from constants import GaussianKernel5x5
import matplotlib.pyplot as plt


# Global Constants
IMG_W = 64;
IMG_H = 64;
PLOT_BATCH = False; # whether or not to plot the batch distances
dataset_limit = 1000; # limit the input dataset (for debugging)

# ====== HYPERPARAMS ======
mining_ratio = 8;
batch_sz = 128;
num_epochs = 1000;
learning_rate = 5e-5;
pct_validation = 10.0;

# ====== LOAD DATA ======
data = h5py.File('data/liberty.h5', 'r')
#training_dset = Dataset(data, batch_size=batch_sz, max_dataset_size=3000);

# ===== TB SUMMARY DIR ======
tb_sum_dir = 'results/network/'

# ====== SETUP ======
if PLOT_BATCH:
    plot_batch_fig, plot_batch_ax = plt.subplots(1,1)
    plt.ion()
    plt.show()

# ====== LOSS FUNCTION ======
def hinge_embed_loss_func(y_true, dist):
    # y_true - should be boolean (int)... 0 if non-corresponding, 1 if matched.
#     dists - is distance from siamese network
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

def get_val_acc(sess_ref, dset_ref):
    # X_valset and y_valset should be lists of np.arrays
    X_valset = dset_ref.val_dataset[0]
    y_valset = dset_ref.val_dataset[1]
    all_dists = np.zeros((0,))
    all_y_true = np.zeros((0,))
    for i in range(0, len(X_valset)):
        feed_dict = {x: dset_ref.fetchImageData(X_valset[i]), y: y_valset[i] }
    
        # check accuracy for this step
        dists_out_np = sess_ref.run(dists_out, feed_dict=feed_dict)
        all_dists = np.concatenate((all_dists, dists_out_np))
        all_y_true = np.concatenate((all_y_true, y_valset[i]))
    val_acc_stats = check_accuracy(all_dists, all_y_true)
    return val_acc_stats


# this is a vanilla tf function which applys gaussian subnorm
def apply_guassian_subnorm_ch(x):
    # x is (IMG_W, IMG_H, filts)
    
    # x_reshaped becomes (filts, IMG_W, IMG_H)
    x_reshaped = tf.transpose(x, [2, 0, 1]);
    
    # sh is (filts, IMG_W, IMG_H)
    sh = tf.shape(x_reshaped);
     
    # x_reshaped becomes (filts, IMG_W, IMG_H, 1)
    x_reshaped = tf.reshape(x, (sh[0], sh[1], sh[2], 1));

    # apply the filter kernel
    global GaussianKernel5x5;
    result = x_reshaped - tf.nn.conv2d(x_reshaped, GaussianKernel5x5, strides=(1, 1, 1, 1), padding="SAME");

    # result shape is now (filts, IMG_W, IMG_H, 1). need to fix.
    # make result be (filts, IMG_W, IMG_H)
    result = tf.reshape(result, sh)
    result = tf.transpose(x, [1, 2, 0]);
    
    return result;



class SiameseNet(tf.keras.Model):

    def __init__(self):
        super(SiameseNet, self).__init__()
#        initializer = tf.variance_scaling_initializer(scale=2.0)
        initializer = tf.initializers.random_normal(stddev=2.0)
        self.conv1 = tf.layers.Conv2D(filters = 32, kernel_size = (7, 7), strides = (2, 2), padding = "SAME", activation = tf.nn.tanh, use_bias = True, kernel_initializer = initializer)
        self.pool1 = tf.layers.MaxPooling2D(pool_size = (2, 2), padding = "SAME", strides = (2, 2))
#        self.norm1 = tf.layers.BatchNormalization(axis = 0, momentum = 0.99, epsilon = 1.0E-7)

        self.conv2 = tf.layers.Conv2D(filters = 64, kernel_size = (6, 6), strides = (3, 3), padding = "SAME", activation = tf.nn.tanh, use_bias = True, kernel_initializer = initializer)
        self.pool2 = tf.layers.MaxPooling2D(pool_size = (3, 3), padding = "SAME", strides = (3, 3))
#        self.norm2 = tf.layers.BatchNormalization(axis = 0, momentum = 0.99, epsilon = 1.0E-7)

        self.conv3 = tf.layers.Conv2D(filters = 128, kernel_size = (5, 5), strides = (4, 4), padding = "SAME", activation = tf.nn.tanh, use_bias = True, kernel_initializer = initializer)
        self.pool3 = tf.layers.MaxPooling2D(pool_size = (4, 4), padding = "SAME", strides = (4, 4))

    
    # apply convnet; operates only on one of the left/right channels at a time.
    def apply_convnet(self, x):

        # CNN architecture
        x_out = self.conv1(x)
        x_out = self.apply_L2_Pool(x_out, 2, 2)
        x_out = self.apply_guassian_subnorm(x_out)
        
        #x_out = self.norm1(x_out)
        
        x_out = self.conv2(x_out)
        x_out = self.apply_L2_Pool(x_out, 3, 3)
        x_out = self.apply_guassian_subnorm(x_out)
        
        x_out = self.conv3(x_out)
        x_out = self.apply_L2_Pool(x_out, 4, 4)
        
        # flatten because at the end we want a single descriptor per input
        x_out = tf.layers.flatten(x_out);
        return x_out;
    

    # spatial guassian subtractive normalization w/ 5x5xFxF kernel
    def apply_guassian_subnorm(self, x_in):
        # x_in starts as (batch_sz, IMG_W, IMG_H, filts)
        return tf.map_fn(apply_guassian_subnorm_ch, x_in);
        return x_in - sub;
    
    def apply_L2_Pool(self, x, window, stride):
        return tf.nn.max_pool(value=x, ksize=(1, window, window, 1), strides=(1, stride, stride, 1), padding="SAME")
        #return tf.sqrt(tf.nn.avg_pool(value=tf.square(x), ksize=(1, window, window, 1), strides=(1, stride, stride, 1), padding="SAME"))
        #return tf.nn.max_pool(value=tf.square(x), ksize=(1, window, window, 1), strides=(1, stride, stride, 1), padding="SAME")
    
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
        
        # Slightly perturb one of these so they can't be identical
        xL = tf.add(xL, 0.0001);
        
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
    # place holders for graph stuff??
    train_acc_tf = tf.placeholder(tf.float32)
    tf.summary.scalar("Training Accuracy", train_acc_tf)
    val_acc_tf = tf.placeholder(tf.float32)
    tf.summary.scalar("Validation Accuracy", val_acc_tf)
    dists_out = SiameseNet()(x);
    loss_vector_calc = hinge_embed_loss_func(y, dists_out);
    loss_scalar_calc = tf.reduce_mean(loss_vector_calc)
    # save loss output for tensorboard:
    tf.summary.scalar('loss', loss_scalar_calc)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_scalar_calc)
    capped_grads = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads]
    train_op = optimizer.apply_gradients(capped_grads)
    merged = tf.summary.merge_all()

def mine_one_batch(session_ref, dataset_ref):
    X_unmined = np.zeros((0, 4), dtype="uint32")
    y_unmined = np.zeros((0,), dtype="uint32")
    loss_unmined = np.zeros((0,), dtype="float32")
    for i in range(mining_ratio):
        try:
            X_batch, y_batch, pct_complete = dataset_ref.next()
        except StopIteration:
            return None, None, None;
        feed_dict = {x: dataset_ref.fetchImageData(X_batch), y: y_batch }
        
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
    
    return X_mined, y_mined, pct_complete


# TODO: look into is_training flag, usually passed into feed_dict
# Run computational graph
with tf.Session() as sess:
    tb_train_writer = tf.summary.FileWriter(tb_sum_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    step = 1;
    val_acc_stats = {'acc': 0.0}
    plt.ion()
    ct = 0;
    
    for epoch_num in range(1,num_epochs+1, 1):
        print 'BEGINNING EPOCH #' + str(epoch_num)
        

        # use 10% of dset for validation
        training_dset = Dataset(data,batch_sz, pct_for_val=pct_validation, max_dataset_size=dataset_limit);
        
        # ======= MINING =======
        print 'Mining...'
        mined_batches = [] # set of mined batches
        while True:
            X_batch, y_batch, pct_complete = mine_one_batch(sess, training_dset)
            if X_batch is None:
                break;
            print str(np.around(pct_complete, 1)) + '% mined'
            mined_batches.append((X_batch, y_batch))
        print 'Done mining!'
        
        # ======= TRAINING =======
        for X_mined, y_mined in mined_batches:
            feed_dict = {x: training_dset.fetchImageData(X_mined),
                         y: y_mined}
            
            # check accuracy for this step
            dists_out_np = sess.run(dists_out, feed_dict=feed_dict)
            train_stats = check_accuracy(dists_out_np, y_mined)
 
            # do training step
            feed_dict = {x: training_dset.fetchImageData(X_mined),
                        y: y_mined,
                        train_acc_tf: 100.0*train_stats['acc'],
                        val_acc_tf: 100.0*val_acc_stats['acc']}

            loss_output, summary, _ = sess.run([loss_scalar_calc, merged, train_op], feed_dict=feed_dict)
            tb_train_writer.add_summary(summary)
            tb_train_writer.flush()

            # ======= LOGGING =======
            # print out to console
            print 'Step', ('%6s' % step), '  |  ', \
                    'Loss', ('%6s' % str(np.around(loss_output, 3))), '  |  ', \
                    'Training Acc', (('%6s' % np.around(100.0*train_stats['acc'], 1)) + '%'), '  |  ', \
                    'Avg Dist', ('%6s' % np.around(train_stats['avg_dist'], 3))

            if np.isnan(loss_output):
                break;
            step += 1

        # check validation accuracy
        val_acc_stats = get_val_acc(sess, training_dset)
        print 'END EPOCH #' + str(epoch_num), '  |  ',\
            'Validation Acc', (('%6s' % np.around(100.0*val_acc_stats['acc'], 1)) + '%')
