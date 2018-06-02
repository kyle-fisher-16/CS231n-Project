import tensorflow as tf
import numpy.random as rng
import numpy as np
import os
import h5py
from dataset import Dataset
from constants import GaussianKernel5x5
import matplotlib.pyplot as plt
from helpers import apply_guassian_subnorm_ch, apply_sparse_connected_conv, generate_model_connections, save_stats


# Global Constants
IMG_W = 64
IMG_H = 64
DESCRIPTOR_SZ = 128

# ====== HYPERPARAMS ======
restore_model_path = str(os.getenv('CS231N_RESTORE_MODEL_PATH', None))
PLOT_BATCH = str(os.getenv('CS231N_PLOT_BATCH', False)).lower().startswith("t"); # whether or not to plot the batch distances
dataset_limit = int(os.getenv('CS231N_DATASET_LIMIT', 100000000)); # limit the input dataset (for debugging)
mining_ratio = int(os.getenv('CS231N_MINING_RATIO', 8))
batch_sz = int(os.getenv('CS231N_BATCH_SZ', 128))
num_epochs = int(os.getenv('CS231N_NUM_EPOCHS', 1000))
override_learning_rate = float(os.getenv('CS231N_OVERRIDE_LEARNING_RATE', 0.01))
pct_validation = float(os.getenv('CS231N_PCT_VALIDATION', 10.0))
pooling_type = str(os.getenv('CS231N_POOLING_TYPE', 'l2')).lower() # l2 or max
init_stddev = float(os.getenv('CS231N_INIT_STDDEV', 0.5))
num_filters_conv1 = int(os.getenv('CS231N_NUM_FILTERS_CONV1', 32))
num_filters_conv2 = int(os.getenv('CS231N_NUM_FILTERS_CONV2', 64))
num_filters_conv3 = DESCRIPTOR_SZ
conv_connectivity = int(os.getenv('CS231N_CONV_CONNECTIVITY', 8)) # the number of input channels that a sparse conv depends on
use_sparsity = str(os.getenv('CS231N_USE_SPARSITY', True)).lower().startswith("t")
saved_models_dir = str(os.getenv('CS231N_SAVED_MODELS_DIR', 'results/model/')).lower()
saved_model_prefix = str(os.getenv('CS231N_SAVED_MODEL_PREFIX', 'sess')).lower()
saves_stats_dir = str(os.getenv('CS231N_SAVED_STATS_DIR', 'results/train_stats/')).lower()
device_name = '/'+str(os.getenv('CS231N_DEVICE_NAME', 'cpu')).lower()+':0'

# Check input args
if (not (pooling_type=='l2' or pooling_type=='max')):
    raise ValueError('Environment variable CS231N_POOLING_TYPE must be "l2" or "max"');

# Display the params at stdout
print
print '===== ENVIRONMENT VARIABLES ====='
print 'Restore Model:', restore_model_path
print 'Plotting Enabled:', PLOT_BATCH
print 'Dataset Limit:', dataset_limit, 'examples'
print 'Mining Ratio:', mining_ratio
print 'Batch Size:', batch_sz
print 'Pooling Type:', pooling_type
print 'Number of Epochs:', num_epochs
print 'Learning Rate:', override_learning_rate
print 'Percent for Validation:', (str(pct_validation) + '%')
print 'Initialization Std. Dev.:', init_stddev
print 'model save location:', saved_models_dir
print 'prefix of saved model:', saved_model_prefix
print 'device:', device_name
#print 'Initial Matching Threshold.:', matching_threshold
print

# ====== LOAD DATA ======
data = h5py.File('data/liberty.h5', 'r')
#training_dset = Dataset(data, batch_size=batch_sz, max_dataset_size=3000);

# ====== SETUP ======
# tensorboard
tb_sum_dir = 'results/network/'



# plotting
if PLOT_BATCH:
    plot_batch_fig, plot_batch_ax = plt.subplots(1,1)
    plt.ion()
    plt.show()


def plot_batch(d, y_true, thresh):
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
    plt.plot([thresh, thresh], [0.0, 1.0], color='k', linestyle='-', linewidth=2)
    plot_batch_fig.canvas.draw()
    plt.pause(0.0001)

# ====== OPTIMAL THRESHOLD ESTIMATION ======
def update_threshold(dists_out_np, y_true, matching_threshold, thresh_vel):
    dists = np.asarray(dists_out_np).reshape((-1))
    offset = 0.01;
    # to do: initialize and track matching threshold
    upper_threshold = matching_threshold + offset
    lower_threshold = matching_threshold - offset
    # get accuracies at thresholds
    acc = calcAcc(dists, matching_threshold, y_true)
    upper_acc = calcAcc(dists, upper_threshold, y_true)
    lower_acc = calcAcc(dists, lower_threshold, y_true)
    # if accuracy is different enough from offset accuracy
    if (((upper_acc - acc) > 0.0001) or ((lower_acc - acc) > 0.0001)):
        # update threshold vel in direction of accuracy increase
        dAcc, dsgn = max((upper_acc - acc, 1), (lower_acc - acc, -1))
        thresh_vel += dAcc * dsgn * 0.1 - 0.0001 * thresh_vel
        matching_threshold += thresh_vel * 0.1
    # update threshold
    return matching_threshold, thresh_vel

def calcAcc(dists, thresh, y):
    match_correct = np.sum((dists < thresh) & (y == 1))
    nomatch_correct = np.sum((dists >= thresh) & (y == 0))
    return (match_correct + nomatch_correct) / (np.float(len(y)))
    
'''
def compute_best_threshold(dists_out_np, y_true):
    dists = np.asarray(dists_out_np).reshape((-1))
    # check in a small neighborhood of the current best threshold
    offsets = np.asarray(range(-5, 6, 1), dtype="float32") / 100.0
    thresh_set = matching_threshold + offsets;
    thresh_set = np.maximum(0.0, thresh_set) # clamp at 0.0
    
    best_acc = 0.5 # acc to beat.
    best_thresh = matching_threshold
    for d in thresh_set:
        match_correct = np.sum((dists<d)&(y_true==1))
        nomatch_correct = np.sum((dists>=d)&(y_true==0))
        acc = (match_correct + nomatch_correct)/(np.float(len(y_true)))
        if acc > best_acc:
            best_thresh = d;
            best_acc = acc
    print 'best_thresh', best_thresh
    return best_thresh
'''

# ====== ACCURACY MEASUREMENT ======
def check_accuracy(dists_out_np, y_true, thresh):
    stats = {}
    
    dists = np.asarray(dists_out_np).reshape((-1))
    
    '''d = 0.5; # thresh distance'''
    d = thresh
    match_correct = np.sum((dists<d)&(y_true==1))
    nomatch_correct = np.sum((dists>=d)&(y_true==0))
    acc = (match_correct + nomatch_correct)/(np.float(len(y_true)))
    
    if PLOT_BATCH:
        plot_batch(dists, y_true, thresh)
    
    avg_pos_dist = np.mean(dists[y_true==1])
    var_pos_dist = np.var(dists[y_true==1])

    avg_neg_dist = np.mean(dists[y_true==0])
    var_neg_dist = np.var(dists[y_true==0])
    
    stats['acc'] = acc;
    stats['avg_dist'] = np.mean(dists);
    stats['avg_pos_dist'] = avg_pos_dist;
    stats['avg_neg_dist'] = avg_neg_dist;
    stats['var_pos_dist'] = var_pos_dist;
    stats['var_neg_dist'] = var_neg_dist;
    return stats

# ====== VAL ACCURACY ======
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
    '''best_threshold = compute_best_threshold(all_dists, all_y_true)'''
    dset_ref.thresh, dset_ref.thresh_v = update_threshold(all_dists, all_y_true, dset_ref.thresh, dset_ref.thresh_v)
    '''val_acc_stats = check_accuracy(all_dists, all_y_true)'''
    val_acc_stats = check_accuracy(all_dists, all_y_true, dset_ref.thresh)
    return val_acc_stats


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

    return loss_val;



# ====== NETWORK ======
class SiameseNet(tf.keras.Model):
    
    def get_config(self):
        return {}

    def __init__(self):
        super(SiameseNet, self).__init__()
        initializer = tf.initializers.random_normal(stddev=init_stddev)
        
        # ====== LAYER 1 ======
        self.conv1 = tf.layers.Conv2D(filters = num_filters_conv1, kernel_size = (7, 7), strides = (2, 2), padding = "VALID", activation = tf.nn.tanh, use_bias = True, kernel_initializer = initializer)
        self.avg_pool1 = tf.layers.AveragePooling2D(pool_size = (2, 2), padding = "VALID", strides = (1, 1))
        self.max_pool1 = tf.layers.MaxPooling2D(pool_size = (2, 2), padding = "VALID", strides = (1, 1))
        
        # ====== LAYER 2 ======
        layer2_kernel_size = (6, 6)
        layer2_strides = (3, 3)
        # FULL CONV
        self.conv2_fully_connected = tf.layers.Conv2D(filters = num_filters_conv2, kernel_size = layer2_kernel_size, strides = layer2_strides, padding = "VALID", activation = tf.nn.tanh, use_bias = True, kernel_initializer = initializer)
        # SPARSE CONV
        self.conv2_conns = generate_model_connections(conv_connectivity, num_filters_conv1, num_filters_conv2)
        self.conv2_sparse = []
        for i in range(num_filters_conv2):
            conv_layer = tf.layers.Conv2D(filters = 1, kernel_size = layer2_kernel_size, strides = layer2_strides, padding = "VALID", activation = tf.nn.tanh, use_bias = True, kernel_initializer = initializer)
            self.conv2_sparse.append(conv_layer)
        # POOLING
        self.avg_pool2 = tf.layers.AveragePooling2D(pool_size = (3, 3), padding = "VALID", strides = (1, 1))
        self.max_pool2 = tf.layers.MaxPooling2D(pool_size = (3, 3), padding = "VALID", strides = (1, 1))
        
        # ====== LAYER 3 ======
        layer3_kernel_size = (4, 4)
        layer3_strides = (3, 3)
        # FULL CONV
        self.conv3_fully_connected = tf.layers.Conv2D(filters = num_filters_conv3, kernel_size = layer3_kernel_size, strides = layer3_strides, padding = "VALID", activation = tf.nn.tanh, use_bias = True, kernel_initializer = initializer)
        # SPARSE CONV
        self.conv3_conns = generate_model_connections(conv_connectivity, num_filters_conv2, num_filters_conv3)
        self.conv3_sparse = []
        for i in range(num_filters_conv3):
            conv_layer = tf.layers.Conv2D(filters = 1, kernel_size = layer3_kernel_size, strides = layer3_strides, padding = "VALID", activation = tf.nn.tanh, use_bias = True, kernel_initializer = initializer)
            self.conv3_sparse.append(conv_layer)
        # POOLING
        self.avg_pool3 = tf.layers.AveragePooling2D(pool_size = (2, 2), padding = "VALID", strides = (2, 2))
        self.max_pool3 = tf.layers.MaxPooling2D(pool_size = (2, 2), padding = "VALID", strides = (2, 2))


    
    # apply convnet; operates only on one of the left/right channels at a time.
    def apply_convnet(self, x):

        # LAYER 1
        x_out = self.conv1(x)
        paddings = tf.constant([[0,0],[1,0],[1,0], [0,0]])
        x_out = tf.pad(x_out, paddings, "SYMMETRIC")
        if (pooling_type=='l2'):
            x_out = self.apply_L2_Pool(x_out, self.avg_pool1)
        elif (pooling_type=='max'):
            x_out = self.max_pool1(x_out)
        x_out = self.apply_guassian_subnorm(x_out)

        # LAYER 2
        if use_sparsity:
            x_out = apply_sparse_connected_conv(x_out, self.conv2_sparse, self.conv2_conns);
        else: # no sparsity
            x_out = self.conv2_fully_connected(x_out)
        paddings = tf.constant([[0,0],[1,1],[1,1], [0,0]])
        x_out = tf.pad(x_out, paddings, "SYMMETRIC")
        if (pooling_type=='l2'):
            x_out = self.apply_L2_Pool(x_out, self.avg_pool2)
        elif (pooling_type=='max'):
            x_out = self.max_pool2(x_out)
        x_out = self.apply_guassian_subnorm(x_out)

        # LAYER 3
        if use_sparsity:
            x_out = apply_sparse_connected_conv(x_out, self.conv3_sparse, self.conv3_conns);
        else: # no sparsity
            x_out = self.conv3_fully_connected(x_out)
        if (pooling_type=='l2'):
            x_out = self.apply_L2_Pool(x_out, self.avg_pool3)
        elif (pooling_type=='max'):
            x_out = self.max_pool3(x_out)
        
        # flatten because at the end we want a single descriptor per input
        x_out = tf.layers.flatten(x_out);
#        x_out = tf.Print(x_out, [tf.shape(x_out)], message="dimension of final output", summarize=10)
        return x_out;

    def apply_sparse_connects(self, x_in):
        return tf.map_fn(apply_sparse_connects_ch, x_in)
    
    # spatial guassian subtractive normalization w/ 5x5xFxF kernel
    def apply_guassian_subnorm(self, x_in):
        # x_in starts as (batch_sz, IMG_W, IMG_H, filts)
        return tf.map_fn(apply_guassian_subnorm_ch, x_in);
    
    def apply_L2_Pool(self, x, pooler_layer_ref):
        out = tf.square(x)
        out = pooler_layer_ref(out)
        out = tf.sqrt(out)
        return out
    
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
with tf.device(device_name):
    x = tf.placeholder(tf.float32, [None, 2, IMG_W, IMG_H])
    y = tf.placeholder(tf.float32, [None])
    # place holders for graph stuff??
    train_acc_tf = tf.placeholder(tf.float32)
    #tf.summary.scalar("Training Accuracy", train_acc_tf)
    val_acc_tf = tf.placeholder(tf.float32)
    #tf.summary.scalar("Validation Accuracy", val_acc_tf)
    snet = SiameseNet()
    dists_out = snet(x);
    loss_vector_calc = hinge_embed_loss_func(y, dists_out);
    loss_scalar_calc = tf.reduce_mean(loss_vector_calc)
    # save loss output for tensorboard:
    #tf.summary.scalar('loss', loss_scalar_calc)
    Gstep = tf.Variable(0.0, trainable=False)
    learning_rate = tf.train.exponential_decay(override_learning_rate, Gstep, 10000, 10)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    grads = optimizer.compute_gradients(loss_scalar_calc)
    capped_grads = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads]
    train_op = optimizer.apply_gradients(capped_grads, global_step=Gstep)
    #merged = tf.summary.merge_all()

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

    # ======= RESTORE MODEL? =======
    if restore_model_path != None:
        print 'Restoring from', restore_model_path, '...'
        tf.train.Saver().restore(sess,  restore_model_path)
        print 'Done.'
    else: 
        sess.run(tf.global_variables_initializer())

    # Stats, plottig, etc
    step = 1;
    val_acc_stats = {'acc': 0.0}
    plt.ion()
    ct = 0;
    best_val_acc = 0.0
    
    for epoch_num in range(1,num_epochs+1, 1):
        print 'BEGINNING EPOCH #' + str(epoch_num)
        if epoch_num > 1:
            thresh = training_dset.thresh
            print "Threshold: ", thresh
            thresh_v = training_dset.thresh_v

        # use 10% of dset for validation
        training_dset = Dataset(data,batch_sz, pct_for_val=pct_validation, max_dataset_size=dataset_limit);
        if epoch_num > 1:
            training_dset.thresh = thresh
            training_dset.thresh_v = thresh_v
        
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
            '''train_stats = check_accuracy(dists_out_np, y_mined)'''
            train_stats = check_accuracy(dists_out_np, y_mined, training_dset.thresh)
 
            # do training step
            feed_dict = {x: training_dset.fetchImageData(X_mined),
                        y: y_mined,
                        train_acc_tf: 100.0*train_stats['acc'],
                        val_acc_tf: 100.0*val_acc_stats['acc']}

            loss_output, _ = sess.run([loss_scalar_calc, train_op], feed_dict=feed_dict)
            #tb_train_writer.add_summary(summary)
            #tb_train_writer.flush()

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
        if val_acc_stats['acc'] > best_val_acc:
            best_val_acc = val_acc_stats['acc']
        print "Saving model..."
        ret = tf.train.Saver().save(sess, saved_models_dir + saved_model_prefix + '_' + str(epoch_num) + '.ckpt')
        print "Saved model to", ret
        
        # ======= SAVE STATS TO FILE =======
        save_stats(saves_stats_dir+str(epoch_num)+".txt",
                   epoch=epoch_num,
                   val_acc=val_acc_stats['acc'],
                   best_val_acc=best_val_acc,
                   current_step=step,
                   threshold=training_dset.thresh,
                   avg_pos_dist=val_acc_stats['avg_pos_dist'],
                   avg_neg_dist=val_acc_stats['avg_neg_dist'],
                   var_pos_dist=val_acc_stats['var_pos_dist'],
                   var_neg_dist=val_acc_stats['var_neg_dist']);
        
        print 'END EPOCH #' + str(epoch_num), '  |  ',\
            'Validation Acc', (('%6s' % np.around(100.0*val_acc_stats['acc'], 1)) + '%')
