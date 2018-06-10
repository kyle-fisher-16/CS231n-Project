import tensorflow as tf
import numpy.random as rng
import numpy as np
import os
import h5py

from dataset import Dataset

# Global Constants
IMG_W = 64
IMG_H = 64

dataset_limit = int(os.getenv('CS231N_DATASET_LIMIT', 100000000)); # limit the input dataset (for debugging)
batch_sz = int(os.getenv('CS231N_BATCH_SZ', 128))
num_epochs = int(os.getenv('CS231N_NUM_EPOCHS', 1000))
override_learning_rate = float(os.getenv('CS231N_OVERRIDE_LEARNING_RATE', 1e-3))
pct_validation = float(os.getenv('CS231N_PCT_VALIDATION', 10.0))
init_stddev = float(os.getenv('CS231N_INIT_STDDEV', 0.5))
num_filters_conv1 = int(os.getenv('CS231N_NUM_FILTERS_CONV1', 32))
num_filters_conv2 = int(os.getenv('CS231N_NUM_FILTERS_CONV2', 32))
device_name = '/'+str(os.getenv('CS231N_DEVICE_NAME', 'cpu')).lower()+':0'


# Display the params at stdout
print
print '===== ENVIRONMENT VARIABLES ====='
print 'Dataset Limit:', dataset_limit, 'examples'
print 'Batch Size:', batch_sz
print 'Number of Epochs:', num_epochs
print 'Learning Rate:', override_learning_rate
print 'Percent for Validation:', (str(pct_validation) + '%')
print 'Initialization Std. Dev.:', init_stddev
print 'device:', device_name

# ====== LOAD DATA ======
data = h5py.File('data/liberty.h5', 'r')

# ====== NETWORK ======
class BinaryClassNet(tf.keras.Model):
    
    def get_config(self):
        return {}
    
    def __init__(self):
        super(BinaryClassNet, self).__init__()
        initializer = tf.initializers.random_normal(stddev=init_stddev)
        
        # ====== LAYER 1 ======
        self.conv1 = tf.layers.Conv2D(filters = num_filters_conv1, kernel_size = (8, 8), strides = (1, 1), padding = "VALID", activation = tf.nn.relu, use_bias = True, kernel_initializer = initializer)
        self.conv2 = tf.layers.Conv2D(filters = num_filters_conv2, kernel_size = (4, 4), strides = (1, 1), padding = "VALID", activation = tf.nn.relu, use_bias = True, kernel_initializer = initializer)
        self.pool1 = tf.layers.MaxPooling2D(pool_size = (2, 2), padding = "VALID", strides = (2, 2))
        self.fc_out = tf.layers.Dense(2, activation=tf.nn.relu,
                            kernel_initializer=initializer)

    # apply convnet; operates only on one of the left/right channels at a time.
    def apply_convnet(self, x):
        x_out = self.conv1(x)
        x_out = self.pool1(x_out)
        x_out = self.conv2(x_out)
        x_out = self.pool1(x_out)
        x_out = tf.layers.flatten(x_out)
        x_out = self.fc_out(x_out)
        return x_out

    # execute the siamese net
    def call(self, x, training=None):
        xL = tf.slice(x, [0, 0, 0, 0], [batch_sz, 1, IMG_W, IMG_H]);
        xR = tf.slice(x, [0, 1, 0, 0], [batch_sz, 1, IMG_W, IMG_H]);
        
        # for conv2d, we need the dims to be ordered (batch_sz, img_w, img_h, channel)
        xL = tf.transpose(xL, [0, 2, 3, 1])
        xR = tf.transpose(xR, [0, 2, 3, 1])

        xAbsDiff = tf.abs(xL-xR)
        
        
        
        scores = self.apply_convnet(xAbsDiff)
        
#        scores = tf.Print(scores, [tf.shape(scores)], message="dimension of scores", summarize=10)

        return scores;



# ====== ACCURACY MEASUREMENT ======
def check_accuracy(scores_out_np, y_true):
    stats = {}
    
    scores_arr = np.asarray(scores_out_np).reshape((-1, 2))
    
    class_predict = np.argmax(scores_arr, axis=1)
    num_correct = np.sum((class_predict==y_true))
    acc = (num_correct)/(np.float(len(y_true)))

    stats['acc'] = acc;
    
    return stats

# ====== VAL ACCURACY ======
def get_val_acc(sess_ref, dset_ref, val_csv_filename=None):
    # X_valset and y_valset should be lists of np.arrays
    X_valset = dset_ref.val_dataset[0]
    y_valset = dset_ref.val_dataset[1]
    all_scores = np.zeros((0,2))
    all_y_true = np.zeros((0,), dtype=np.int32)
    for i in range(0, len(X_valset)):
        feed_dict = {x_tf: dset_ref.fetchImageData(X_valset[i]), y_tf: y_valset[i] }
        
        # check accuracy for this step
        scores_out_np = sess_ref.run(scores, feed_dict=feed_dict)
        all_scores = np.vstack((all_scores, scores_out_np))
        all_y_true = np.concatenate((all_y_true, y_valset[i]))
    
    val_acc_stats = check_accuracy(all_scores, all_y_true)
#    if val_csv_filename != None:
#        all_y_true = all_y_true.astype(np.int32)
#        np.savetxt(val_csv_filename, np.hstack((all_scores.reshape(-1,1), all_y_true.reshape(-1,1))))
    return val_acc_stats

# ====== TRAINING ======
# Construct computational graph
tf.reset_default_graph()
with tf.device(device_name):
    x_tf = tf.placeholder(tf.float32, [None, 2, IMG_W, IMG_H])
    y_tf = tf.placeholder(tf.int32, [None])

    bin_net = BinaryClassNet()
    scores = bin_net(x_tf);
    
    print scores.get_shape()
    print y_tf.get_shape()
    
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_tf, logits=scores)
    loss = tf.reduce_mean(losses)

    # save loss output for tensorboard:
    optimizer = tf.train.AdamOptimizer(learning_rate=override_learning_rate)
    train_op = optimizer.minimize(loss)
    #merged = tf.summary.merge_all()


# Run computational graph
with tf.Session() as sess:
    step = 1;
    val_acc_stats = {'acc': 0.0}
    ct = 0;
    best_val_acc = 0.0
    
    sess.run(tf.global_variables_initializer())
    
    for epoch_num in range(1,num_epochs+1, 1):
        print 'BEGINNING EPOCH #' + str(epoch_num)
        
        # use 10% of dset for validation
        training_dset = Dataset(data,batch_sz, pct_for_val=pct_validation, max_dataset_size=dataset_limit);

        while(True):
            try: # attempt another step
                X_batch, y_batch, pct_complete = training_dset.next()
                y_batch = np.asarray(y_batch, dtype="int32")
                feed_dict = {x_tf: training_dset.fetchImageData(X_batch), y_tf: y_batch}
                loss_output, scores_np, _ = sess.run([loss, scores, train_op], feed_dict=feed_dict)
                
                # check accuracy for this step
                train_stats = check_accuracy(scores_np, y_batch)
                
                print scores_np
                    
                # ======= LOGGING =======
                # print out to console
                print 'Step', ('%6s' % step), '  |  ', \
                    'Loss', ('%6s' % str(np.around(loss_output, 3))), '  |  ', \
                    'Training Acc', (('%6s' % np.around(100.0*train_stats['acc'], 1)) + '%')
                
                step += 1;
            except StopIteration: # epoch is over
                break;
    
        val_acc_stats = get_val_acc(sess, training_dset)
        print 'END EPOCH #' + str(epoch_num), '  |  ',\
            'Validation Acc', (('%6s' % np.around(100.0*val_acc_stats['acc'], 1)) + '%')





