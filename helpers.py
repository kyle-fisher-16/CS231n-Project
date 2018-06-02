import tensorflow as tf
import numpy.random as rng
import numpy as np
import os
import h5py
from dataset import Dataset
from constants import GaussianKernel5x5
import matplotlib.pyplot as plt
import datetime

def save_stats(filename, epoch, val_acc, best_val_acc, current_step, threshold, avg_pos_dist, avg_neg_dist, var_pos_dist, var_neg_dist):
    try:
        text_file = open(filename, "w")
        memos = []
        memos.append(str(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
        memos.append("Epoch: " + str(epoch))
        memos.append("Val Acc: " + str(val_acc))
        memos.append("Record Val Acc: " + str(best_val_acc))
        memos.append("Current Step #: " + str(current_step))
        memos.append("Threshold: " + str(threshold))
        memos.append("Avg Distance (Positives): " + str(avg_pos_dist))
        memos.append("Avg Distance (Negatives): " + str(avg_neg_dist))
        memos.append("Variance of Distances (Positives): " + str(var_pos_dist))
        memos.append("Variance of Distances (Negatives): " + str(var_neg_dist))
        
        for memo in memos:
            text_file.write(memo + "\n")
        
        
        text_file.close()
    except:
        print "save_stats(): Failed to write", filename

# this is a vanilla tf function which applys gaussian subnorm
def apply_guassian_subnorm_ch(x):
    
    # x is (IMG_W, IMG_H, filts)
    
    # x_reshaped becomes (filts, IMG_W, IMG_H)
    x_reshaped = tf.transpose(x, [2, 0, 1]);
    
    # sh is (filts, IMG_W, IMG_H)
    sh = tf.shape(x_reshaped);
    
    # x_reshaped becomes (filts, IMG_W, IMG_H, 1)
    x_reshaped = tf.reshape(x_reshaped, (sh[0], sh[1], sh[2], 1));
    
    paddings = tf.constant([[0,0],[2,2],[2,2], [0,0]])
    x_padded = tf.pad(x_reshaped, paddings, "SYMMETRIC")
    
    # apply the filter kernel
    global GaussianKernel5x5;
    result = x_reshaped - tf.nn.conv2d(x_padded, GaussianKernel5x5, strides=(1, 1, 1, 1), padding="VALID");
    
    # result shape is now (filts, IMG_W, IMG_H, 1). need to fix.
    # make result be (filts, IMG_W, IMG_H)
    result = tf.reshape(result, sh)
    result = tf.transpose(result, [1, 2, 0]);
    
    return result;

# x_in: prev layer input (batch_sz, IMG_W, IMG_H, filts)
# conv_array: the array of convolution layers (made in SiameseNet init)
# conn_idx: the tf.constant array of indices specifying the connections (also made in SiameseNet init)
def apply_sparse_connected_conv(x_in, conv_array, conn_idx):
    # iterate through all the output filters
    num_filters = len(conn_idx)
    print 'Building sparse conv with', num_filters, 'filters...'
    x_out = [];
    for out_ch in range(num_filters):
        # cut out a slice from x_in
        input_filts_idx = conn_idx[out_ch];
        input_slice = tf.gather(x_in, input_filts_idx, axis=3)
        # apply the conv
        conv_out = conv_array[out_ch](input_slice)
        # built the list of output filter tensors
        x_out.append(conv_out)
    
    x_out = tf.concat(x_out,axis=3) # concat these down along filter axis
    print 'Done!'
    return x_out


def generate_model_connections(num_connections, num_input_layers, num_output_layers):
    result = []
    in_layer_idxs = np.zeros((num_connections), dtype="int32")
    for i in range(0,num_output_layers):
        connections = np.sort(np.random.choice(range(0,num_input_layers),num_connections, replace=False)).reshape(-1)
        in_layer_idxs = tf.constant(connections, dtype="int32")
        result.append(in_layer_idxs)
    return result;

