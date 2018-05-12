# takes in 1 command-line argument: dataset name (e.g. "liberty")

import h5py
import numpy as np
import sys

from collections import defaultdict
h5_file = h5py.File('data/images.h5', 'r')

dataset_name = sys.argv[1];
dataset_infopath = 'data/' + dataset_name + '.txt'
raw_dataset = h5_file[dataset_name]
raw_info = np.loadtxt(dataset_infopath)
raw_info = raw_info[:,0]
out_file = h5py.File('data/' + dataset_name + '.h5', 'w')


# loop over all data
prev_patch_idx = raw_info[0]
patch_group = raw_dataset[0]
for i in xrange(1, raw_dataset.shape[0]):
  new_patch_idx = raw_info[i]
  if new_patch_idx != prev_patch_idx:
    patch_key = str(int(prev_patch_idx));
    dset = out_file.create_dataset(patch_key, patch_group.shape, dtype='uint8')
    dset = patch_group
    patch_group = np.zeros((0, 4096))
  patch_group = np.vstack((patch_group, raw_dataset[i]))
  prev_patch_idx = new_patch_idx
