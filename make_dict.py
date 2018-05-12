import h5py
import numpy as np
from collections import defaultdict


h5_file = h5py.File('data/images.h5', 'r')

# pick a specific dataset to dictionarize
dataset_name = 'liberty'
dataset_infopath = 'data/' + dataset_name + '.txt'
raw_dataset = h5_file[dataset_name]
raw_info = np.loadtxt(dataset_infopath)
raw_info = raw_info[:,0]
#
#
#
#h = h5py.File('myfile.hdf5')
#


out_file = h5py.File('myfile.hdf5', 'w')


prev_patch_idx = raw_info[0]
patch_group = raw_dataset[0]

for i in xrange(1, raw_dataset.shape[0]):
  new_patch_idx = raw_info[i]

  if new_patch_idx != prev_patch_idx:
    patch_key = str(int(prev_patch_idx));
    print 'DEBUG', patch_key
    out_file.create_dataset(patch_key, patch_group.shape, dtype='uint8')
#    out_file.create_dataset("mydataset", (100,), dtype='i')

#    out_file[patch_key] = patch_group

    patch_group = np.zeros((0, 4096))

  patch_group = np.append(patch_group, raw_dataset[i])

  prev_patch_idx = new_patch_idx;

  if i > 100:
    break;



#
#
#  h.create_dataset(raw_dataset[i].reshape((64, 64)),
#                   data=np.array(v, dtype=np.int8));
#





#data_dict = defaultdict(lambda: [])
#
#for i in range(raw_dataset.shape[0]):
#  data_dict[raw_info[i]] += [raw_dataset[i].reshape((64, 64))]
#  if (i == 50):
#    break;
#
#print len(data_dict[1])
#
#
#h5_file = h5py.File('data/images.h5', 'w')



