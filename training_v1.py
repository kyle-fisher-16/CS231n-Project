import h5py

import cv2
import numpy as np


h5_file = h5py.File('patchdata_64x64.h5', 'r');

dataset_name = 'liberty';
dataset_infopath = dataset_name + '/info.txt';

raw_dataset = h5_file[dataset_name];

#raw_info = np.loadtxt(dataset_infopath);
raw_info = np.zeros((raw_dataset.shape[0],2));
raw_info = raw_info[:,0];

print raw_dataset.shape, raw_info.shape

class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
            Construct a Dataset object to iterate over data X and labels y

            Inputs:
            - X: Numpy array of data, of any shape
            - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
            - batch_size: Integer giving number of elements per minibatch
            - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
            """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.N = X.shape[0];
        self.idx = 0;
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        return self
    
    def next(self):
        
        # TODO: some shuffling of the data?
        
        if self.idx + self.batch_size < self.N: # TODO: END CONDITION
            batch_X = self.X[self.idx:self.idx+self.batch_size];
            batch_y = self.y[self.idx:self.idx+self.batch_size];
            
            # TODO: return pairs!
            
            return (batch_X, batch_y)
        else:
            raise StopIteration()

train_dset = Dataset(raw_dataset, raw_info, batch_size=64, shuffle=True)

for X, y in train_dset:
    print X.shape, y.shape
    break;




