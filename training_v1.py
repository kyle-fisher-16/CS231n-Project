#!/usr/bin/python

import h5py
import numpy as np

data = h5py.File('data/liberty.h5', 'r')

class Dataset(object):

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.nGroups = int(self.data.keys()[-1])
        self.groupIdx = xrange(self.nGroups + 1)

    def __iter__(self):
        return self

    def next(self):
        # initialize X and y
        X = np.zeros((self.batch_size, 2, 64, 64))
        y = np.zeros(self.batch_size,)
        # for i in batch size
        for i in xrange(self.batch_size):
            # if i is even
            if i % 2 == 0:
                # generate a positive example
                example = self.generatePositiveExample()
            # if i is odd
            else:
                # generate a negative example
                example = self.generatePositiveExample()
            # store example
            X[i] = example.copy()
            # store label
            y[i] = int(i % 2 == 0)
        # return examples and labels
        return X, y
    
    def generatePositiveExample(self):
        # generate random 3D index
        grpIdx = np.random.choice(self.groupIdx).astype(str)
        # generate distinct patch pair
        patchIdx = xrange(self.data[grpIdx].shape[0])
        pchIdx = np.random.choice(patchIdx, 2)
        # assemble example
        example = np.zeros((2, 64, 64))
        example[0] = np.reshape(data[grpIdx][pchIdx[0]], (64, 64))
        example[1] = np.reshape(data[grpIdx][pchIdx[1]], (64, 64))
        # return example
        return example

    def generateNegativeExample(self):
        # generate random 3D index pair
        grpIdx = np.random.choice(self.groupIdx, 2).astype(str)
        # generate random patch index pair
        pchSize = (self.data[grpIdx[0]].shape[0],
                   self.data[grpIdx[1]].shape[0])
        pchIdx = (np.random.randint(pchSize[0]),
                  np.random.randint(pchSize[1]))
        # assemble example
        example = np.zeros((2, 64, 64))
        example[0] = np.reshape(data[grpIdx[0]][pchIdx[0]], (64, 64))
        example[1] = np.reshape(data[grpIdx[1]][pchIdx[1]], (64, 64))
        return example      


if __name__ == '__main__':
    train_dset = Dataset(data, batch_size=5)
    X, y = train_dset.next()

