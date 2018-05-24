#!/usr/bin/python

import h5py
import numpy as np
from itertools import combinations as comb
import pdb

data = h5py.File('data/liberty.h5', 'r')

class Dataset(object):

    def __init__(self, data, batch_size, max_dataset_size=np.inf):
        self.data = data
        self.batch_size = batch_size
        
        self.nGroups = int(self.data.keys()[-1])
        self.nGroups = np.minimum(self.nGroups, max_dataset_size)
        
        self.groupIdx = xrange(self.nGroups + 1)
        self._pDict = self._getAllPositiveExampleIndices()
        self._keypoint = 0
        self._combIdx = 0
        self._maxKeypoints = self.nGroups + 1

    def __iter__(self):
        return self

    def next(self):
        # initialize X and y
        X = np.zeros((self.batch_size, 2, 64, 64), dtype = "uint8")
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
                example = self.generateNegativeExample()
            # store example
            X[i] = example.copy()
            # store label
            y[i] = int(i % 2 == 0)
        # return examples and labels
        return X, y
    
    def generatePositiveExample(self):
        # get the current positive example
        exIdx = self._pDict[self._keypoint][self._combIdx]
        example = self.data[str(self._keypoint)][exIdx, :]
        example = np.reshape(example, (2, 64, 64))
        # increment the comb index
        self._combIdx += 1
        # if the comb index equals the number of combs
        if (self._combIdx == len(self._pDict[self._keypoint])):
            # reset the comb index to 0
            self._combIdx = 0
            # increment the keypoint
            self._keypoint += 1
            # if the keypoint equals the number of dictionary items
            if (self._keypoint == self._maxKeypoints):
                # reset the keypoint
                self._keypoint = 0
                # raise the stop iteration flag
                raise StopIteration
        # return the example
        return example
    
    '''
    def generatePositiveExample(self):
        # generate random 3D index
        grpIdx = np.random.choice(self.groupIdx).astype(str)
        # generate distinct patch pair
        patchIdx = xrange(self.data[grpIdx].shape[0])
        pchIdx = np.random.choice(patchIdx, size=2, replace=False)
        # assemble example
        example = np.zeros((2, 64, 64))
        example[0] = np.reshape(data[grpIdx][pchIdx[0]], (64, 64))
        example[1] = np.reshape(data[grpIdx][pchIdx[1]], (64, 64))
        # return example
        return example
    '''

    def generateNegativeExample(self):
        # generate random 3D index pair
        grpIdx = np.random.choice(self.groupIdx, size=2, replace=False).astype(str)
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
    
    '''
    Create a dictionary of every positive example
    Returns a dictionary where the keys correspond to a 3D point
    (i.e. the keys are the same as those for the self.data field,
    except that the keys are integers instead of strings).
    The item associated with each key is a list of tuples consisting
    of every unique combination of the patch indices at the 
    3D coordinate (i.e. every pair from N choose 2)
    '''
    def _getAllPositiveExampleIndices(self):
        # initialize dictionary for positive example indices
        pDict = {}
        # for every data index
        for i in self.groupIdx:
            # get the number of patches at the data index
            numPatch = data[str(i)].shape[0]
            # create a list of the patch indices
            indices = range(numPatch)
            # create a list of tuples of every N choose 2 combination 
            pairs = comb(indices, 2)
            # store the list in the dictionary
            pDict[i] = list(pairs)
        return pDict


if __name__ == '__main__':
    train_dset = Dataset(data, batch_size=5, max_dataset_size=1000)
    train_dset.next()
    example = train_dset.generatePositiveExample()
    print example.shape

