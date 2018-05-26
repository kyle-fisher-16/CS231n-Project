
import h5py
import numpy as np
from itertools import combinations as comb
from copy import deepcopy
import pdb

data = h5py.File('data/liberty.h5', 'r')

class Dataset(object):

    def __init__(self, data, batch_size, pct_for_val=10.0, max_dataset_size=100000000):
        self.data = data
        self.batch_size = batch_size
        
        self.nGroups = int(self.data.keys()[-1])
        print 'Loaded dataset, complete size =', self.nGroups, 'keypoints.'
        self.nGroups = np.minimum(self.nGroups, max_dataset_size)
        print 'Limited dataset size to', self.nGroups, 'keypoints.'
        
        self.groupIdx = xrange(self.nGroups + 1)
        self._pDict = self._getAllPositiveExampleIndices()
        self._keypoint = 0
        self._combIdx = 0
        self._maxKeypoints = self.nGroups + 1
    
        self.val_dataset = self.get_val_data(pct_for_val)

    def __iter__(self):
        return self

    def next(self):
        # initialize X and y
        X = np.zeros((self.batch_size, 4), dtype = "int32") # indices
        y = np.zeros(self.batch_size,)
        pct_complete = 0.0;
        # for i in batch size
        for i in xrange(self.batch_size):
            # if i is even
            if i % 2 == 0:
                # generate a positive example
                example_idx, pct_complete = self.generatePositiveExample()
            # if i is odd
            else:
                # generate a negative example
                example_idx = self.generateNegativeExample()

            # store example
            X[i] = example_idx;
            # store label
            y[i] = int(i % 2 == 0)
        # return examples and labels
        return X, y, pct_complete

    def get_val_data(self, pct_for_val):
        print 'Generating validation dataset...'
        # seed rng so that always generates same negative examples for validation
        np.random.seed(231)
        pct_total_data = 0
        X_valset = [] # indices, in list form for each batch
        y_valset = []
        while (pct_total_data < pct_for_val):
            X, y, pct_total_data = self.next()
            X_valset.append(X)
            y_valset.append(y)
        # reset rng so that all negative examples will be random here after
        np.random.seed()
        return (X_valset, y_valset)
    
    def generatePositiveExample(self):
        # get the current positive example
        exIdx = self._pDict[self._keypoint][self._combIdx]
        
        # write out the keypoint # and the two patch indices
        outIdx = np.array([self._keypoint, exIdx[0], self._keypoint, exIdx[1]], dtype="int32")
        
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
        # percentage of the data we've visited
        pct = 100.0 * float(self._keypoint) / self._maxKeypoints;
        
        return outIdx, pct
    
    # fetch the image data (given the indices stored in batch_idx)
    # indices are an Nx4 array.
    def fetchImageData(self, batch_idx):
        # TODO: optimize for performance
        out_imgs = np.zeros((len(batch_idx), 2, 64, 64), dtype="uint8")
        ct = 0;
        for idx in batch_idx:
            # left patch
            left_patch_group = self.data[str(idx[0])];
            left_patch = left_patch_group[idx[1], :].reshape(64, 64);
            
            # right patch
            right_patch_group = self.data[str(idx[2])];
            right_patch = right_patch_group[idx[3], :].reshape(64, 64);
            
            if (idx[0] == idx[2] and idx[1] == idx[3]):
                print
                print 'WARNING: DUPLICATE INDICES FOUND'
                print
            
            # check for duplicates
            if (np.all(left_patch == right_patch)):
                print
                print 'WARNING: DUPLICATE IMAGE FOUND'
                print 'Indices:', idx[0], idx[2], idx[1], idx[3]
                print

            # assign this example into the batch
            out_imgs[ct, 0, :, :] = left_patch;
            out_imgs[ct, 1, :, :] = right_patch;
            
            ct += 1;
        
        return out_imgs[0:ct] # TODO: fix the skipped y-idc

    # generate indices for a negative example
    def generateNegativeExample(self):
        # generate random 3D index pair
        grpIdx = np.random.choice(self.groupIdx, size=2, replace=False).astype(str)
        # generate random patch index pair
        pchSize = (self.data[grpIdx[0]].shape[0],
                   self.data[grpIdx[1]].shape[0])
        pchIdx = (np.random.randint(pchSize[0]),
                  np.random.randint(pchSize[1]));
        outIdx = np.array([grpIdx[0], pchIdx[0], grpIdx[1], pchIdx[1]], dtype="int32")
#        print outIdx
        return outIdx
    
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
        print 'Generating all example indices...'
        pDict = {}
        to_do = float(len(self.groupIdx));
        # for every data index
        pct_done_prev = -5.0;
        for i in self.groupIdx:
            pct_done = 100.0*float(i)/to_do;
            if (pct_done - pct_done_prev) >= 5.0:
                print str(np.around(pct_done, 0)) + '% generated'
                pct_done_prev = pct_done;
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
    train_dset = Dataset(data, batch_size=5, max_dataset_size=5000)
    X_batch, y_batch = train_dset.next()
    batch_imgs = train_dset.fetchImageData(X_batch)
#    print batch_imgs
#    example = train_dset.generatePositiveExample()
#    print example




# EXTRA CODE HERE

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
