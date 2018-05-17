
import cv2
import numpy as np
import h5py
from dataset import Dataset
import csv

sift = cv2.xfeatures2d.SIFT_create()

# ====== LOAD DATA ======
data = h5py.File('data/liberty.h5', 'r')
training_dset = Dataset(data, batch_size=2);

def get_des(patch):
  test_kp = cv2.KeyPoint(31,31,64,_angle=0)
  # try to make a descriptor
  test_des = sift.compute(patch, [test_kp])
  return test_des[1].reshape((-1))

def get_des_dist(left, right):
  left_des = get_des(left)
  right_des = get_des(right)
  dist = np.linalg.norm(left_des-right_des)
  return dist


num_batches = 5000;
dist_data = np.zeros((num_batches, 2));
batch_num = 0;
# should be 1 matching pair only
for X_batch, y_batch in training_dset:
  for n in range(len(X_batch)):
    left_im, right_im = X_batch[n,0,:,:], X_batch[n,1,:,:]
#    cv2.imshow('img', np.hstack((left_im,right_im)))
#    cv2.waitKey(0)
    # make a test keypoint
    dist = get_des_dist(left_im, right_im)
    dist_data[batch_num, n] = dist;
  
  batch_num += 1
  if batch_num >= num_batches:
    break;

with open('dist_data.csv', 'wb') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerows(dist_data)
