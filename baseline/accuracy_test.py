import numpy as np
data = np.genfromtxt('dist_data.csv', delimiter=',')


range_lim = 500;
num_bins = 40;

max_acc = 0
thresh_dist = 0
for i in range(500):
  # i is the thresh
  print i
  num_correct_match = np.sum(data[:,0]<i)
  num_correct_notmatch = np.sum(data[:,1]>=i)
  accuracy = (num_correct_match + num_correct_notmatch)/(len(data)*2.0)
  if accuracy > max_acc:
    max_acc = accuracy
    thresh_dist = i

print max_acc, thresh_dist

