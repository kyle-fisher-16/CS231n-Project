import numpy as np
data = np.genfromtxt('dist_data.csv', delimiter=',')


range_lim = 500;
num_bins = 40;

for i in range(2):
  hist_out = np.histogram(data[:,i], range=(0, range_lim), bins=num_bins)
  vals = hist_out[0].reshape((-1, 1))
  ranges = hist_out[1][:-1].reshape((-1, 1)) + range_lim/num_bins/2.0;
  out_arr = np.hstack((ranges, vals));
  np.savetxt("hist_bins_" + str(i) + ".csv", out_arr, delimiter=",")

