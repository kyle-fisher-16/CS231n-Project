import numpy as np

GaussianKernel5x5 = 1. / 115 * np.array([[2, 4, 5, 4, 2],
         [4, 9, 12, 9, 4],
         [5, 12, 15, 12, 5],
         [4, 9, 12, 9, 4],
         [2, 4, 5, 4, 2]]);
GaussianKernel5x5 = np.reshape(GaussianKernel5x5, (5, 5, 1, 1))
