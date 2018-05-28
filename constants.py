import numpy as np

GaussianKernel5x5 = 1. / 256 * np.array([[1, 4, 6, 4, 1],
                                         [4, 16, 24, 16, 4],
                                         [6, 24, 36, 24, 6],
                                         [4, 16, 24, 16, 4],
                                         [1, 4, 6, 4, 1]], dtype="float32");
GaussianKernel5x5 = np.reshape(GaussianKernel5x5, (5, 5, 1, 1))
