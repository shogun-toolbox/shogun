import modshogun as sg
import data
import numpy as np

# load data
feature_matrix = data.swissroll()
# create features instance
features = sg.RealFeatures(feature_matrix)

# create Kernel Locally Linear Embedding converter instance
converter = sg.KernelLocallyLinearEmbedding()

# set target dimensionality
converter.set_target_dim(2)
# set number of neighbors
converter.set_k(10)
# set number of threads
converter.parallel.set_num_threads(2)
# set nullspace shift (optional)
converter.set_nullspace_shift(-1e-6)

# create Gaussian kernel instance
kernel = sg.GaussianKernel(100,10.0)
# enable converter instance to use created kernel instance
converter.set_kernel(kernel)

# compute embedding with Kernel Locally Linear Embedding method
embedding = converter.embed(features)

# compute linear kernel matrix
kernel_matrix = np.dot(feature_matrix.T,feature_matrix)
# create Custom Kernel instance
custom_kernel = sg.CustomKernel(kernel_matrix)
# construct embedding based on created kernel
kernel_embedding = converter.embed_kernel(custom_kernel)
