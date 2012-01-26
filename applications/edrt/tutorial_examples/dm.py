import modshogun as sg
import data
import numpy as np

# load data
feature_matrix = data.swissroll()
# create features instance
features = sg.RealFeatures(feature_matrix)

# create Diffusion Maps converter instance
converter = sg.DiffusionMaps()

# set target dimensionality
converter.set_target_dim(2)
# set number of time-steps
converter.set_t(2)

# create Gaussian kernel instance
kernel = sg.GaussianKernel(100,10.0)
# enable converter instance to use created kernel instance
converter.set_kernel(kernel)

# compute embedding with Diffusion Maps method
embedding = converter.embed(features)

# compute linear kernel matrix
kernel_matrix = np.dot(feature_matrix.T,feature_matrix)
# create Custom Kernel instance
custom_kernel = sg.CustomKernel(kernel_matrix)
# construct embedding based on created kernel
kernel_embedding = converter.embed_kernel(custom_kernel)
