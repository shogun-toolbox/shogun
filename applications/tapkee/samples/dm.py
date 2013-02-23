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
# set width of gaussian kernel
converter.set_width(10.0)

# create euclidean distance instance
distance = sg.EuclideanDistance()
# enable converter instance to use created distance instance
converter.set_distance(distance)

# compute embedding with Diffusion Maps method
embedding = converter.embed(features)

# compute custom distance matrix
distance_matrix = np.exp(-np.dot(feature_matrix.T,feature_matrix))
# create Custom Kernel instance
custom_distance = sg.CustomDistance(distance_matrix)
# construct embedding based on created distance
distance_embedding = converter.embed_distance(custom_distance)
