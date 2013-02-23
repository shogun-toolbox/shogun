import modshogun as sg
import data
import numpy as np

# load data
feature_matrix = data.swissroll()
# create features instance
features = sg.RealFeatures(feature_matrix)

# create Multidimensional Scaling converter instance
converter = sg.MultidimensionalScaling()

# set target dimensionality
converter.set_target_dim(2)

# compute embedding with Multidimensional Scaling method
embedding = converter.embed(features)

# enable landmark approximation
converter.set_landmark(True)
# set number of landmarks
converter.set_landmark_number(100)
# set number of threads
converter.parallel.set_num_threads(2)
# compute approximate embedding
approx_embedding = converter.embed(features)
# disable landmark approximation
converter.set_landmark(False)

# compute cosine distance matrix 'manually'
N = features.get_num_vectors()
distance_matrix = np.zeros((N,N))
for i in range(N):
	for j in range(N):
		distance_matrix[i,j] = \
		  np.linalg.norm(feature_matrix[:,i]-feature_matrix[:,j],2)
# create custom distance instance
distance = sg.CustomDistance(distance_matrix)
# construct embedding based on created distance
converter.embed_distance(distance)
