import modshogun as sg
import data
import numpy as np

# load data
feature_matrix = data.swissroll()
# create features instance
features = sg.RealFeatures(feature_matrix)

# create Isomap converter instance
converter = sg.Isomap()

# set number of neighbors to be used
converter.set_k(15)

# set target dimensionality
converter.set_target_dim(2)

# compute embedding with Isomap method
embedding = converter.embed(features)

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
