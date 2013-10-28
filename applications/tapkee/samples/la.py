import modshogun as sg
import data
import numpy as np

# load data
feature_matrix = data.swissroll()
# create features instance
features = sg.RealFeatures(feature_matrix)

# create Laplacian Eigenmaps converter instance
converter = sg.LaplacianEigenmaps()

# set target dimensionality
converter.set_target_dim(2)
# set number of neighbors
converter.set_k(20)
# set tau multiplier
converter.set_tau(1.0)

# compute embedding with Laplacian Eigenmaps method
embedding = converter.embed(features)

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
