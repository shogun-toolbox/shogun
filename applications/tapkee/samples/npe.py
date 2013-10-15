import modshogun as sg
import data

# load data
feature_matrix = data.swissroll()
# create features instance
features = sg.RealFeatures(feature_matrix)

# create Neighborhood Preserving Embedding converter instance
converter = sg.NeighborhoodPreservingEmbedding()

# set target dimensionality
converter.set_target_dim(2)
# set number of neighbors
converter.set_k(10)
# set number of threads
converter.parallel.set_num_threads(2)
# set nullspace shift (optional)
converter.set_nullspace_shift(-1e-6)

# compute embedding with Neighborhood Preserving Projections method
embedding = converter.embed(features)
