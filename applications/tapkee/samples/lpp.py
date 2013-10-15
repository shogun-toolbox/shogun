import modshogun as sg
import data

# load data
feature_matrix = data.swissroll()
# create features instance
features = sg.RealFeatures(feature_matrix)

# create Locality Preserving Projections converter instance
converter = sg.LocalityPreservingProjections()

# set target dimensionality
converter.set_target_dim(2)
# set number of neighbors
converter.set_k(10)
# set number of threads
converter.parallel.set_num_threads(2)

# compute embedding with Locality Preserving Projections method
embedding = converter.embed(features)
