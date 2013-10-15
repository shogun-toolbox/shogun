import modshogun as sg
import data

# load data
feature_matrix = data.swissroll()
# create features instance
features = sg.RealFeatures(feature_matrix)

# create Locally Linear Embedding converter instance
converter = sg.LocallyLinearEmbedding()

# set target dimensionality
converter.set_target_dim(2)
# set number of neighbors
converter.set_k(10)
# set reconstruction shift (optional)
converter.set_reconstruction_shift(1e-3)
# set nullspace shift (optional)
converter.set_nullspace_shift(-1e-6)

# compute embedding with Locally Linear Embedding method
embedding_first = converter.embed(features)

# set number of neighbors to be used
converter.set_k(50)

# compute embedding with Locally Linear Embedding method
embedding_second = converter.embed(features)
