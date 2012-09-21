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
# set number of threads
converter.parallel.set_num_threads(2)
# set reconstruction shift (optional)
converter.set_reconstruction_shift(1e-3)
# set nullspace shift (optional)
converter.set_nullspace_shift(-1e-6)
# check whether arpack is used
if converter.get_use_arpack():
	print 'ARPACK is used'
else:
	print 'LAPACK is used'

# compute embedding with Locally Linear Embedding method
embedding_first = converter.embed(features)

# enable auto k search in range of (10,100)
# based on reconstruction error
converter.set_k(50)
converter.set_max_k(100)
converter.set_auto_k(True)

# compute embedding with Locally Linear Embedding method
embedding_second = converter.embed(features)
