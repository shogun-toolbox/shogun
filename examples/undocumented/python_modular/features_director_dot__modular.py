import numpy
try:
	from shogun.Features import DirectorDotFeatures
except ImportError:
	print "recompile shogun with --enable-swig-directors"
	import sys
	sys.exit(0)

data=None

class NumpyFeatures(DirectorDotFeatures):
	def __init__(self, d):
		global data
		data = d
		DirectorDotFeatures.__init__(self)
	
	def add_to_dense_sgvec(self, alpha, vec_idx1, vec2, abs):
		vec2+=alpha*numpy.abs(data[vec_idx1])

	def get_num_vectors(self):
		return data.shape[0]
	
	def get_dim_feature_space(self):
		return data.shape[1]

traindat = numpy.random.random_sample((10,10))
parameter_list=[[traindat]]

def features_director_dot_modular (fm_train_real=traindat):
	feats_train=NumpyFeatures(fm_train_real)
	feats_train.io.enable_file_and_line()
	data=feats_train.get_computed_dot_feature_matrix()
	print data
	print fm_train_real
	 
	return data

if __name__=='__main__':
	print('DirectorLinear')
	features_director_dot_modular(*parameter_list[0])
