#!/usr/bin/env python
#!/usr/bin/env python 
# -*- coding: utf-8 -*-

parameter_list=[['../data/DynProg_example_py.pickle.gz']]

from modshogun import *

import numpy
from numpy import array,Inf,float64,matrix,frompyfunc,zeros

#from IPython.Shell import IPShellEmbed
#ipshell = IPShellEmbed()

import gzip
import scipy
from scipy.io import loadmat

import pickle

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

def get_ver(ver_str):
	scipy_ver=[int(i) for i in scipy.__version__.split('.')]
	v=0
	for i in range(len(scipy_ver)):
		v+=10**(len(scipy_ver)-i)*scipy_ver[i]
	return v

if get_ver(scipy.__version__) >= get_ver('0.7.0'):
	renametable = {
			'scipy.io.mio5': 'scipy.io.matlab.mio5',
			'scipy.sparse.sparse' : 'scipy.sparse',
			}
else:
	renametable = {}

def mapname(name):
	if name in renametable:
		return renametable[name]
	return name

# scipy compatibility class
class mat_struct(object):
    pass

def mapped_load_global(self):
	module = mapname(self.readline()[:-1])
	name = mapname(self.readline()[:-1])

	if name=='mat_struct':
		klass=mat_struct
	else:
		klass = self.find_class(module, name)

	self.append(klass)

def loads(str):
	file = StringIO(str)
	unpickler = pickle.Unpickler(file)
	unpickler.dispatch[pickle.GLOBAL] = mapped_load_global
	return unpickler.load()

def structure_dynprog_modular (fname):

	data_dict = loads(gzip.GzipFile(fname).read())
	#data_dict = loadmat('../data/DynProg_example_py.dat.mat', appendmat=False, struct_as_record=False)

	#print(data_dict)
	#print(len(data_dict['penalty_array'][0][0][0][0].limits[0]))
	num_plifs,num_limits = len(data_dict['penalty_array']),len(data_dict['penalty_array'][0].limits)
	pm = PlifMatrix()
	pm.create_plifs(num_plifs,num_limits)

	ids = numpy.array(list(range(num_plifs)),dtype=numpy.int32)
	min_values = numpy.array(list(range(num_plifs)),dtype=numpy.float64)
	max_values = numpy.array(list(range(num_plifs)),dtype=numpy.float64)
	all_use_cache = numpy.array(list(range(num_plifs)),dtype=numpy.bool)
	all_use_svm = numpy.array(list(range(num_plifs)),dtype=numpy.int32)
	all_limits = zeros((num_plifs,num_limits))
	all_penalties = zeros((num_plifs,num_limits))
	all_names = ['']*num_plifs
	all_transforms = ['']*num_plifs
	for plif_idx in range(num_plifs):
		ids[plif_idx]          = data_dict['penalty_array'][plif_idx].id-1
		min_values[plif_idx]   = data_dict['penalty_array'][plif_idx].min_value
		max_values[plif_idx]   = data_dict['penalty_array'][plif_idx].max_value
		all_use_cache[plif_idx]   = data_dict['penalty_array'][plif_idx].use_cache
		all_use_svm[plif_idx]   = data_dict['penalty_array'][plif_idx].use_svm
		all_limits[plif_idx]   = data_dict['penalty_array'][plif_idx].limits
		all_penalties[plif_idx]   = data_dict['penalty_array'][plif_idx].penalties
		all_names[plif_idx]   = str(data_dict['penalty_array'][plif_idx].name)
		all_transforms[plif_idx]   = str(data_dict['penalty_array'][plif_idx].transform)
		if all_transforms[plif_idx] == '[]':
			all_transforms[plif_idx] = 'linear'

	pm.set_plif_ids(ids)
	pm.set_plif_min_values(min_values)
	pm.set_plif_max_values(max_values)
	pm.set_plif_use_cache(all_use_cache)
	pm.set_plif_use_svm(all_use_svm)
	pm.set_plif_limits(all_limits)
	pm.set_plif_penalties(all_penalties)
	#pm.set_plif_names(all_names)
	#pm.set_plif_transform_type(all_transforms)

	transition_ptrs = data_dict['model'].transition_pointers
	transition_ptrs = transition_ptrs[:,:,0:2]
	transition_ptrs = transition_ptrs.astype(numpy.float64)

	pm.compute_plif_matrix(transition_ptrs)

	# init_dyn_prog
	num_svms = 8
	dyn = DynProg(num_svms)
	orf_info = data_dict['model'].orf_info
	orf_info = orf_info.astype(numpy.int32)
	num_states = orf_info.shape[0]
	dyn.set_num_states(num_states)

	block = data_dict['block']
	seq_len = len(block.seq)
	seq = str(block.seq)
	gene_string = array([elem for elem in seq])

	# precompute_content_svms
	pos = block.all_pos-1
	pos = pos.astype(numpy.int32)
	snd_pos = pos
	dyn.set_pos(pos)
	dyn.set_gene_string(gene_string)
	dyn.create_word_string()
	dyn.precompute_stop_codons()
	dyn.init_content_svm_value_array(num_svms)
	dict_weights = data_dict['content_weights']
	dict_weights = dict_weights.reshape(8,1).astype(numpy.float64)
	dict_weights = zeros((8,5440))
	dyn.set_dict_weights(dict_weights.T)

	dyn.precompute_content_values()

	dyn.init_mod_words_array(data_dict['model'].mod_words.astype(numpy.int32))
	pm.compute_signal_plifs(data_dict['state_signals'].astype(numpy.int32))

	dyn.set_orf_info(orf_info)

	#
	p = data_dict['model'].p
	q = data_dict['model'].q
	dyn.set_p_vector(p)
	dyn.set_q_vector(q)
	a_trans = data_dict['a_trans']
	a_trans = a_trans.astype(float64)

	dyn.set_a_trans_matrix(a_trans)


	dyn.check_svm_arrays()
	features = data_dict['block'].features

	dyn.set_observation_matrix(features)

	dyn.set_content_type_array(data_dict['seg_path'].astype(numpy.float64))
	dyn.best_path_set_segment_loss(data_dict['loss'].astype(numpy.float64))

	use_orf = True
	feat_dims = [25,201,2]

	dyn.set_plif_matrices(pm);

	#dyn.compute_nbest_paths(features.shape[2], use_orf, 1,True,False)

	## fetch results
	#states = dyn.get_states()
	##print(states)
	#scores = dyn.get_scores()
	##print(scores)
	#positions = dyn.get_positions()
	##print(positions)

	#return states, scores, positions

if __name__ == '__main__':
	print("Structure")
	structure_dynprog_modular(*parameter_list[0])
