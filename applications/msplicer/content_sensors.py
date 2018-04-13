# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Soeren Sonnenburg, Gunnar Raetsch

import numpy

class content_sensors:
	def __init__(self, model):
		self.dict_weights_intron=numpy.array(model.dict_weights_intron, dtype=numpy.float64)
		self.dict_weights_coding=numpy.array(model.dict_weights_coding, dtype=numpy.float64)

		self.dicts=numpy.concatenate((self.dict_weights_coding,self.dict_weights_intron, self.dict_weights_coding, self.dict_weights_intron, self.dict_weights_coding,self.dict_weights_intron, self.dict_weights_coding, self.dict_weights_intron), axis=0)

		self.dicts[0, 64:] = 0      # only order 3 info
		self.dicts[1, 64:] = 0      # only order 3 info
		self.dicts[2, 0:64] = 0     # only order 4 info
		self.dicts[2, 320:] = 0
		self.dicts[3, 0:64] = 0     # only order 4 info
		self.dicts[3, 320:] = 0
		self.dicts[4, 0:320] = 0    # only order 5 info
		self.dicts[4, 1344:] = 0
		self.dicts[5, 0:320] = 0    # only order 5 info
		self.dicts[5, 1344:] = 0
		self.dicts[6, 0:1344] = 0   # only order 6 info
		self.dicts[7, 0:1344] = 0   # only order 6 info

		self.model = model

	def get_dict_weights(self):
		return self.dicts.T

	def initialize_content(self, dyn):
		dyn.init_svm_arrays(len(self.model.word_degree), len(self.model.mod_words))

		word_degree = numpy.array(self.model.word_degree, numpy.int32)
		dyn.init_word_degree_array(word_degree)

		mod_words = numpy.array(4**word_degree, numpy.int32)
		dyn.init_num_words_array(mod_words)

		cum_mod_words=numpy.zeros(len(mod_words)+1, numpy.int32)
		cum_mod_words[1:] = numpy.cumsum(mod_words)
		dyn.init_cum_num_words_array(cum_mod_words)

		dyn.init_mod_words_array(numpy.array(self.model.mod_words, numpy.int32))
		dyn.init_sign_words_array(numpy.array(self.model.sign_words, numpy.bool))
		dyn.init_string_words_array(numpy.zeros(len(self.model.sign_words), numpy.int32))

		assert(dyn.check_svm_arrays())
