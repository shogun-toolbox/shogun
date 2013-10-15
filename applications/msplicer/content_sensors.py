#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2006-2007 Soeren Sonnenburg
# Written (W) 2007 Gunnar Raetsch
# Copyright (C) 2007-2008 Fraunhofer Institute FIRST and Max-Planck-Society
#

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
