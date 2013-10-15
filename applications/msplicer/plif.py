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

from numpy import array
from shogun.Structure import Plif
from shogun.Structure import PlifArray
from shogun.Library import DynamicPlifArray

class plif:
	def __init__(self, model):
		min_exon_len=2
		min_intron_len=30
		max_len=22222;
		#extract plifs from model
		l=array(model.penalty_acceptor_boundaries).flatten()
		p=array(model.penalty_acceptor_penalty).flatten()
		self.acceptor=Plif(len(l))
		self.acceptor.set_plif_limits(l)
		self.acceptor.set_plif_penalty(p)
		self.acceptor.set_min_value(-1e+20)
		self.acceptor.set_max_value(1e+20)
		self.acceptor.set_plif_name("acceptor")

		l=array(model.penalty_donor_boundaries).flatten()
		p=array(model.penalty_donor_penalty).flatten()
		self.donor=Plif(len(l))
		self.donor.set_plif_limits(l)
		self.donor.set_plif_penalty(p)
		self.donor.set_min_value(-1e+20)
		self.donor.set_max_value(1e+20)
		self.donor.set_plif_name("donor")

		l=array(model.penalty_coding_len_boundaries).flatten()
		p=array(model.penalty_coding_len_penalty).flatten()
		self.coding_len=Plif(len(l))
		self.coding_len.set_plif_limits(l)
		self.coding_len.set_plif_penalty(p)
		self.coding_len.set_min_value(min_exon_len)
		self.coding_len.set_max_value(max_len)
		self.coding_len.set_plif_name('coding_len')
		self.coding_len.set_transform_type("log(+1)")

		l=array(model.penalty_first_coding_len_boundaries).flatten()
		p=array(model.penalty_first_coding_len_penalty).flatten()
		self.first_coding_len=Plif(len(l))
		self.first_coding_len.set_plif_limits(l)
		self.first_coding_len.set_plif_penalty(p)
		self.first_coding_len.set_min_value(min_exon_len)
		self.first_coding_len.set_max_value(max_len)
		self.first_coding_len.set_plif_name("first_coding_len")
		self.first_coding_len.set_transform_type("log(+1)")

		l=array(model.penalty_last_coding_len_boundaries).flatten()
		p=array(model.penalty_last_coding_len_penalty).flatten()
		self.last_coding_len=Plif(len(l))
		self.last_coding_len.set_plif_limits(l)
		self.last_coding_len.set_plif_penalty(p)
		self.last_coding_len.set_min_value(min_exon_len)
		self.last_coding_len.set_max_value(max_len)
		self.last_coding_len.set_plif_name('last_coding_len')
		self.last_coding_len.set_transform_type("log(+1)")

		l=array(model.penalty_single_coding_len_boundaries).flatten()
		p=array(model.penalty_single_coding_len_penalty).flatten()
		self.single_coding_len=Plif(len(l))
		self.single_coding_len.set_plif_limits(l)
		self.single_coding_len.set_plif_penalty(p)
		self.single_coding_len.set_min_value(min_exon_len)
		self.single_coding_len.set_max_value(max_len)
		self.single_coding_len.set_plif_name('single_coding_len')
		self.single_coding_len.set_transform_type("log(+1)")

		l=array(model.penalty_intron_len_boundaries).flatten()
		p=array(model.penalty_intron_len_penalty).flatten()
		self.intron_len=Plif(len(l))
		self.intron_len.set_plif_limits(l)
		self.intron_len.set_plif_penalty(p)
		self.intron_len.set_min_value(min_intron_len)
		self.intron_len.set_max_value(max_len)
		self.intron_len.set_plif_name('intron_len')
		self.intron_len.set_transform_type("log(+1)")

		l=array(model.penalty_coding_boundaries).flatten()
		p=array(model.penalty_coding_penalty).flatten()
		self.coding=Plif(len(l))
		self.coding.set_use_svm(1)
		self.coding.set_plif_limits(l)
		self.coding.set_plif_penalty(p)
		self.coding.set_min_value(-1e+20)
		self.coding.set_max_value(1e+20)
		self.coding.set_plif_name('coding')

		l=array(model.penalty_coding2_boundaries).flatten()
		p=array(model.penalty_coding2_penalty).flatten()
		self.coding2=Plif(len(l))
		self.coding2.set_use_svm(3)
		self.coding2.set_plif_limits(l)
		self.coding2.set_plif_penalty(p)
		self.coding2.set_min_value(-1e+20)
		self.coding2.set_max_value(1e+20)
		self.coding2.set_plif_name('coding2')

		l=array(model.penalty_coding3_boundaries).flatten()
		p=array(model.penalty_coding3_penalty).flatten()
		self.coding3=Plif(len(l))
		self.coding3.set_use_svm(5)
		self.coding3.set_plif_limits(l)
		self.coding3.set_plif_penalty(p)
		self.coding3.set_min_value(-1e+20)
		self.coding3.set_max_value(1e+20)
		self.coding3.set_plif_name('coding3')

		l=array(model.penalty_coding4_boundaries).flatten()
		p=array(model.penalty_coding4_penalty).flatten()
		self.coding4=Plif(len(l))
		self.coding4.set_use_svm(7)
		self.coding4.set_plif_limits(l)
		self.coding4.set_plif_penalty(p)
		self.coding4.set_min_value(-1e+20)
		self.coding4.set_max_value(1e+20)
		self.coding4.set_plif_name('coding4')

		l=array(model.penalty_intron_boundaries).flatten()
		p=array(model.penalty_intron_penalty).flatten()
		self.intron=Plif(len(l))
		self.intron.set_use_svm(2)
		self.intron.set_plif_limits(l)
		self.intron.set_plif_penalty(p)
		self.intron.set_min_value(-1e+20)
		self.intron.set_max_value(1e+20)
		self.intron.set_plif_name('intron')

		l=array(model.penalty_intron2_boundaries).flatten()
		p=array(model.penalty_intron2_penalty).flatten()
		self.intron2=Plif(len(l))
		self.intron2.set_use_svm(4)
		self.intron2.set_plif_limits(l)
		self.intron2.set_plif_penalty(p)
		self.intron2.set_min_value(-1e+20)
		self.intron2.set_max_value(1e+20)
		self.intron2.set_plif_name('intron2')

		l=array(model.penalty_intron3_boundaries).flatten()
		p=array(model.penalty_intron3_penalty).flatten()
		self.intron3=Plif(len(l))
		self.intron3.set_use_svm(6)
		self.intron3.set_plif_limits(l)
		self.intron3.set_plif_penalty(p)
		self.intron3.set_min_value(-1e+20)
		self.intron3.set_max_value(1e+20)
		self.intron3.set_plif_name('intron3')

		l=array(model.penalty_intron4_boundaries).flatten()
		p=array(model.penalty_intron4_penalty).flatten()
		self.intron4=Plif(len(l))
		self.intron4.set_use_svm(8)
		self.intron4.set_plif_limits(l)
		self.intron4.set_plif_penalty(p)
		self.intron4.set_min_value(-1e+20)
		self.intron4.set_max_value(1e+20)
		self.intron4.set_plif_name('intron4')

		p=array(model.penalty_transitions_penalty).flatten()
		self.transitions=Plif(len(p))
		self.transitions.set_plif_penalty(p)
		self.transitions.set_min_value(-1e+20)
		self.transitions.set_max_value(1e+20)

		#create magic plifarrays
		self.first_coding_plif_array=PlifArray()
		self.first_coding_plif_array.add_plif(self.first_coding_len)
		self.first_coding_plif_array.add_plif(self.coding)
		self.first_coding_plif_array.add_plif(self.coding2)
		self.first_coding_plif_array.add_plif(self.coding3)
		self.first_coding_plif_array.add_plif(self.coding4)

		self.last_coding_plif_array=PlifArray()
		self.last_coding_plif_array.add_plif(self.last_coding_len)
		self.last_coding_plif_array.add_plif(self.coding)
		self.last_coding_plif_array.add_plif(self.coding2)
		self.last_coding_plif_array.add_plif(self.coding3)
		self.last_coding_plif_array.add_plif(self.coding4)

		self.coding_plif_array=PlifArray()
		self.coding_plif_array.add_plif(self.coding_len)
		self.coding_plif_array.add_plif(self.coding)
		self.coding_plif_array.add_plif(self.coding2)
		self.coding_plif_array.add_plif(self.coding3)
		self.coding_plif_array.add_plif(self.coding4)

		self.single_coding_plif_array=PlifArray()
		self.single_coding_plif_array.add_plif(self.single_coding_len)
		self.single_coding_plif_array.add_plif(self.coding)
		self.single_coding_plif_array.add_plif(self.coding2)
		self.single_coding_plif_array.add_plif(self.coding3)
		self.single_coding_plif_array.add_plif(self.coding4)

		self.intron_plif_array=PlifArray()
		self.intron_plif_array.add_plif(self.intron_len)
		self.intron_plif_array.add_plif(self.intron)
		self.intron_plif_array.add_plif(self.intron2)
		self.intron_plif_array.add_plif(self.intron3)
		self.intron_plif_array.add_plif(self.intron4)

		#finally create a single array with all the plifs
		self.plif_array=DynamicPlifArray()
		self.plif_array.append_element(self.acceptor)
		self.plif_array.append_element(self.donor)
		self.plif_array.append_element(self.first_coding_plif_array)
		self.plif_array.append_element(self.last_coding_plif_array)
		self.plif_array.append_element(self.coding_plif_array)
		self.plif_array.append_element(self.single_coding_plif_array)
		self.plif_array.append_element(self.intron_plif_array)

	def get_plif_array(self):
		return self.plif_array
