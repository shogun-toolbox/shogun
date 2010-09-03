# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# Written (W) 2008 Soeren Sonnenburg
# Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society

import numpy
import sys

from util import *

from shogun.Features import StringCharFeatures, StringWordFeatures, CombinedFeatures, DNA
from shogun.Kernel import CombinedKernel, WeightedDegreePositionStringKernel
from shogun.Kernel import K_COMMWORDSTRING, CommWordStringKernel, IdentityKernelNormalizer
from shogun.PreProc import SortWordString
from shogun.Classifier import SVM

class sensor(object):
	"""
	sensor has window (left,center,right) of length right-left+1
	with center at "center"
	"""

	def __init__(self, window=None, kernel=None, train_features=None):
		self.kernel=kernel
		self.window=window
		self.train_features=train_features
		self.preproc=None

	def from_file(self, file, num):
		"""
		parse lines with num as suffix, e.g.

		kernel_<arg><num>=<value>
		"""
		l=file.readline()

		name=None
		left=None
		right=None
		center=None
		order=None
		shift=None
		svs=None

		while l:
			if l.find('%d=' % num)>-1:
				if name is None: name=parse_name(l, 'kernel_name%d' % num)
				if left is None: left=parse_int(l, 'kernel_left%d' % num)
				if right is None: right=parse_int(l, 'kernel_right%d' % num)
				if center is None: center=parse_int(l, 'kernel_center%d' % num)
				if order is None: order=parse_int(l, 'kernel_order%d' % num)
				if shift is None: shift=parse_int(l, 'kernel_shift%d' % num)
				if svs is None: svs=parse_string(l, file, 'kernel_svs%d' % num)
			else:
				self.window=(left, center, right)
				return self.init_sensor({ 'name' : name, 'order': order, 'shift' : shift}, svs)

			l=file.readline()

	def init_sensor(self, kernel, svs):
		f=StringCharFeatures(DNA)
		f.set_string_features(svs)

		kname=kernel['name']
		if  kname == 'spectrum':
			wf=StringWordFeatures(f.get_alphabet())
			wf.obtain_from_char(f, kernel['order']-1, kernel['order'], 0, False)

			pre = SortWordString()
			pre.init(wf)
			wf.add_preproc(pre)
			wf.apply_preproc()
			f=wf

			k=CommWordStringKernel(0, False)
			k.set_use_dict_diagonal_optimization(kernel['order']<8)
			self.preproc=pre

		elif kname == 'wdshift':
				k = WeightedDegreePositionStringKernel(0, kernel['order'])
				k.set_normalizer(IdentityKernelNormalizer())
				k.set_shifts( kernel['shift'] *
						numpy.ones(f.get_max_vector_length(), dtype=numpy.int32) )
				k.set_position_weights( 1.0/f.get_max_vector_length() *
						numpy.ones(f.get_max_vector_length(), dtype=numpy.float64) )
		else:
			raise "Currently, only wdshift and spectrum kernels supported"

		self.kernel=k
		self.train_features=f

		return (self.kernel, self.train_features)

	def get_test_features(self, seq, window):
		start=self.window[0]-window[0]
		end=len(seq)-window[1]+self.window[2]
		size=self.window[2]-self.window[0]+1
		seq=seq[start:end]
		f=StringCharFeatures(DNA)
		f.set_string_features([seq])

		if self.preproc:
			wf=StringWordFeatures(f.get_alphabet())
			o=self.train_features.get_order()
			wf.obtain_from_char(f, 0, o, 0, False)
			f=wf
			f.obtain_by_sliding_window(size, 1, o-1)
		else:
			f.obtain_by_sliding_window(size, 1)

		return f

class signal_sensor(object):
	"""
	A collection of sensors
	"""
	def __init__(self):
		self.sensors=list()
		self.kernel=CombinedKernel()
		self.svs=CombinedFeatures()
		self.svm=None
		self.window=(+100000, -1000000)

	def from_file(self, file):
		sys.stderr.write('loading model file')
		l=file.readline();

		if l != '%arts version: 1.0\n':
			sys.stderr.write("\nfile not an arts definition file\n")
			return None

		bias=None
		alphas=None
		num_kernels=None

		while l:
			# skip comment or empty line
			if not ( l.startswith('%') or l.startswith('\n') ): 
				if bias is None: bias=parse_float(l, 'b')
				if alphas is None: alphas=parse_vector(l, file, 'alphas')
				if num_kernels is None: num_kernels=parse_int(l, 'num_kernels')

				if num_kernels and bias and alphas is not None:
					for i in xrange(num_kernels):
						s=sensor()
						(k,f)=s.from_file(file, i+1)
						k.io.enable_progress()
						self.window = (min(self.window[0], s.window[0]),
								max(self.window[1], s.window[2]))
						self.sensors.append(s)
						self.kernel.append_kernel(k)
						self.svs.append_feature_obj(f)

					self.kernel.init(self.svs, self.svs)
					self.svm = SVM(self.kernel, alphas, 
							numpy.arange(len(alphas),dtype=numpy.int32), bias)
					self.svm.io.set_target_to_stderr()
					self.svm.io.enable_progress()
					self.svm.parallel.set_num_threads(self.svm.parallel.get_num_cpus())
					sys.stderr.write('done\n')
					return

			l=file.readline()

		sys.stderr.write('error loading model file\n')
	
	def predict(self, seq):
		tf=CombinedFeatures()
		for i in xrange(len(self.sensors)):
			f=self.sensors[i].get_test_features(seq, self.window)
			tf.append_feature_obj(f)

		sys.stderr.write("initialising kernel...")
		self.kernel.init(self.svs, tf)
		sys.stderr.write("..done\n")
		l=(-self.window[0])*[-42]
		r=self.window[1]*[-42]
		return numpy.concatenate((l, self.svm.classify().get_labels(), r))
