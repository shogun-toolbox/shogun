#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2006-2008 Soeren Sonnenburg
# Written (W) 2007 Gunnar Raetsch
# Copyright (C) 2006-2008 Fraunhofer Institute FIRST and Max-Planck-Society
#

import sys
import numpy
import seqdict

from modshogun import KernelMachine,StringCharFeatures,DNA,WeightedDegreeStringKernel

class svm_splice_model(object):
	def __init__(self, order, traindat, alphas, b, (window_left,offset,window_right), consensus):

		f=StringCharFeatures(DNA)
		f.set_features(traindat)
		wd_kernel = WeightedDegreeStringKernel(f,f, int(order))
		wd_kernel.io.set_target_to_stderr()

		self.svm=KernelMachine(wd_kernel, alphas, numpy.arange(len(alphas), dtype=numpy.int32), b)
		self.svm.io.set_target_to_stderr()
		self.svm.parallel.set_num_threads(self.svm.parallel.get_num_cpus())
		self.svm.set_linadd_enabled(False)
		self.svm.set_batch_computation_enabled(False)

		self.window_left=int(window_left)
		self.window_right=int(window_right)

		self.consensus=consensus
		self.wd_kernel=wd_kernel
		self.traindat=f
		self.offset=offset

	def get_positions(self, sequence):
		positions=list()

		for cons in self.consensus:
			l=sequence.find(cons)
			while l>-1:
				if l<len(sequence)-self.window_right-2 and l>self.window_left:
					positions.append(l+self.offset)
				l=sequence.find(cons, l+1)

		positions.sort()
		return positions

	def get_predictions_from_seqdict(self, seqdic, site):
		""" we need to generate a huge test features object
			containing all locations found in each seqdict-sequence
			and each location (this is necessary to efficiently
			(==fast,low memory) compute the splice outputs
		"""

		seqlen=self.window_right+self.window_left+2

		num=0
		for s in seqdic:
			num+= len(s.preds[site].positions)

		testdat = []

		for s in seqdic:
			sequence=s.seq
			positions=s.preds[site].positions
			for j in xrange(len(positions)):
				i=positions[j] - self.offset
				s=sequence[i-self.window_left:i+self.window_right+2]
				testdat.append(s)

		t=StringCharFeatures(testdat, DNA)

		self.wd_kernel.init(self.traindat, t)
		self.svm.set_kernel(self.wd_kernel)
		l=self.svm.apply().get_labels()
		sys.stderr.write("\n...done...\n")

		k=0
		for s in seqdic:
			num=len(s.preds[site].positions)
			scores= num * [0]
			for j in xrange(num):
				scores[j]=l[k]
				k+=1
			s.preds[site].set_scores(scores)

	def get_positions_from_seqdict(self, seqdic, site):
		for d in seqdic:
			positions=list()
			sequence=d.seq
			for cons in self.consensus:
				l=sequence.find(cons)
				while l>-1:
					if l<len(sequence)-self.window_right-2 and l>self.window_left:
						positions.append(l+self.offset)
					l=sequence.find(cons, l+1)
			positions.sort()
			d.preds[site].set_positions(positions)

	def get_predictions(self, sequence, positions):

		seqlen=self.window_right+self.window_left+2
		num=len(positions)

		testdat = []

		for j in xrange(num):
			i=positions[j] - self.offset ;
			s=sequence[i-self.window_left:i+self.window_right+2]
			testdat.append(s)

		t=StringCharFeatures(DNA)
		t.set_string_features(testdat)

		self.wd_kernel.init(self.traindat, t)
		l=self.svm.classify().get_labels()
		sys.stderr.write("\n...done...\n")
		return l

class signal_detectors(object):
	def __init__(self, model):
		if model.don_splice_use_gc:
			don_consensus=['GC','GT']
		else:
			don_consensus=['GT']

		self.acceptor=svm_splice_model(model.acc_splice_order, model.acc_splice_svs,
				numpy.array(model.acc_splice_alphas).flatten(), model.acc_splice_b,
				(model.acc_splice_window_left, 2, model.acc_splice_window_right), ['AG'])
		self.donor=svm_splice_model(model.don_splice_order, model.don_splice_svs,
				numpy.array(model.don_splice_alphas).flatten(), model.don_splice_b,
				(model.don_splice_window_left, 0, model.don_splice_window_right),
				don_consensus)

	def set_sequence(self, seq):
		self.acceptor.set_sequence(seq)
		self.donor.set_sequence(seq)

	def predict_acceptor_sites(self, seq):
		pos=self.acceptor.get_positions(seq)
		sys.stderr.write("computing svm output for acceptor positions\n")
		pred=self.acceptor.get_predictions(seq, pos)
		return (pos,pred)

	def predict_donor_sites(self,seq):
		pos=self.donor.get_positions(seq)
		sys.stderr.write("computing svm output for donor positions\n")
		pred=self.donor.get_predictions(seq, pos)
		return (pos,pred)

	def predict_acceptor_sites_from_seqdict(self, seqs):
		self.acceptor.get_positions_from_seqdict(seqs, 'acceptor')
		sys.stderr.write("computing svm output for acceptor positions\n")
		self.acceptor.get_predictions_from_seqdict(seqs, 'acceptor')

	def predict_donor_sites_from_seqdict(self, seqs):
		self.donor.get_positions_from_seqdict(seqs, 'donor')
		sys.stderr.write("computing svm output for donor positions\n")
		self.donor.get_predictions_from_seqdict(seqs, 'donor')
