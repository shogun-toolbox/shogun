#!/usr/bin/env python

from numpy import double, fromfile, loadtxt

class LoadMatrix:
	def __init__(self):
		pass


	def load_numbers(self, filename):
		return loadtxt(filename).T

	def load_dna(self, filename):
		fh=open(filename, 'r');
		matrix=[]

		for line in fh:
			matrix.append(line[:-1])

		return matrix


	def load_cubes(self, filename):
		fh=open(filename, 'r');
		matrix=[]

		for line in fh:
			matrix.append(line.split(' ')[0][:-1])

		fh.close()

		return matrix


	def load_labels(self, filename):
		return fromfile(filename, dtype=double, sep=' ')
