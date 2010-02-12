#!/usr/bin/env python

from numpy import double, fromfile

class LoadMatrix:
	def __init__(self):
		pass


	def load_numbers(self, filename):
		matrix=fromfile(filename, sep=' ')
		# whole matrix is 1-dim now, so reshape
		matrix=matrix.reshape(2, len(matrix)/2)
		#matrix=matrix.reshape(len(matrix)/2, 2)

		return matrix


	def load_dna(self, filename):
		fh=open(filename, 'r');
		matrix=[]

		# handle row/column brain damage
		for line in fh:
			matrix.append(line[:-1])
		fh.close()

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
