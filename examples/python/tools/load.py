#!/usr/bin/env python

from numpy import array, char, double, fromfile

def load_features(filename, type=double):
	if type==char: # numpy.fromfile does not like char
		fh=open(filename, 'r');

		# handle row/column brain damage
		transposed=[]
		for line in fh:
			string=line.split(' ')[0][:-1]
			len_string=len(string)

			if len(transposed)==0:
				for i in xrange(len_string):
					transposed.append([])

			for i in xrange(len_string):
				transposed[i].append(string[i])

		fh.close()

		for i in xrange(len(transposed)):
			transposed[i]="".join(transposed[i])
		matrix=transposed

	else:
		matrix=fromfile(filename, dtype=type, sep=' ')
		# whole matrix is 1-dim now, so reshape
		matrix=matrix.reshape(2, len(matrix)/2)
		#matrix=matrix.reshape(len(matrix)/2, 2)

	return matrix


def load_labels(filename):
	return fromfile(filename, dtype=double, sep=' ')
