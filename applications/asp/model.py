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
from numpy import mat,array,inf,any,reshape,int32

class model(object):
	#acceptor
	acc_splice_b=None
	acc_splice_order=None
	acc_splice_window_left=None
	acc_splice_window_right=None
	acc_splice_alphas=None
	acc_splice_svs=None

	#donor
	don_splice_b=None
	don_splice_order=None
	don_splice_window_left=None
	don_splice_window_right=None
	don_splice_alphas=None
	don_splice_svs=None

def parse_file(file):
	m=model()

	l=file.readline();

	if l != '%asplicer definition file version: 1.0\n':
		sys.stdout.write("\nfile not a asplicer definition file\n")
		return None

	while l:
		if not ( l.startswith('%') or l.startswith('\n') ): # comment

			#acceptor
			if m.acc_splice_b is None: m.acc_splice_b=parse_value(l, 'acc_splice_b')
			if m.acc_splice_order is None: m.acc_splice_order=parse_value(l, 'acc_splice_order')
			if m.acc_splice_window_left is None: m.acc_splice_window_left=parse_value(l, 'acc_splice_window_left')
			if m.acc_splice_window_right is None: m.acc_splice_window_right=parse_value(l, 'acc_splice_window_right')
			if m.acc_splice_alphas is None: m.acc_splice_alphas=parse_vector(l, file, 'acc_splice_alphas')
			if m.acc_splice_svs is None: m.acc_splice_svs=parse_string(l, file, 'acc_splice_svs')

			#donor
			if m.don_splice_b is None: m.don_splice_b=parse_value(l, 'don_splice_b')
			if m.don_splice_order is None: m.don_splice_order=parse_value(l, 'don_splice_order')
			if m.don_splice_window_left is None: m.don_splice_window_left=parse_value(l, 'don_splice_window_left')
			if m.don_splice_window_right is None: m.don_splice_window_right=parse_value(l, 'don_splice_window_right')
			if m.don_splice_alphas is None: m.don_splice_alphas=parse_vector(l, file, 'don_splice_alphas')
			if m.don_splice_svs is None: m.don_splice_svs=parse_string(l, file, 'don_splice_svs')

		l=file.readline()

	sys.stdout.write('done\n')
	return m

def parse_value(line, name):
	if (line.startswith(name)):
		sys.stdout.write('.'); sys.stdout.flush()
		return float(line[line.find('=')+1:-1])
	else:
		return None

def parse_vector(line, file, name):
    mat = parse_matrix(line, file, name)
    if mat is None:
     return mat
    else:
     mat = array(mat).flatten()
     return mat

def parse_matrix(line, file, name):
	if (line.startswith(name)):
		sys.stdout.write('.'); sys.stdout.flush()
		if line.find(']') < 0:
			l=''
			while l is not None and l.find(']') < 0:
				line+=l
				l=file.readline()
			if l is not None and l.find(']') >= 0:
				line+=l

		if line.find(']') < 0:
			sys.stdout.write("matrix `" + name + "' ended without ']'\n")
			return None
		else:
			mm = mat(line[line.find('['):line.find(']')+1])
			if len(mm.shape)==1:
				mm = reshape(mm.shape[0],1)
			return mm
	else:
		return None

def parse_string(line, file, name):
	if (line.startswith(name)):
		sys.stdout.write('.'); sys.stdout.flush()
		l=''
		lines=[]
		while l is not None and l.find(']') < 0:
			if l:
				lines.append(l[:-1])
			l=file.readline()

		if l.find(']') < 0:
			sys.stdout.write("string ended without ']'\n")
			return None
		else:
			return lines
	else:
		return None

if __name__ == '__main__':
	import bz2
	import sys
	import hotshot, hotshot.stats

	def load():
		f=file('data/asp_test.dat');
		m=parse_file(f);

		print m.acc_splice_b is None
		print m.acc_splice_order is None
		print m.acc_splice_window_left is None
		print m.acc_splice_window_right is None
		print m.acc_splice_alphas is None
		print m.acc_splice_svs is None

		print m.don_splice_b is None
		print m.don_splice_order is None
		print m.don_splice_window_left is None
		print m.don_splice_window_right is None
		print m.don_splice_alphas is None
		print m.don_splice_svs is None

	load()
