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
	#model matrices
	bins=None
	dict_weights_intron=None
	dict_weights_coding=None
	a_trans=None
	p=None
	q=None

	statedescr = None
	plifidmat = None
	orf_info = None
	use_orf = None

	word_degree = None
	mod_words = None
	sign_words = None

	#penalties
	penalty_acceptor_boundaries=None
	penalty_acceptor_penalty=None
	penalty_donor_boundaries=None
	penalty_donor_penalty=None
	penalty_coding_len_boundaries=None
	penalty_coding_len_penalty=None
	penalty_first_coding_len_boundaries=None
	penalty_first_coding_len_penalty=None
	penalty_last_coding_len_boundaries=None
	penalty_last_coding_len_penalty=None
	penalty_single_coding_len_boundaries=None
	penalty_single_coding_len_penalty=None
	penalty_intron_len_boundaries=None
	penalty_intron_len_penalty=None
	penalty_coding_boundaries=None
	penalty_coding_penalty=None
	penalty_coding2_boundaries=None
	penalty_coding2_penalty=None
	penalty_coding3_boundaries=None
	penalty_coding3_penalty=None
	penalty_coding4_boundaries=None
	penalty_coding4_penalty=None
	penalty_intron_boundaries=None
	penalty_intron_penalty=None
	penalty_intron2_boundaries=None
	penalty_intron2_penalty=None
	penalty_intron3_boundaries=None
	penalty_intron3_penalty=None
	penalty_intron4_boundaries=None
	penalty_intron4_penalty=None
	penalty_transitions_penalty=None

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
	don_splice_use_gc=None
	don_splice_window_left=None
	don_splice_window_right=None
	don_splice_alphas=None
	don_splice_svs=None



def parse_file(file):
	m=model()

	l=file.readline();

	if l != '%msplicer definition file version: 1.0\n':
		sys.stderr.write("\nfile not a msplicer definition file\n")
		return None

	while l:
		if not ( l.startswith('%') or l.startswith('\n') ): # comment
			if m.bins is None: m.bins=parse_value(l, 'bins')
			if m.dict_weights_intron is None: m.dict_weights_intron=parse_matrix(l, file, 'dict_weights_intron')
			if m.dict_weights_coding is None: m.dict_weights_coding=parse_matrix(l, file, 'dict_weights_coding')
			if m.a_trans is None: m.a_trans=parse_matrix(l, file, 'msplicer_a_trans')
			if m.p is None:
				m.p=parse_vector(l, file, 'msplicer_p')
				if m.p is not None:
					m.p[m.p==32768]=-inf
			if m.q is None:
				m.q=parse_vector(l, file, 'msplicer_q')
				if m.q is not None:
					m.q[m.q==32768]=-inf

			if m.statedescr is None:
				m.statedescr=parse_vector(l, file, 'statedescr')
				if m.statedescr is not None:
					m.statedescr=array(m.statedescr, int32)

			if m.plifidmat is None:
				m.plifidmat=parse_matrix(l, file, 'plifidmat')
				if m.plifidmat is not None:
					m.plifidmat = array(m.plifidmat, int32)

			if m.orf_info is None:
				m.orf_info=parse_matrix(l, file, 'orf_info')
				if m.orf_info is not None:
					m.orf_info=array(m.orf_info, int32).T
					if any(m.orf_info != -1):
						m.use_orf = True
					else:
						m.use_orf = False

			if m.word_degree is None: m.word_degree=parse_vector(l, file, 'word_degree')
			if m.mod_words is None: m.mod_words=parse_matrix(l, file, 'mod_words')
			if m.sign_words is None: m.sign_words=parse_vector(l, file, 'sign_words')

			#penalties
			if m.penalty_acceptor_boundaries is None: m.penalty_acceptor_boundaries=parse_vector(l, file, 'penalty_acceptor_boundaries')
			if m.penalty_acceptor_penalty is None: m.penalty_acceptor_penalty=parse_vector(l, file, 'penalty_acceptor_penalty')
			if m.penalty_donor_boundaries is None: m.penalty_donor_boundaries=parse_vector(l, file, 'penalty_donor_boundaries')
			if m.penalty_donor_penalty is None: m.penalty_donor_penalty=parse_vector(l, file, 'penalty_donor_penalty')
			if m.penalty_coding_len_boundaries is None: m.penalty_coding_len_boundaries=parse_vector(l, file, 'penalty_coding_len_boundaries')
			if m.penalty_coding_len_penalty is None: m.penalty_coding_len_penalty=parse_vector(l, file, 'penalty_coding_len_penalty')
			if m.penalty_first_coding_len_boundaries is None: m.penalty_first_coding_len_boundaries=parse_vector(l, file, 'penalty_first_coding_len_boundaries')
			if m.penalty_first_coding_len_penalty is None: m.penalty_first_coding_len_penalty=parse_vector(l, file, 'penalty_first_coding_len_penalty')
			if m.penalty_last_coding_len_boundaries is None: m.penalty_last_coding_len_boundaries=parse_vector(l, file, 'penalty_last_coding_len_boundaries')
			if m.penalty_last_coding_len_penalty is None: m.penalty_last_coding_len_penalty=parse_vector(l, file, 'penalty_last_coding_len_penalty')
			if m.penalty_single_coding_len_boundaries is None: m.penalty_single_coding_len_boundaries=parse_vector(l, file, 'penalty_single_coding_len_boundaries')
			if m.penalty_single_coding_len_penalty is None: m.penalty_single_coding_len_penalty=parse_vector(l, file, 'penalty_single_coding_len_penalty')
			if m.penalty_intron_len_boundaries is None: m.penalty_intron_len_boundaries=parse_vector(l, file, 'penalty_intron_len_boundaries')
			if m.penalty_intron_len_penalty is None: m.penalty_intron_len_penalty=parse_vector(l, file, 'penalty_intron_len_penalty')
			if m.penalty_coding_boundaries is None: m.penalty_coding_boundaries=parse_vector(l, file, 'penalty_coding_boundaries')
			if m.penalty_coding_penalty is None: m.penalty_coding_penalty=parse_vector(l, file, 'penalty_coding_penalty')
			if m.penalty_coding2_boundaries is None: m.penalty_coding2_boundaries=parse_vector(l, file, 'penalty_coding2_boundaries')
			if m.penalty_coding2_penalty is None: m.penalty_coding2_penalty=parse_vector(l, file, 'penalty_coding2_penalty')
			if m.penalty_coding3_boundaries is None: m.penalty_coding3_boundaries=parse_vector(l, file, 'penalty_coding3_boundaries')
			if m.penalty_coding3_penalty is None: m.penalty_coding3_penalty=parse_vector(l, file, 'penalty_coding3_penalty')
			if m.penalty_coding4_boundaries is None: m.penalty_coding4_boundaries=parse_vector(l, file, 'penalty_coding4_boundaries')
			if m.penalty_coding4_penalty is None: m.penalty_coding4_penalty=parse_vector(l, file, 'penalty_coding4_penalty')
			if m.penalty_intron_boundaries is None: m.penalty_intron_boundaries=parse_vector(l, file, 'penalty_intron_boundaries')
			if m.penalty_intron_penalty is None: m.penalty_intron_penalty=parse_vector(l, file, 'penalty_intron_penalty')
			if m.penalty_intron2_boundaries is None: m.penalty_intron2_boundaries=parse_vector(l, file, 'penalty_intron2_boundaries')
			if m.penalty_intron2_penalty is None: m.penalty_intron2_penalty=parse_vector(l, file, 'penalty_intron2_penalty')
			if m.penalty_intron3_boundaries is None: m.penalty_intron3_boundaries=parse_vector(l, file, 'penalty_intron3_boundaries')
			if m.penalty_intron3_penalty is None: m.penalty_intron3_penalty=parse_vector(l, file, 'penalty_intron3_penalty')
			if m.penalty_intron4_boundaries is None: m.penalty_intron4_boundaries=parse_vector(l, file, 'penalty_intron4_boundaries')
			if m.penalty_intron4_penalty is None: m.penalty_intron4_penalty=parse_vector(l, file, 'penalty_intron4_penalty')
			if m.penalty_transitions_penalty is None: m.penalty_transitions_penalty=parse_vector(l, file, 'penalty_transitions_penalty')

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
			if m.don_splice_use_gc is None: m.don_splice_use_gc=parse_value(l, 'don_splice_use_gc')
			if m.don_splice_window_left is None: m.don_splice_window_left=parse_value(l, 'don_splice_window_left')
			if m.don_splice_window_right is None: m.don_splice_window_right=parse_value(l, 'don_splice_window_right')
			if m.don_splice_alphas is None: m.don_splice_alphas=parse_vector(l, file, 'don_splice_alphas')
			if m.don_splice_svs is None: m.don_splice_svs=parse_string(l, file, 'don_splice_svs')

		l=file.readline()

	sys.stderr.write('done\n')
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
			sys.stderr.write("matrix `" + name + "' ended without ']'\n")
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
			sys.stderr.write("string ended without ']'\n")
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
		#f=bz2.BZ2File('data/msplicer_arabidopsis10_gc=1_orf=0.dat.bz2');
		f=file('data/msplicer_arabidopsis10_gc=1_orf=0.dat');
		m=parse_file(f);

		print m.penalty_acceptor_boundaries is None
		print m.penalty_acceptor_penalty is None
		print m.penalty_donor_boundaries is None
		print m.penalty_donor_penalty is None
		print m.penalty_coding_len_boundaries is None
		print m.penalty_coding_len_penalty is None
		print m.penalty_first_coding_len_boundaries is None
		print m.penalty_first_coding_len_penalty is None
		print m.penalty_last_coding_len_boundaries is None
		print m.penalty_last_coding_len_penalty is None
		print m.penalty_single_coding_len_boundaries is None
		print m.penalty_single_coding_len_penalty is None
		print m.penalty_intron_len_boundaries is None
		print m.penalty_intron_len_penalty is None
		print m.penalty_coding_boundaries is None
		print m.penalty_coding_penalty is None
		print m.penalty_coding2_boundaries is None
		print m.penalty_coding2_penalty is None
		print m.penalty_coding3_boundaries is None
		print m.penalty_coding3_penalty is None
		print m.penalty_coding4_boundaries is None
		print m.penalty_coding4_penalty is None
		print m.penalty_intron_boundaries is None
		print m.penalty_intron_penalty is None
		print m.penalty_intron2_boundaries is None
		print m.penalty_intron2_penalty is None
		print m.penalty_intron3_boundaries is None
		print m.penalty_intron3_penalty is None
		print m.penalty_intron4_boundaries is None
		print m.penalty_intron4_penalty is None
		print m.penalty_transitions_penalty is None

		print m.acc_splice_b is None
		print m.acc_splice_order is None
		print m.acc_splice_window_left is None
		print m.acc_splice_window_right is None
		print m.acc_splice_alphas is None
		print m.acc_splice_svs is None

		print m.don_splice_b is None
		print m.don_splice_order is None
		print m.don_splice_use_gc is None
		print m.don_splice_window_left is None
		print m.don_splice_window_right is None
		print m.don_splice_alphas is None
		print m.don_splice_svs is None

	load()

	#prof = hotshot.Profile("model.prof")
	#benchtime = prof.runcall(load)
	#prof.close()
	#stats = hotshot.stats.load("model.prof")
	#stats.strip_dirs()
	#stats.sort_stats('time', 'calls')
	#stats.print_stats(20)
