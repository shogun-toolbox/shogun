#!/usr/bin/env python
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# Written (W) 2007 Gunnar Raetsch
# Written (W) 2006-2008 Soeren Sonnenburg
# Copyright (C) 2006-2008 Fraunhofer Institute FIRST and Max-Planck-Society
# 

try:
	import os
	import os.path
	import sys
	import pickle
	import bz2
	import numpy
	import optparse
	import array

	import genomic
	import model
	import seqdict
	import shogun.Kernel

	d=shogun.Kernel.WeightedDegreeStringKernel(1)
	if (d.version.get_version_revision() < 2997):
		print
		print "ERROR: SHOGUN VERSION 0.6.2 or later required"
		print
		sys.exit(1)
	from signal_detectors import signal_detectors
except ImportError:
	print
	print "ERROR IMPORTING MODULES, MAKE SURE YOU HAVE SHOGUN INSTALLED"
	print
	sys.exit(1)


asp_version='v0.1'

class asp:
	def __init__(self):
		self.model = None
		self.signal = None
		self.model_name = None

	def load_model(self, filename):
		self.model_name = filename
		sys.stderr.write('loading model file\n')
		f=None
		picklefile=filename+'.pickle'
		if os.path.isfile(picklefile):
			self.model=pickle.load(file(picklefile))	
		else:
			if filename.endswith('.bz2'):
				f=bz2.BZ2File(filename);
			else:
				f=file(filename);

			self.model=model.parse_file(f)
			f.close()

			f=file(picklefile,'w')
			pickle.dump(self.model, f)
			f.close()

		self.signal=signal_detectors(self.model)

	def write_gff(self, outfile, preds, name, skipheader):
		descr=list()
		for i in xrange(len(preds)):
			d=dict()
			d['seqname']=name
			d['source']='asp'
			d['feature']=preds[i][0]
			d['start']=preds[i][1]
			d['end']=preds[i][1]+1
			d['score']=preds[i][2]
			d['strand']='+'
			d['frame']=0
			descr.append(d)

		genomic.write_gff(outfile, ('asp',asp_version + ' ' + self.model_name),
				('DNA', name), descr, skipheader)

	def write_binary(self, preds):
		out=array.array('d')
		out.fromlist([p[2] for p in preds])
		pos=array.array('l')
		pos.fromlist([p[1] for p in preds])
		out.tofile(binary_out)
		pos.tofile(binary_pos)


	def predict_file(self, fname, (start,end)):
		skipheader=False
		fasta_dict = genomic.read_fasta(file(fname))
		sys.stderr.write('found fasta file with ' + `len(fasta_dict)` + ' sequence(s)\n')
		seqs= seqdict.seqdict(fasta_dict, (start,end))

		#get donor/acceptor signal predictions for all sequences
		self.signal.predict_acceptor_sites_from_seqdict(seqs)
		self.signal.predict_donor_sites_from_seqdict(seqs)

		for seq in seqs:
			l=len(seq.preds['donor'].get_positions())
			p=[i+1 for i in seq.preds['donor'].get_positions()]
			s=seq.preds['donor'].get_scores()
			f=[]
			for pos in p:
				if seq.seq[pos-1:pos+1]=='GT':
					f.append(('GT'))
				else:
					f.append(('GC'))

			l=len(seq.preds['acceptor'].get_positions())
			p.extend([i-1 for i in seq.preds['acceptor'].get_positions()])
			s.extend(seq.preds['acceptor'].get_scores())
			f.extend(l*['AG'])
			preds=zip(f,p,s)
			preds.sort(lambda x,y: x[1]-y[1])

			if binary_out and binary_pos:
				self.write_binary(preds)
				binary_out.close()
				binary_pos.close()
			else:
				self.write_gff(outfile, preds, seq.name, skipheader)
				if outfile!='stdout':
					outfile.close()

def print_version():
	sys.stderr.write('asp '+asp_version+'\n')

def parse_options():
	parser = optparse.OptionParser(usage="usage: %prog [options] seq.fa")
	
	parser.add_option("-o", "--outfile", type="str", default='stdout',
			                  help="File to write the results to")
	parser.add_option("-b", "--binary-basename", type="str",
			                  help="Write results in binary format to file starting with this basename")
	parser.add_option("-v", "--version", default=False,
			                  help="Show some more information")
	parser.add_option("--start", type="int", default=499,
			                  help="coding start (zero based, relative to sequence start)")
	parser.add_option("--stop", type="int", default=-499,
			                  help="""coding stop (zero based, if positive relative to
							  sequence start, if negative relative to sequence end)""")
	parser.add_option("--organism", type="str", default='Worm',
			                  help="""use asp model for organism when predicting 
							  (one of Cress, Fish, Fly, Human, Worm)""")

	(options, args) = parser.parse_args()
	if options.version:
		print_version()
		sys.exit(0)

	if len(args) != 1:
		parser.error("incorrect number of arguments")

	fafname=args[0]
	if not os.path.isfile(fafname):
		parser.error("fasta file does not exist")

	modelfname = 'data/%s.dat.bz2' % options.organism
	print "loading model file " + modelfname,
	
	if not os.path.isfile(modelfname):
		print "...not found!\n"
		parser.error("""model should be one of:

worm, fly, cress, fish
""")


	if options.binary_basename and options.outfile != 'stdout':
		parser.error("Only one of the options --binary-basename and --outfile may be given")

	if options.outfile == 'stdout':
		outfile=sys.stdout
	else:
		try:
			outfile=file(options.outfile,'w')
		except IOError:
			parser.error("could not open %s for writing" % options.outfile)

	if options.binary_basename:
		try:
			binary_out=file(options.binary_basename+'.out','w')
			binary_pos=file(options.binary_basename+'.pos','w')
		except IOError:
			parser.error("could not open %s.{out,pos} for writing" % options.binary_basename)
	else:
		binary_out=None
		binary_pos=None

	if options.start<80:
		parser.error("--start value must be >=80")

	if options.stop > 0 and options.start >= options.stop - 80:
		parser.error("--stop value must be > start + 80")

	if options.stop < 0 and options.stop > -80:
		parser.error("--stop value must be <= - 80")

	# shift the start and stop a bit 
	options.start -= 1 ;
	options.stop -= 1 ;
	
	return ((options.start,options.stop), fafname, modelfname, outfile, binary_out, binary_pos)


if __name__ == '__main__':
	(startstop, fafname, modelfname, outfile, binary_out, binary_pos ) = parse_options()
	p=asp()
	p.load_model(modelfname);
	p.predict_file(fafname, startstop)
