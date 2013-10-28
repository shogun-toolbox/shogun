#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2006-2009 Soeren Sonnenburg
# Written (W) 2006-2007 Mikio Braun
# Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society

import time
from string import maketrans

class ordered_dict(dict):
    """
    Provide an ordered dictionary with chromosome identifiers.
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._order = self.keys()

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self._order.remove(key)

    def ordered_items(self):
        return [(key,self[key]) for key in self._order]


""" read a table browser ascii output file (http://genome.ucsc.edu/cgi-bin/hgTables) """
def read_table_browser(f):
	table=dict();
	for l in f.readlines():
		if not l.startswith('#'):
			(name,chrom,strand,txStart,txEnd,cdsStart,cdsEnd,exonCount,exonStarts,exonEnds,proteinID,alignID)=l.split('\t')
			exonStarts=[ int(i) for i in exonStarts.split(',')[:-1] ]
			exonEnds=[ int(i) for i in exonEnds.split(',')[:-1] ]

			table[name]={ 'chrom': chrom, 'strand': strand, 'txStart': int(txStart), 'txEnd': int(txEnd),
			'cdsStart': int(cdsStart), 'cdsEnd': int(cdsEnd), 'exonCount': int(exonCount), 'exonStarts': exonStarts,
			'exonEnds': exonEnds, 'proteinID': proteinID, 'alignID': alignID[:-1] }

	return table

""" get promoter region """
def get_promoter_region(chromosome, strand, gene_start, gene_end, genome, length):

	if strand == '+':
		return load_genomic(chromosome, strand, gene_start, gene_start+length, genome, one_based=False)
	elif strand == '-':
		return load_genomic(chromosome, strand, gene_end, gene_end+length, genome, one_based=False)
	else:
		print 'unknown strand'
		return None

""" reverse + complement a DNA sequence (only letters ACGT are translated!)
	FIXME won't work with all the rest like y... """
def reverse_complement(str):
	t=maketrans('acgtACGT','tgcaTGCA')
	return str[len(str)::-1].translate(t)

""" works only with .fa files that contain a single entry """
def read_single_fasta(fname):
	str=file(fname).read()
	str=str[str.index('\n')+1:].replace('\n','')
	return str

""" writes only single enty .fa files """
def write_single_fasta(fname, name, str, linelen=60):
	header= '>' + name + '\n'
	f=file(fname,'a')
	f.write(header)
	for i in xrange(0,len(str),linelen):
		f.write(str[i:i+linelen]+'\n')
	f.close()

""" read fasta as dictionary """
def read_fasta(f):
	fasta=ordered_dict()
	fa=""
	key=None
	for s in f.readlines():
		if s.startswith('>'):
			if fa and key:
				fasta[key]=fa
			key=s[1:-1]
			fasta[key]=""
			fa=""
		else:
			fa+=s[:-1]

	if fa and key:
		fasta[key]=fa

	return fasta

def write_fasta(f, d, linelen=60):
    """ write dictionary fasta """
    for k in sorted(d):
        f.write('>%s\n' % k);
        s = d[k]
        for i in xrange(0, len(s), linelen):
            f.write(s[i:i+linelen] + '\n')

def write_gff_header(f, (source, version), (seqtype, seqname)):
	""" writes a gff version 2 file
		descrlist is a list of dictionaries, each of which contain these fields:
		<seqname> <source> <feature> <start> <end> <score> <strand> <frame> [attributes] [comments]
	"""
	f.write('##gff-version 2\n')
	f.write('##source-version %s %s\n' % (source, version) )

	t=time.localtime()
	f.write("##date %d-%d-%d %d:%d:%d\n" % t[0:6])

	f.write('##Type %s %s\n' % (seqtype, seqname) )

def write_gff_line(f, descr):
	d=descr
	f.write('%s\t%s\t%s\t%d\t%d\t%f\t%s\t%d' % (d['seqname'], d['source'],
										d['feature'], d['start'], d['end'],
										d['score'], d['strand'], d['frame']))
	if d.has_key('attributes'):
		f.write('\t' + d['attributes'])
		if d.has_key('comments'):
			f.write('\t' + d['comments'])
	f.write('\n')

def write_spf_header(f, (source, version), (seqtype, seqname)):
	""" writes a gff version 2 file
		descrlist is a list of dictionaries, each of which contain these fields:
		<seqname> <source> <feature> <start> <end> <score> <strand> <frame> [attributes] [comments]
	"""

	f.write('##spf-version 1\n')
	f.write('##source-version %s %s\n' % (source, version) )

	t=time.localtime()
	f.write("##date %d-%d-%d %d:%d:%d\n" % t[0:6])

	f.write('##Type %s %s\n' % (seqtype, seqname) )

def write_spf_line(f, descr):
	d=descr
	f.write('%s\t%s\t%s\t%d\t%s\t%f' % (d['seqname'], d['source'],
										d['feature'], d['position'],
										d['strand'], d['score']))
	if d.has_key('attributes'):
		f.write('\t' + d['attributes'])
		if d.has_key('comments'):
			f.write('\t' + d['comments'])
	f.write('\n')

def write_gff(f, (source, version), (seqtype, seqname), descrlist, skipheader=False):
	""" writes a gff version 2 file
		descrlist is a list of dictionaries, each of which contain these fields:
		<seqname> <source> <feature> <start> <end> <score> <strand> <frame> [attributes] [comments]
	"""

	if not skipheader:
		f.write('##gff-version 2\n')
		f.write('##source-version %s %s\n' % (source, version) )

		t=time.localtime()
		f.write("##date %d-%d-%d %d:%d:%d\n" % t[0:6])

	f.write('##Type %s %s\n' % (seqtype, seqname) )

	for d in descrlist:
		f.write('%s\t%s\t%s\t%d\t%d\t%f\t%s\t%d' % (d['seqname'], d['source'],
											d['feature'], d['start'], d['end'],
											d['score'], d['strand'], d['frame']))
		if d.has_key('attributes'):
			f.write('\t' + d['attributes'])
			if d.has_key('comments'):
				f.write('\t' + d['comments'])
		f.write('\n')

