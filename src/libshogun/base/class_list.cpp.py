#!/bin/bash

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2008-2009 Soeren Sonnenburg
# Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society

class_str='class'
types=["BOOL", "CHAR", "INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32",
		"INT64", "UINT64", "FLOAT32", "FLOAT64", "FLOATMAX"] 

def check_class(line):
	if not (line.find('public')==-1 and
			line.find('private')==-1 and
			line.find('protected')==-1):
		return True

def check_abstract_class(line):
	line=line.replace(' ','').replace('\t','').strip()
	return line.endswith('=0;')

def extract_class_name(lines, line_nr, line=None):
	try:
		if not line:
			line=lines[line_nr]
		c=line[line.index(class_str)+len(class_str):]
		if not ':' in c:
			return
		if not check_class(line):
			if not check_class(lines[line_nr+1]):
				return
		c=c.split()[0]
	except:
		return


	c=c.strip(':').strip()

	if not c.startswith('C'):
		return
	if c.endswith(';'):
		return
	if '>' in c:
		return
	if not (len(c)>2 and c[1].isupper()):
		return
	return c[1:]

def get_includes(classes):
	from subprocess import Popen, PIPE 
	from StringIO import StringIO
	cmd=["find", ".", "-false"]
	for c in classes:
		cmd.extend(["-o", "-name", "%s.h" % c])
	p = Popen(cmd, stdout=PIPE)
	output = StringIO(p.communicate()[0])
	includes=[]
	for o in output:
		includes.append('#include "%s"' % o.strip().lstrip('./'))
	return includes

def get_definitions(classes):
	definitions=[]
	for c in classes:
		d="static CSGObject* __new_C%s(EPrimitiveType g) { return g == PT_NOT_GENERIC? new C%s(): NULL; }" % (c,c)
		definitions.append(d)
	return definitions

def get_template_definitions(classes):
	definitions=[]
	for c in classes:
		d=[]
		d.append("static CSGObject* __new_C%s(EPrimitiveType g)\n{\n\tswitch (g)\n\t{\n" % c)
		for t in types:
			if t in ('BOOL','CHAR'):
				suffix=''
			else:
				suffix='_t'
			d.append("\t\tcase PT_%s: return new C%s<%s%s>();\n" % (t,c,t.lower(),suffix))
		d.append("\t\tcase PT_SGOBJECT: return NULL;\n\t}\n\treturn NULL;\n}")
		definitions.append(''.join(d))
	return definitions

def get_struct(classes):
	struct=[]
	for c in classes:
		s='{m_class_name: "%s", m_new_sgserializable: __new_C%s},' % (c,c)
		struct.append(s)
	return struct

def extract_block(c, lines, start_line, stop_line, start_sym, stop_sym):
	sym_cnt=0

	block_start=-1;
	block_stop=-1;

	for line_nr in xrange(start_line, stop_line):
		line=lines[line_nr]
		if line.find(start_sym)!=-1:
			sym_cnt+=1
			if block_start==-1:
				block_start=line_nr
		if line.find(stop_sym)!=-1:
			block_stop=line_nr+1
			sym_cnt-=1
		if sym_cnt==0 and block_start!=-1 and block_stop!=-1:
			return block_start,block_stop

	return block_start,block_stop

def test_candidate(c, lines, line_nr):
	start,stop=extract_block(c, lines, line_nr, len(lines), '{','}')
	if stop<line_nr:
		return False, line_nr+1
	for line_nr in xrange(start, stop):
		line=lines[line_nr]
		if line.find('virtual')!=-1:
			if check_abstract_class(line):
				return False, stop
			else:
				vstart,vstop=extract_block(c, lines, line_nr, len(lines), '(',')')
				for line_nr in xrange(vstart, vstop):
					line=lines[line_nr]
					if check_abstract_class(line):
						return False, stop

	return True, stop


def extract_classes(HEADERS, template):
	"""
	Search in headers for non-template/non-abstract class-names starting
	with `C'.

	Does not support local nor multiple classes and
	drops classes with pure virtual functions
	"""
	classes=list()
	for fname in HEADERS:
		lines=file(fname).readlines()
		line_nr=0
		while line_nr<len(lines):
			line=lines[line_nr]

			if line.find('IGNORE_IN_CLASSLIST')!=-1:
				line_nr+=1
				continue
			c=None
			if template:
				tp=line.find('template')
				if tp!=-1:
					line=line[tp:]
					cp=line.find('>')
					line=line[cp+1:]
					cp=line.find(class_str)
					if cp!=-1:
						c=extract_class_name(lines, line_nr, line)
			else:
				if line.find(class_str)!=-1:
					c=extract_class_name(lines, line_nr)
			if c:
				ok, line_nr=test_candidate(c, lines, line_nr)
				if ok:
					classes.append(c)
				continue

			line_nr+=1
	return classes


def write_templated_file(fname, substitutes):
	template=file(fname).readlines()

	f=file(fname,'w')
	for line in template:
		l=line.strip()
		if l.startswith('REPLACE') and l.endswith('THIS'):
			l=line.split()[1]
			for s in substitutes.iterkeys():
				if l==s:
					f.write('\n'.join(substitutes[s]))
				continue
		else:
			f.write(line)

if __name__=='__main__':
	import sys
	TEMPL_FILE=sys.argv[1]
	HEADERS=sys.argv[2:]

	classes = extract_classes(HEADERS, False)
	template_classes = extract_classes(HEADERS, True)
	includes = get_includes(classes+template_classes)
	definitions = get_definitions(classes)
	template_definitions = get_template_definitions(template_classes)
	struct = get_struct(classes+template_classes)
	substitutes = {'includes': includes,
		'definitions' :definitions,
		'template_definitions' : template_definitions,
		'struct' : struct
		}

	write_templated_file(TEMPL_FILE, substitutes)
