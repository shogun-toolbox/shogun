#!/usr/bin/env python
import modshogun
import pickle
import os
import filecmp
import numpy

from generator import setup_tests, get_fname, blacklist, get_test_mod, run_test

def typecheck(a, b):
	if str(type(a)).find('shogun')>=0 and \
		str(type(a)).find('Labels')>=0 and \
		str(type(b)).find('shogun')>=0 and \
		str(type(b)).find('Labels')>=0:
		 return True
	return type(a) == type(b)


def compare(a, b, tolerance):
	if not typecheck(a,b): return False

	if type(a) == numpy.ndarray: 
		if tolerance:
			return numpy.max(numpy.abs(a - b)) < tolerance
		else:
			return numpy.all(a == b)
	elif isinstance(a, modshogun.SGObject):
		return pickle.dumps(a) == pickle.dumps(b)
	elif type(a) in (tuple,list):
		if len(a) != len(b): return False
		for obj1, obj2 in zip(a,b):
			if not compare(obj1, obj2, tolerance): return False
		return True

	return a == b

def compare_dbg(a, b):
	if not typecheck(a,b):
		print "Type mismatch (type(a)=%s vs type(b)=%s)" % (str(type(a)),str(type(b)))
		return False

	if type(a) == numpy.ndarray:
		if numpy.all(a == b):
			return True
		else:
			print "Numpy Array mismatch"
			print a-b
	elif isinstance(a, modshogun.SGObject):
		if pickle.dumps(a) == pickle.dumps(b):
			return True
		print "a", pickle.dumps(a)
		print "b", pickle.dumps(b)
		return False
	elif type(a) in (tuple,list):
		if len(a) != len(b):
			print "Length mismatch (len(a)=%d vs len(b)=%d)" % (len(a), len(b))
			return False
		for obj1, obj2 in zip(a,b):
			if not compare_dbg(obj1, obj2):
				return False
		return True

	if (a==b):
		return True
	else:
		print "a!=b"
		print "a", a
		print "b", b
		return False

def tester(tests, cmp_method, tolerance, failures):
	for t in tests:
		try:
			mod, mod_name = get_test_mod(t)
		except TypeError:
			continue
		except Exception, e:
			print e
			continue
		fname = ""

		n=len(mod.parameter_list)
		for i in xrange(n):
			fname = get_fname(mod_name, i)
			setting_str = "%s setting %d/%d" % (t,i+1,n)
			try:
				a = run_test(mod, mod_name, i)
				b = pickle.load(file(fname))

				try:
					if cmp_method(a,b,tolerance):
						if not failures:
							print "%-60s OK" % setting_str
					else:
						print "%-60s ERROR" % setting_str
				except Exception, e:
					print setting_str, e

					import pdb
					pdb.set_trace()
			except IOError, e:
				if not failures:
					print "%-60s NO TEST" % (setting_str)
			except Exception, e:
				print "%-60s EXCEPTION %s" % (setting_str,e)
				pass


if __name__=='__main__':
	from optparse import OptionParser
	op=OptionParser()
	op.add_option("-d", "--debug", action="store_true", default=False,
				help="detailed debug output of objects that don't match")
	op.add_option("-f", "--failures", action="store_true", default=False,
				help="show only failures")
	op.add_option("-t", "--tolerance", action="store", default=None,
	              help="tolerance used to estimate accuracy")

	op.set_usage("[<file1> <file2> ...]")
	(opts, args)=op.parse_args()
	if opts.debug:
		cmp_method=compare_dbg
	else:
		cmp_method=compare
	tests = setup_tests(args)
	tester(tests, cmp_method, opts.tolerance, opts.failures)
