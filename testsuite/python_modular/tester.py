import pickle
import os
import filecmp
import numpy

from generator import setup_tests, get_fname, blacklist, get_test_mod, run_test

def compare(a, b):
	if type(a) != type(b): return False

	if type(a) == numpy.ndarray: return numpy.all(a == b)
	elif type(a) in (tuple,list):
		if len(a) != len(b): return False
		for obj1, obj2 in zip(a,b):
			if type(obj1) != type(obj2): return False
			if not compare(obj1, obj2): return False
		return True

	return a == b

def tester(tests):
	for t in tests:
		try:
			mod, mod_name = get_test_mod(t)
		except:
			continue
		fname = ""

		try:
			for i in xrange(len(mod.parameter_list)):
				fname = get_fname(mod_name, i)
				a = run_test(mod, mod_name, i)
				b = pickle.load(file(fname))

				if compare(a,b):
					print t, "setting", i, "OK"
				else:
					print t, "setting", i, "ERROR"

		except Exception, e:
			print " ERROR generating '%s' using '%s'" % (fname,t)
			print e
			continue

if __name__=='__main__':
	from optparse import OptionParser
	op=OptionParser()
	op.set_usage("[<file1> <file2> ...]")
	(opts, args)=op.parse_args()
	tests = setup_tests(args)
	tester(tests)
