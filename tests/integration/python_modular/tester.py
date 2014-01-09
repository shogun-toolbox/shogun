#!/usr/bin/env python

import modshogun
import pickle
import os
import filecmp
import numpy
import sys
import difflib

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
		# old binary way to compare, via serialization and string comparison
		#return pickle.dumps(a) == pickle.dumps(b)

		# new, parameter framework based comparison up to tolerance
		shogun_tolerance = 1e-5 if tolerance is None else tolerance
		result = a.equals(b, shogun_tolerance)

		# print debug output in case of failure
		if not result:
			print("Equals failed with debug output")
			old_loglevel=a.io.get_loglevel()
			a.io.set_loglevel(1)
			a.equals(b, shogun_tolerance)
			a.io.set_loglevel(old_loglevel)

		return result
	elif type(a) in (tuple,list):
		if len(a) != len(b): return False
		for obj1, obj2 in zip(a,b):
			if not compare(obj1, obj2, tolerance): return False
		return True

	return a == b

def compare_dbg(a, b, tolerance):
	if not compare_dbg_helper(a, b, tolerance):
		import pdb
		pdb.set_trace()

def compare_dbg_helper(a, b, tolerance):
	if not typecheck(a,b):
		print("Type mismatch (type(a)=%s vs type(b)=%s)" % (str(type(a)),str(type(b))))
		return False

	if type(a) == numpy.ndarray:
		if tolerance:
			if numpy.max(numpy.abs(a - b)) < tolerance:
				return True
			else:
				print("Numpy Array mismatch > max_tol")
				print(a-b)
				return False
		else:
			if numpy.all(a == b):
				return True
			else:
				print("Numpy Array mismatch")
				print(a-b)
				return False
	elif isinstance(a, modshogun.SGObject):
		if pickle.dumps(a) == pickle.dumps(b):
			return True
		print("a", pickle.dumps(a))
		print("b", pickle.dumps(b))
		return False
	elif type(a) in (tuple,list):
		if len(a) != len(b):
			print("Length mismatch (len(a)=%d vs len(b)=%d)" % (len(a), len(b)))
			return False
		for obj1, obj2 in zip(a,b):
			if not compare_dbg(obj1, obj2, tolerance):
				return False
		return True

	if (a==b):
		return True
	else:
		print("a!=b")
		print("a", a)
		print("b", b)
		return False

def get_fail_string(a):
	failed_string = []
	if type(a) in (tuple,list):
		for i in range(len(a)):
			failed_string.append(get_fail_string(a[i]))
	elif isinstance(a, modshogun.SGObject):
		failed_string.append(pickle.dumps(a))
	else:
		failed_string.append(str(a))
	return failed_string

def get_split_string(a):
	strs=[]
	for l in a:
		if type(l) in (tuple,list):
			e=l[0]
		else:
			e=l
		strs.extend(e.replace('\\n','\n').splitlines())
	return strs

def tester(tests, cmp_method, tolerance, failures, missing):
	failed=[]

	for t in tests:
		try:
			mod, mod_name = get_test_mod(t)
			n=len(mod.parameter_list)
		except TypeError:
			continue
		except Exception as e:
			print("%-60s ERROR (%s)" % (t,e))
			failed.append(t)
			continue
		fname = ""

		for i in range(n):
			fname = get_fname(mod_name, i)
			setting_str = "%s setting %d/%d" % (t,i+1,n)
			try:
				a = run_test(mod, mod_name, i)

				try:
					b = pickle.load(open(fname))
				except:
					b = pickle.load(open(fname, 'rb'))

				try:
					if cmp_method(a,b,tolerance):
						if not failures and not missing:
							print("%-60s OK" % setting_str)
					else:
						if not missing:
							failed.append((setting_str, get_fail_string(a), get_fail_string(b)))
							print("%-60s ERROR" % setting_str)
				except Exception as e:
					print(setting_str, e)
			except IOError as e:
				if not failures:
					print("%-60s NO TEST (%s)" % (setting_str, e))
			except Exception as e:
				failed.append(setting_str)
				if not missing:
					print("%-60s EXCEPTION %s" % (setting_str,e))
	return failed

if __name__=='__main__':
	import sys
	from optparse import OptionParser
	op=OptionParser()
	op.add_option("-d", "--debug", action="store_true", default=False,
				help="detailed debug output of objects that don't match")
	op.add_option("-f", "--failures", action="store_true", default=False,
				help="show only failures")
	op.add_option("-m", "--missing", action="store_true", default=False,
				help="show only missing tests")
	op.add_option("-t", "--tolerance", action="store", default=None,
	              help="tolerance used to estimate accuracy")

	op.set_usage("[<file1> <file2> ...]")
	(opts, args)=op.parse_args()
	if opts.debug:
		cmp_method=compare_dbg
	else:
		cmp_method=compare
	tests = setup_tests(args)
	failed = tester(tests, cmp_method, opts.tolerance, opts.failures, opts.missing)
	if failed:
		print()
		print("The following tests failed!")
		for f in failed:
			print("\t", f[0])

		print()
		print("Detailled failures:")
		print()
		for f in failed:
			print("\t", f[0])
			got=get_split_string(f[1])
			expected=get_split_string(f[2])
			#print "=== EXPECTED =========="
			#import pdb
			#pdb.set_trace()
			#print '\n'.join(expected)
			#print "=== GOT ==============="
			#print '\n'.join(got)
			print("====DIFF================")
			print('\n'.join(difflib.unified_diff(expected, got, fromfile='expected', tofile='got')))
			print("====EOT================")
			print("\n\n\n")

		sys.exit(1)
	sys.exit(0)
