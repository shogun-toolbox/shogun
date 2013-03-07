import os
import sys
import pickle

example_dir = '../../../examples/undocumented/python_modular'
test_dir = '../../../tests/integration/tests'
blacklist = ("__init__.py", "classifier_libsvm_minimal_modular.py",
		"kernel_combined_modular.py",
		"kernel_distance_modular.py",
		"distribution_hmm_modular.py")

def get_fname(mod_name, i):
	return os.path.join(test_dir, mod_name + str(i) + '.txt')

def setup_tests(tests=[]):
	os.chdir(example_dir)

	if not len(tests):
		tests =  os.listdir(".")
		tests.sort()
	else:
		tests = [ os.path.basename(t) for t in tests ]

	sys.path.insert(0, '.')
	return tests

def check_for_function(fname):
	for l in file(fname).readlines():
		if l.startswith("def "):
			return True
	return False

def get_test_mod(t):
	if t.endswith(".py") and not t.startswith('.') and t not in blacklist:
		mod_name = t[:-3]

		if not check_for_function(t):
			raise Exception("ERROR (no function)")

		return __import__(mod_name), mod_name

def run_test(mod, mod_name, i):
	fname = get_fname(mod_name, i)
	par=mod.parameter_list[i]
	a =  getattr(mod, mod_name)(*par)
	return a

def generator(tests):
	for t in tests:
		try:
			mod, mod_name = get_test_mod(t)
		except TypeError:
			continue
		except Exception, e:
			print "%-60s" % mod_name,
			print e
			continue
		fname = ""

		print "%-60s" % mod_name,
		#print "%+60s" % "...",
		try:
			for i in xrange(len(mod.parameter_list)):
				fname = get_fname(mod_name, i)
				a = run_test(mod, mod_name, i)
				pickle.dump(a,file(fname, "w"))
			print "OK"
		except Exception, e:
			print "ERROR generating '%s' using '%s'" % (fname,t)
			print e
			continue

if __name__=='__main__':
	from optparse import OptionParser
	op=OptionParser()
	op.set_usage("[<file1> <file2> ...]")
	(opts, args)=op.parse_args()
	tests = setup_tests(args)
	generator(tests)
