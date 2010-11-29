import os
import sys
import pickle

example_dir = '../../examples/undocumented/python_modular'
test_dir = '../../../testsuite/tests'
blacklist = ("structure_dynprog_modular.py",
            "__init__.py",
            "serialization_svmlight_modular.py")

def get_fname(mod_name, i):
    return os.path.join(test_dir, mod_name + str(i) + '.txt')

def setup_tests(tests=[]):
    os.chdir(example_dir)

    if not len(tests):
        tests =  os.listdir(".")
    else:
        tests = [ os.path.basename(t) for t in tests ]

    sys.path.insert(0, '.')
    return tests

def check_for_function(fname):
    for l in file(fname).readlines():
        if l.startswith("def "):
            return True
    return False

def generator(tests):
    for t in tests:
        if t.endswith(".py") and not t.startswith('.') and t not in blacklist:
            mod_name = t[:-3]
            print mod_name,

            if not check_for_function(t):
                print " ERROR (no function)"
                continue

            mod = __import__(mod_name)
            fname = ""

            try:
                for i in xrange(len(mod.parameter_list)):
                    fname = get_fname(mod_name, i)
                    f = open(fname, "w")
                    par=mod.parameter_list[i]
                    a =  getattr(mod, mod_name)(*par)
                    pickle.dump(a,f)
                print " OK"
            except Exception, e:
                fname = get_fname(mod_name, i)
                print " ERROR generating '%s' using '%s'" % (fname,t)
                print e
                continue

if __name__=='__main__':
    from optparse import OptionParser
    op=OptionParser()
    op.set_usage("[<file1> <file2> ...]")
    (opts, args)=op.parse_args()
    tests = setup_tests(args)
    generator(tests)
