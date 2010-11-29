import pickle
import os
import filecmp
import numpy

from generator import tests, test_dir, get_fname, blacklist

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

def tester():
    for t in tests:
        if t.endswith(".py") and not t.startswith('.') and t not in blacklist:
            mod_name = t[:-3]
            mod = __import__(mod_name)
            for i in xrange(len(mod.parameter_list)):
                fname = get_fname(mod_name, i)
                par=mod.parameter_list[i]
                a =  getattr(mod, mod_name)(*par)
                b = pickle.load(file(fname))

                if compare(a,b):
                    print t, "setting", i, "OK"
                else:
                    print t, "setting", i, "ERROR"

if __name__=='__main__':
    tester()
