import pickle
import os
import filecmp
from generator import tests, test_dir, get_fname
import numpy


def compare(a, b):
    if type(a)!=type(b):
        return False

    if type(a) in (tuple,list):
        if len(a)!=len(b):
            return False
        for obj1,obj2 in zip(a,b):
            if type(obj1)!=type(obj2):
                return False
            if type(obj1) == numpy.ndarray:
                if not numpy.all(obj1==obj2):
                    return False
            else:
                if obj1!=obj2:
                    return False
    else:
        return a == b

    return True

def tester():
    for t in tests:
        if t.endswith(".py"):
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
