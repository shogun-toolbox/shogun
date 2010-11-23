import pickle
import os
import filecmp
tests = os.listdir("./Test")

for i in tests:
    if i.endswith(".py"):
        mod_name = i[0:len(i)-3]
        mod = __import__(mod_name)
        f = open('./example_files/' +mod_name + '.txt','r')
        for j in mod.parameter_list:
            a =  getattr(mod, mod_name)(*j)
            print a
            if (a==pickle.load(f)):
                print "gleich"




