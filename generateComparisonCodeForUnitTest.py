#!/usr/bin/env python
# -*- coding: UTF-8  -*-

"""docstring
"""

__revision__ = '0.1'

import sys,os
import getopt

def usage():
    print """python generateComparisonCodeForUnitTest.py
    --input=<input>
    --output=<output>
    --arrname=<array or maxtrix name>
    --funname=<function name>
    --sym=<1 for symmetric, others for asymmetric>
    eg:
    python generateComparisonCodeForUnitTest.py --input=inputfile --arrname=probabilities --funname=get_abs_tolorance_classifier --sym=0
    will output:
    abs_tolorance = get_abs_tolorance_classifier(0.488226466245922, rel_tolorance);
    EXPECT_NEAR(probabilities[0],  0.488226466245922,  abs_tolorance);
    abs_tolorance = get_abs_tolorance_classifier(0.276093816652870, rel_tolorance);
    EXPECT_NEAR(probabilities[2],  0.276093816652870,  abs_tolorance);
    if inputfile contains the following lines
    0.488226466245922
    0.276093816652870
    """

def error():
    usage()
    sys.exit(-1)

def cmdProcess(argv):
    myArgs={
        "defaulArgument1":"",
    }
    try:
        opts, args = getopt.getopt(argv,"h",["help","input=","arrname=","funname=","sym="])
    except getopt.GetoptError:
        error()
    for opt, arg in opts:
        if opt in ("--help","-h"):
            usage()
            sys.exit()
        else:
            opt="".join(opt[2:])
            myArgs[opt]=arg
    return myArgs


if __name__=="__main__":

    argvNum=1
    if len(sys.argv)<=argvNum:
        error()
    myArgs=cmdProcess(sys.argv[1:])
    inf = myArgs['input']
    arrname = myArgs["arrname"]
    funname = myArgs['funname']
    sym = int(myArgs['sym'])
    all = []
    size = -1
    for line in open(inf):
        line = line.strip()
        if len(line) == 0:
            continue
        info = line.split()
        true_info =[]
        for item in info:
            if len(item) == 0:
                continue
            true_info.append(item)
        if len(all) == 0:
            size = len(true_info)
        assert size == len(true_info)
        all.append(true_info)

    pattern = "abs_tolorance = %s(%s, rel_tolorance);\nEXPECT_NEAR(%s,  %s,  abs_tolorance);"
    width = size;
    depth = len(all)
    for i in range(depth):
        for j in range(width):
            if sym == 1 and width >1 and i > j:
                continue
            if width == 1:
                array = "%s[%d]"%(arrname,i)
            else:
                array = "%s(%d,%d)"%(arrname,i,j)
            item = (all[i])[j]
            out = pattern%(funname,item,array,item)
            print out
        if width > 1:
            print ""

