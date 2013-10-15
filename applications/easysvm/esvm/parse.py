"""
This module contains code to parse the input arguments to the command line:
- easysvm.py
- datagen.py
"""

#############################################################################################
#                                                                                           #
#    This program is free software; you can redistribute it and/or modify                   #
#    it under the terms of the GNU General Public License as published by                   #
#    the Free Software Foundation; either version 3 of the License, or                      #
#    (at your option) any later version.                                                    #
#                                                                                           #
#    This program is distributed in the hope that it will be useful,                        #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of                         #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                           #
#    GNU General Public License for more details.                                           #
#                                                                                           #
#    You should have received a copy of the GNU General Public License                      #
#    along with this program; if not, see http://www.gnu.org/licenses                       #
#    or write to the Free Software Foundation, Inc., 51 Franklin Street,                    #
#    Fifth Floor, Boston, MA 02110-1301  USA                                                #
#                                                                                           #
#############################################################################################

import datafuncs
import sys

################################################################################
# basic types

def parse_range(str):
    list=str.split("-")

    if len(list)==1:
        return (int(list[0]), int(list[0]))
    if len(list)==2:
        return (int(list[0]), int(list[1]))
    sys.stderr.write("Cannot parse range '%s'\n" %str)
    sys.exit(-1)


def parse_float_list(str):
    list=str.split(",")
    float_list=[] ;
    for elem in list:
        float_list.append(float(elem))
    return float_list


def parse_int_list(str):
    list=str.split(",")
    int_list=[] ;
    for elem in list:
        int_list.append(int(elem))
    return int_list


################################################################################
# input files

def parse_input_file_train(kernelname, argv):
    """Parse the input and output file names"""

    if len(argv)<2 or (argv[0]=="fasta" and len(argv)<3) or (argv[0]!='fasta' and argv[0]!='arff'):
        sys.stderr.write("data usage: arff <train.arff>\n        or: fasta <train_pos.fa> <train_neg.fa>\n")
        sys.exit(-1)

    if argv[0] == 'fasta':
        datafilenamepos = argv[1]
        datafilenameneg = argv[2]
        (examples, labels) = datafuncs.fastaread(datafilenamepos, datafilenameneg)
        argv_rest=argv[3:]
    elif argv[0] == 'arff':
        datafilename = argv[1]
        (examples, labels) = datafuncs.arffread(kernelname, datafilename)
        argv_rest=argv[2:]
    else:
        print 'Error in parse_input_file'

    return (examples,labels,argv_rest)


def parse_input_file_train_test(kernelname, argv):
    """Parse the input and output file names"""

    if len(argv)<3 or (argv[0]=="fasta" and len(argv)<4) or (argv[0]!='fasta' and argv[0]!='arff'):
        sys.stderr.write("data usage: arff <train.arff> <test.arff>\n        or: fasta <train_pos.fa> <train_neg.fa> <test.fa>\n")
        sys.exit(-1)

    if argv[0] == 'fasta':
        datafilenamepos = argv[1]
        datafilenameneg = argv[2]
        datafilenametest = argv[3]
        (trainex, trainlab) = datafuncs.fastaread(datafilenamepos, datafilenameneg)
        (testex, testlab) = datafuncs.fastaread(datafilenametest)
        argv_rest=argv[4:]
    elif argv[0] == 'arff':
        datafilename = argv[1]
        datafilenametest = argv[2]
        (trainex, trainlab) = datafuncs.arffread(kernelname, datafilename)
        (testex, testlab) = datafuncs.arffread(kernelname, datafilenametest)
        argv_rest=argv[3:]
    else:
        print 'Error in parse_input_file'

    return (trainex,trainlab,testex,argv_rest)

################################################################################
# prediction file

def parse_prediction_file(fname):
    outputs=[]
    splitassignments=[]

    f = open(fname)
    str=f.read()
    lines = str.split('\n')
    num=0
    for line in lines:
        if len(line)>0 and line[0] != '#':
            elems=line.split('\t')
            assert(len(elems)>1)
            assert(int(elems[0]) == num)
            num+=1
            if len(elems)==2:
                outputs.append(float(elems[1]))
            else:
                assert(len(elems)==3)
                outputs.append(float(elems[1]))
                splitassignments.append(float(elems[2]))
    f.close()
    if len(splitassignments)==0:
        splitassignments = None

    return (outputs, splitassignments)

################################################################################
# kernel parameters

def parse_kernel_param(argv, allow_modelsel_params):
    """Parse the arguments for a particular kernel"""

    if len(argv)<1:
        sys.stderr.write("kernel usage: <kernelname> [<parameters>]\n")
        sys.exit(-1)

    kernelname = argv[0]
    kparam = {}
    kparam["name"]=kernelname
    kparam["modelsel_name"]=None
    kparam["modelsel_params"]=None

    if kernelname == 'gauss':
        if len(argv)<2:
            sys.stderr.write("kernel usage: gauss <width>\n")
            sys.exit(-1)
        if allow_modelsel_params:
            kparam['width'] = None
            kparam["modelsel_name"]="width"
            kparam["modelsel_params"]=parse_float_list(argv[1])
        else:
            kparam['width'] = float(argv[1])
        argv_rest=argv[2:]
    elif kernelname == 'linear':
	kparam['scale']=1
        # no parameters
        argv_rest=argv[1:]
    elif kernelname == 'poly':
        if len(argv)<4:
            sys.stderr.write("kernel usage: poly <degree> <true|false> <true|false>\n")
            sys.exit(-1)
        if allow_modelsel_params:
            kparam['degree'] = None
            kparam["modelsel_name"]="degree"
            kparam["modelsel_params"]=parse_int_list(argv[1])
        else:
            kparam['degree'] = int(argv[1])
        kparam['inhomogene'] = (argv[2] == 'true')
        kparam['normal'] = (argv[3] == 'true')
        argv_rest=argv[4:]
    elif kernelname == 'wd':
        if len(argv)<3:
            sys.stderr.write("kernel usage: wd <degree> <shift>\n")
            sys.exit(-1)
        if allow_modelsel_params:
            kparam['degree'] = None
            kparam["modelsel_name"]="degree"
            kparam["modelsel_params"]=parse_int_list(argv[1])
        else:
            kparam['degree'] = int(argv[1])
        if allow_modelsel_params and len(kparam["modelsel_params"])==1:
            kparam['degree'] = kparam["modelsel_params"][0]
            kparam['shift'] = None
            kparam["modelsel_name"] = "shift"
            kparam["modelsel_params"]=parse_int_list(argv[2])
        else:
            kparam['shift'] = int(argv[2])
        argv_rest=argv[3:]
    elif kernelname == 'spec':
        if len(argv)<2:
            sys.stderr.write("kernel usage: spec <degree>\n")
            sys.exit(-1)
        if allow_modelsel_params:
            kparam['degree'] = None
            kparam["modelsel_name"]="degree"
            kparam["modelsel_params"]=parse_int_list(argv[1])
        else:
            kparam['degree'] = int(argv[1])
        argv_rest=argv[2:]
    elif kernelname == 'localalign':
        # no parameters
        argv_rest=argv[1:]
    elif kernelname == 'localimprove':
        if len(argv)<4:
            sys.stderr.write("kernel usage: localimprove <length> <indegree> <outdegree>\n")
            sys.exit(-1)
        kparam['length'] = int(argv[1])
        if allow_modelsel_params:
            kparam['width'] = None
            kparam["modelsel_name"]="indeg"
            kparam["modelsel_params"]=parse_int_list(argv[2])
        else:
            kparam['indeg'] = int(argv[2])
        kparam['outdeg'] = int(argv[3])
        argv_rest=argv[4:]
    else:
        sys.stderr.write( 'Unknown kernel name %s in parse_kernel_param\n' % kernelname )
        sys.exit(-1)

    return kernelname,kparam,argv_rest

