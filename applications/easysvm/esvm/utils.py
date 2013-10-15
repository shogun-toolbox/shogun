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

import sys
import random
import numpy
import warnings
import shutil

from shogun.Features import Labels
from shogun.Evaluation import *

################################################################################
# evaluation functions

def confusionMatrix(labels_test, labels_predicted):
    """Compute the matrix of predictions versus labels"""
    if len(labels_test) != len(labels_predicted):
        return 0
    TP = 0; FP = 0; TN = 0; FN = 0
    for i in range(0, len(labels_test)):
        if labels_test[i] == 0 or labels_predicted[i] == 0:
            return 0
        if labels_test[i] > 0:
            if labels_predicted[i] > 0: TP += 1
            else: FN +=1
        else:
            if labels_predicted[i] > 0: FP += 1
            else: TN += 1
    return (TP, TN, FP, FN)

def accuracy(output, labels_test):
    """How many correct predictions?"""
    TP, TN, FP, FN = confusionMatrix(labels_test, numpy.sign(output))
    return float(TP + TN) / (TP + TN + FP + FN)

def calcroc(output, LTE):
    """The area under the receiver operating characteristic curve"""
    pm=ROCEvaluation()
    pm.evaluate(Labels(numpy.array(output)), Labels(numpy.array(LTE)))

    auROC=pm.get_auROC()
    return auROC

def calcprc(output, LTE):
    """The area under the precision recall curve"""
    pm=PRCEvaluation()
    pm.evaluate(Labels(numpy.array(output)), Labels(numpy.array(LTE)))

    auPRC=pm.get_auPRC()
    return auPRC


def calcperf(output, LTE, perflist):
    """Compute all the performance measures in perflist"""
    resperf = []
    for perf in perflist:
        resperf.append(apply(perf,(output,LTE)))

    return resperf


################################################################################
# splitting functions

def getPartitionedSet(total, crossval_repeat, seed=None):
    """Generate a list of indices, splitting the dataset"""
    if seed==None:
        random.seed(123456789)
    else:
        random.seed(seed)

    size = int(total / crossval_repeat)
    mod = total % crossval_repeat

    splits = []
    for i in range(0, crossval_repeat):
        if i < mod:
            splits.append(size + 1)
        else:
            splits.append(size)

    ipartition = random.sample(xrange(0,total), total) # random sampling

    index = 0
    partitions = []

    for size in splits:
        partitions.append(ipartition[index:index+size])
        index += size

    return partitions


def getCurrentSplit(repetition, partitions, labels, seqs):
    """Split the data into training and test sets"""
    X = []; Y = []; XT = []; YT = []
    for i in range(0, len(partitions)):
        if type(seqs) == type(list([])):
            for j in range(0, len(partitions[i])):
                if repetition != i:
                    X.append(seqs[partitions[i][j]])
                    Y.append(labels[partitions[i][j]])
                else:
                    XT.append(seqs[partitions[i][j]])
                    YT.append(labels[partitions[i][j]])
        else:
            if repetition != i:
                if len(X) == 0:
                    X = seqs.take(partitions[i],axis=1)
                    Y = labels.take(partitions[i])
                else:
                    X = numpy.concatenate((X,seqs.take(partitions[i],axis=1)),axis=1)
                    Y = numpy.concatenate((Y,labels.take(partitions[i])))
            else:
                XT = seqs.take(partitions[i],axis=1)
                YT = labels.take(partitions[i])

    return X, Y, XT, YT

################################################################################

def check_params(params, C, max_len):
    """Check for validity of parameters"""
    if (C<=0):
        sys.stderr.write( "\nerror: the parameter 'C' has to be larger than 0\n" )
        assert(C>0)

    if params.has_key("degree"):
        if (params["degree"]<=0):
            sys.stderr.write( "\nerror: the parameter 'degree' has to be larger than 0\n" )
            assert(params["degree"]>0)

    if params.has_key("width"):
	print params["width"]
        if (params["width"]<=0):
            sys.stderr.write( "\nerror: the parameter 'width' has to be larger than 0\n" )
            assert(params["width"]>0)

    if params.has_key("shift"):
        if (params["shift"]<0) or (params["shift"]>max_len):
            sys.stderr.write( "\nerror: the parameter 'shift' has to be larger than 0 and smaller than %i\n" % max_len )
            assert((params["shift"]>=0) and (params["shift"]<=max_len))

    if params.has_key("poim_degree"):
        if params["poim_degree"]>8:
            sys.stderr.write( "\nerror: the parameter 'poim_degree' has to be smaller than 8\n" )
            assert(params["poim_degree"]<=8)

    if params.has_key("crossval_repeat"):
        if params["crossval_repeat"]<1:
            sys.stderr.write( "\nerror: number of cross-validation repeats has to be larger than one\n" )
            assert(params["crossval_repeat"]>1)

    if params.has_key("inhomogene"):
        if params["inhomogene"]!=True and params["inhomogene"]!=False:
            sys.stderr.write( "\nerror: the parameter 'inhomogene' has to be True or False\n" )
            assert(params["inhomogene"]==True or params["inhomogene"]==False)

    if params.has_key("normal"):
        if params["normal"]!=True and params["normal"]!=False:
            sys.stderr.write( "\nerror: the parameter 'normal' has to be True or False\n" )
            assert(params["normal"]==True or params["normal"]==False)

