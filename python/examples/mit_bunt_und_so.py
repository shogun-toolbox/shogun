#!/usr/bin/env python
# -*- coding: latin-1 -*-

import features.RealFeatures as rf
import features.Labels as L
import classifier.svm.SVM_light as S
import random 

#import matplotlib
#from MLab import *
#from numarray import *

def generateData(size,offset):
    xList = [0.0]*size
    yList = [0.0]*size

    x1 = y1 = 2
    x2 = y2 = 8

    for i in range(size):
        xList[i] = random.uniform(x1,x2) + offset
        yList[i] = random.uniform(y1,y2) + offset

    cl = [xList,yList]
    return cl

def createSVM(trainlabels,kernel):
    svm = S.CSVMLight()
    svm.set_C(10,10)
    svm.set_labels(trainlabels)
    svm.set_kernel(kernel)
    return svm

if __name__ == '__main__':
    cl1 = generateData(200,1)
    cl2 = generateData(200,-1)
    
    trainCoords = cl1[0][0:100]+cl2[0][0:100]+cl1[1][0:100]+cl2[1][0:100]
    testCoords = cl1[0][100:200]+cl2[0][100:200]+cl1[1][100:200]+cl2[1][100:200]

    trainfeat = rf.createDoubleArray(trainCoords)
    testfeat = rf.createDoubleArray(testCoords)

    traindat = rf.CRealFeatures(trainfeat,2,200)
    testdat = rf.CRealFeatures(testfeat,2,200)

    import kernel.GaussianKernel as gk

    num = 200
    sigma = 1

    kernel = gk.CGaussianKernel(traindat, traindat, num,sigma)

    trainlabels = L.CLabels(200)
    for i in range(200):
        if i < 100:
            trainlabels.set_int_label(i,1)
        else:
            trainlabels.set_int_label(i,-1)

    svmList = [None]*20

    for i in range(20):
        svmList[i] = createSVM(trainlabels, kernel)

    for j in range(20):
        print "Training svm nr. %d" % (j)
        currentSVM = svmList[j]
        print "Trained"
        currentSVM.train()
        print "Trained"
        
        kernel2 = gk.CGaussianKernel(traindat, testdat, num,sigma)
        currentSVM.set_kernel(kernel2)

        resultLabels = currentSVM.classify()
        for i in range(resultLabels.get_num_labels()):
            print str(resultLabels.get_label(i)) + " ",
            

