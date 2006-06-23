#!/usr/bin/env python
# -*- coding: latin-1 -*-

import features.RealFeatures as rf
import features.Labels as L
import classifier.svm.SVM_light as S
import random 

def generateData(size,flag):
    xList = [0.0]*size
    yList = [0.0]*size

    if flag == 1:
        x1 = 2
        x2 = 6
        y1 = 8
        y2 = 12
    else:
        x1 = 4
        x2 = 8
        y1 = 2
        y2 = 6

    for i in range(size):
        xList[i] = random.uniform(x1,x2)
        yList[i] = random.uniform(y1,y2)

    cl = [xList,yList]
    return cl

def createSVM(trainlabels,kernel):
    svm = S.CSVMLight()
    svm.set_weight_epsilon(1e-5)
    svm.set_epsilon(1e-5)
    svm.set_tube_epsilon(1e-2)
    svm.set_C(10,10)
    svm.set_qpsize(41)
    svm.set_mkl_enabled(False)
    svm.set_linadd_enabled(False)
    svm.set_labels(trainlabels)
    svm.set_kernel(kernel)
    svm.set_precomputed_subkernels_enabled(False)
    return svm

if __name__ == '__main__':

    cl1 = generateData(100,1)
    cl2 = generateData(100,2)

    coordList = cl1[0]+cl2[0]+cl1[1]+cl2[1]
    cl1ar = rf.createDoubleArray(coordList)
    cl2ar = rf.createDoubleArray(coordList)

    feat1 = rf.CRealFeatures(cl1ar,2,100)
    feat2 = rf.CRealFeatures(cl2ar,2,100)

    import kernel.GaussianKernel as gk

    kernel = gk.CGaussianKernel(200,1)

    kernel.init(feat1,feat2,True)

    #kernel.set_precompute_matrix(True,True)

    trainlabels = L.CLabels(200)
    for i in range(200):
        if i < 100:
            trainlabels.set_int_label(i,1)
        else:
            trainlabels.set_int_label(i,-1)

    testlabels = L.CLabels(200)
    for i in range(200):
        if i < 100:
            testlabels.set_int_label(i,1)
        else:
            testlabels.set_int_label(i,-1)

    svmList = [None]*20

    for i in range(20):
        svmList[i] = createSVM()

    for j in range(20):
        print "Training svm nr. %d" % (j)
        currentSVM = svmList[j]
        currentSVM.train()


