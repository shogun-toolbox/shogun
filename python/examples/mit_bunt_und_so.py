#!/usr/bin/env python
# -*- coding: latin-1 -*-

import features.RealFeatures as rf
import features.Labels as L
import classifier.svm.SVM_light as S
import math

import pylab
from MLab import *
import numpy as ny

def createSVM(trainlabels,kernel):
    svm = S.CSVMLight()
    svm.set_C(10,10)
    svm.set_labels(trainlabels)
    svm.set_kernel(kernel)
    return svm

if __name__ == '__main__':
    SIZE = 400
    numSVMs = 10
    num = SIZE

    samples = ny.random.uniform(0, 20.0, (SIZE, 2))

    samples[0:SIZE/2,0] = samples[0:SIZE/2,0] + 1  
    samples[0:SIZE/2,1] = samples[0:SIZE/2,1] + 1  

    samples[SIZE/2:,0] = samples[SIZE/2:,0] - 1  
    samples[SIZE/2:,1] = samples[SIZE/2:,1] - 1  

    trainCoords = samples[:,0] + samples[:,1]

    trainfeat = rf.createDoubleArray(trainCoords)
    testfeat = rf.createDoubleArray(trainCoords)

    traindat = rf.CRealFeatures(trainfeat,2,SIZE)
    testdat = rf.CRealFeatures(testfeat,2,SIZE)
    features = samples

    import kernel.GaussianKernel as gk

    sigma = 1
    kernel = gk.CGaussianKernel(traindat, traindat, num,sigma)

    trainlabels = L.CLabels(SIZE)
    labels = [0]*SIZE

    for i in range(SIZE):
        if i < SIZE/2:
            trainlabels.set_int_label(i,1)
            labels[i] = 1
        else:
            trainlabels.set_int_label(i,-1)
            labels[i] = -1

    svmList = [None]*numSVMs

    for i in range(numSVMs):
        svmList[i] = createSVM(trainlabels, kernel)

    for j in range(numSVMs):
        print "Training svm nr. %d\n" % (j)
        currentSVM = svmList[j]
        currentSVM.train()
        print "Trained"
        
        #kernel2 = gk.CGaussianKernel(traindat, testdat, num,sigma)
        kernel2 = gk.CGaussianKernel(traindat, traindat, num,sigma)
        currentSVM.set_kernel(kernel2)

        pylab.subplot(numSVMs/2,2,j+1)
        x1=pylab.linspace(1.2*(features[:,0]).min(),1.2*(features[:,0]).max(), 20)
        x2=pylab.linspace(1.2*(features[:,1]).min(),1.2*(features[:,1]).max(), 20)
        x,y=pylab.meshgrid(x1,x2);

        out = []
        resultLabels = currentSVM.classify()
        for i in range(resultLabels.get_num_labels()):
            out.append(resultLabels.get_label(i))

        out_a = ny.array(out)
        z=ny.reshape(out_a,(20,20))
        pylab.pcolor(x, y, z.transpose(), shading='flat')
        #pylab.pcolor(x, y, z, shading='flat')
        pylab.scatter(features[:,0],features[:,1], s=20, c=labels, marker='o', hold=True)
        pylab.contour(x, y, z.transpose(), linewidths=1, colors='black', hold=True)
        #pylab.contour(x, y, z, linewidths=1, colors='black', hold=True)
        #pylab.colorbar()

    locals_dict = locals()
    for elem in dir():
        try:
            locals_dict[elem].thisown = 0
        except:
            pass

    for elem in svmList:
        elem.thisown = 0
    
    pylab.show()
