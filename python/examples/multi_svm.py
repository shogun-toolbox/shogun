#!/usr/bin/env python
# -*- coding: latin-1 -*-

import features.RealFeatures as rf
import features.Labels as L
import classifier.svm.SVM_light as S
import math
import random

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
    SIZE = 30
    numSVMs = 6

    svmList = [None]*numSVMs

    for j in range(numSVMs):
        pos = ny.random.uniform(0, 2, (SIZE/2, 2)) + random.random()
        neg = ny.random.uniform(0, 2, (SIZE/2, 2)) - random.random()

        trainCoords = ny.mat([pos.ravel(),neg.ravel()])
        trainCoords2 = ny.array(trainCoords.flatten(0))
        trainCoords2 = trainCoords2[0]

        trainfeat = rf.createDoubleArray(trainCoords2)
        traindat = rf.CRealFeatures(trainfeat,2,SIZE)

        import kernel.GaussianKernel as gk

        sigma = 1.0
        kernel = gk.CGaussianKernel(traindat, traindat, 20,sigma)

        trainlabels = L.CLabels(SIZE)
        labels = [0]*SIZE

        for i in range(SIZE):
            if i < SIZE/2:
                trainlabels.set_int_label(i,1)
                labels[i] = 1
            else:
                trainlabels.set_int_label(i,-1)
                labels[i] = -1

        svmList[j] = createSVM(trainlabels, kernel)

        print "Training svm nr. %d\n" % (j)
        currentSVM = svmList[j]
        currentSVM.train()
        print "Trained"

        pylab.subplot(numSVMs/2,2,j+1)
        x1=pylab.linspace(1.5*neg[:,0].min(),1.5*pos[:,0].max(), 50)
        x2=pylab.linspace(1.5*neg[:,1].min(),1.5*pos[:,1].max(), 50)
        x,y=pylab.meshgrid(x1,x2);

        all_features = [0.0]*2*len(x.ravel())

        for i in range(len(x.ravel())):
            all_features[2*i] = x.ravel()[i]
            all_features[2*i+1] = y.ravel()[i]

        test_features = rf.createDoubleArray(all_features)

        testdat = rf.CRealFeatures(test_features,2,len(all_features)/2)
        kernel2 = gk.CGaussianKernel(traindat, testdat, 10,sigma)
        currentSVM.set_kernel(kernel2)

        out = []
        resultLabels = currentSVM.classify()

        for i in range(resultLabels.get_num_labels()):
            out.append(resultLabels.get_label(i))


        out_a = ny.array(out)
        z=ny.reshape(out_a,(50,50))
        pylab.pcolor(x, y, z.transpose(), shading='interp')
        pylab.contour(x, y, z.transpose(), linewidths=1, colors='black', hold=True)
        pylab.scatter(pos[:,0],pos[:,1],s=50, c=labels[:(SIZE/2)], marker='o', hold=True)
        pylab.scatter(neg[:,0],neg[:,1], s=50, c=labels[(SIZE/2):], marker='o', hold=True)

        locals_dict = locals()
        for elem in dir():
            try:
                locals_dict[elem].thisown = 0
            except:
                pass


    for elem in svmList:
        elem.thisown = 0
    
    pylab.show()
