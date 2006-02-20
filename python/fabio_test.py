#!/usr/bin/env python

import kernel.WeightedDegreeCharKernel as wd

iA = wd.intArray(5)
dA = wd.doubleArray(5)

dA[0] = 0.167
dA[1] = 0.25
dA[2] = 0.5
dA[3] = 0.25
dA[4] = 0.167

iA[0] = -1
iA[1] = 1
iA[2] = 1
iA[3] = 1
iA[4] = -1

kernel = wd.CWeightedDegreeCharKernel(10,dA,2,0,True,False,1)

seq1 = wd.charArray(3)
seq2 = wd.charArray(3)
seq1[0] = 97
seq1[1] = 98
seq1[2] = 97

seq2[0] = 97
seq2[1] = 98
seq2[2] = 97

import features.CharFeatures as cf

features1 = cf.CCharFeatures(cf.DNA,"fileA.dna")
features2 = cf.CCharFeatures(cf.DNA,"fileB.dna")

kernel.init(features1,features1,True)
kernel.set_precompute_matrix(True,True)


import features.Labels as L

trainlabels = L.CLabels(2)
trainlabels.set_int_label(0,1)
trainlabels.set_int_label(1,-1)

testlabels = L.CLabels(2)
testlabels.set_int_label(0,1)
testlabels.set_int_label(1,1)

import classifier.svm.SVM_light as S

svm = S.CSVMLight()
svm.set_weight_epsilon(1e-5)
svm.set_epsilon(1e-5)
svm.set_tube_epsilon(1e-2)
svm.set_C_mkl(0)
svm.set_C(1,1)
svm.set_qpsize(41)
svm.set_mkl_enabled(False)
svm.set_linadd_enabled(False)
svm.set_labels(trainlabels)
svm.set_kernel(kernel)
svm.set_precomputed_subkernels_enabled(False)
result = svm.train();
print result

svm.classify(trainlabels)
print 'Trainlabels'
for i in range(2):
   print trainlabels.get_label(i)

kmat = kernel.getKernelMatrixReal()
print "Type of kmat is " + str(type(kmat))
print kmat
