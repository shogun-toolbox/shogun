#!/usr/bin/env python

import kernel.WeightedDegreeCharKernel as wd
import features.CharFeatures as cf
import features.Labels as L
import classifier.svm.SVM_light as S

dA = wd.createDoubleArray([0.167, 0.25, 0.5, 0.25, 0.167])

feat1 = cf.CCharFeatures(cf.DNA,"fileA.dna")
feat2 = cf.CCharFeatures(cf.DNA,"fileB.dna")

kernel = wd.CWeightedDegreeCharKernel(feat1,feat2,10,dA,2,0,True,False,1)

kernel.set_precompute_matrix(True,True)

trainlabels = L.CLabels(2)
trainlabels.set_int_label(0,1)
trainlabels.set_int_label(1,-1)

testlabels = L.CLabels(2)
testlabels.set_int_label(0,1)
testlabels.set_int_label(1,1)

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
result = svm.train();
print result
print svm.classify_example(1)
#print svm.classify()

#print 'Trainlabels'
#for i in range(2):
#   print trainlabels.get_label(i)

#kmat = kernel.getKernelMatrixReal()
#print "Type of kmat is " + str(type(kmat))
#print kmat
