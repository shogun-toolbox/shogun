
%{
 #include "classifier/svm/SVM.h"
 #include "classifier/svm/SVM_light.h"
%}

%include "lib/common.i"
%include "lib/numpy.i"

%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** alphas, INT* d1)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* alphas, INT d)};

%rename(SVM) CSVM;

%include "kernel/KernelMachine.i"
%include "classifier/svm/SVM.h"
%include "classifier/svm/SVM_light.i"
%include "classifier/svm/LibSVM.i"
