%module(directors="1") SVM
%{
 #include "classifier/svm/SVM.h"
 #include "classifier/svm/SVM_light.h"
%}

%include "lib/common.i"

%feature("director") CSVM;

%include "kernel/KernelMachine.i"
%include "classifier/svm/SVM.h"
%include "classifier/svm/SVM_light.h"
%include "classifier/svm/SVM_light.i"
%include "classifier/svm/LibSVM.i"

