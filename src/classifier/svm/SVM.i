%module(directors="1") SVM
%{
 #define SWIG_FILE_WITH_INIT
 #include "classifier/svm/SVM.h"
 #include "classifier/svm/SVM_light.h"
%}

%include "lib/common.i"

%init %{
   import_array();
%}

%include "lib/numpy.i"

%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** alphas, INT* d1)};

%feature("director");
%rename(SVM) CSVM;

%include "kernel/KernelMachine.i"
%include "classifier/svm/SVM.h"
%include "classifier/svm/SVM_light.i"
%include "classifier/svm/LibSVM.i"
