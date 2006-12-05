%{
 #include "classifier/svm/LibSVM.h" 
%}

%rename(LibSVM) CLibSVM;

%include "lib/common.i"
%include "classifier/svm/SVM.i" 
%include "classifier/svm/LibSVM.h"
