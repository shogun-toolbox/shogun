%{
 #include "classifier/svm/MultiClassSVM.h"
%}

%rename(MultiClassSVM) CMultiClassSVM;

%include "classifier/svm/MultiClassSVM.h"
%include "classifier/svm/LibSVMMultiClass.i"
%include "classifier/svm/GMNPSVM.i"
