%{
 #include "classifier/svm/MultiClassSVM.h"
%}

%newobject classify;
%rename(MultiClassSVM) CMultiClassSVM;

%include "classifier/svm/MultiClassSVM.h"
%include "classifier/svm/LibSVMMultiClass.i"
%include "classifier/svm/GMNPSVM.i"
