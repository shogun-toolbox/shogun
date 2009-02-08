%{
 #include <shogun/classifier/svm/MultiClassSVM.h>
%}

%newobject classify;
%rename(MultiClassSVM) CMultiClassSVM;

%include <shogun/classifier/svm/MultiClassSVM.h>
%include "LibSVMMultiClass.i"
%include "GMNPSVM.i"
