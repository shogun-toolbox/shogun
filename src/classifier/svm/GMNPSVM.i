%{
 #include "classifier/svm/GMNPSVM.h" 
%}

%rename(GMNPSVM) CGMNPSVM;

%include "lib/common.i"
%include "classifier/svm/SVM.i" 
%include "classifier/svm/GMNPSVM.h"
