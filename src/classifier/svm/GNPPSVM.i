%{
 #include "classifier/svm/GNPPSVM.h" 
%}

%rename(GNPPSVM) CGNPPSVM;

%include "lib/common.i"
%include "classifier/svm/SVM.i" 
%include "classifier/svm/GNPPSVM.h"
