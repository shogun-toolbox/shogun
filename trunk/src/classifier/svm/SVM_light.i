%{
 #include "classifier/svm/SVM_light.h" 
%}

%rename(SVMLight) CSVMLight;

%include "lib/common.i"
%include "classifier/svm/SVM.i" 
%include "classifier/svm/SVM_light.h" 
