%{
 #include <shogun/classifier/svm/SVM_light.h>
%}

%rename(SVMLight) CSVMLight;

%ignore VERSION;
%ignore VERSION_DATE;
%ignore MAXSHRINK;
%ignore SHRINK_STATE;
%ignore MODEL;
%ignore LEARN_PARM;
%ignore TIMING;

%include <shogun/classifier/svm/SVM_light.h>
