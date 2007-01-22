%{
 #include "regression/svr/SVR_light.h" 
%}

%rename(SVRLight) CSVRLight;

%include "classifier/svm/SVM_light.i"
%include "regression/svr/SVR_light.h" 
