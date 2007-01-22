%{
 #include "regression/svr/LibSVR.h" 
%}

%rename(LibSVR) CLibSVR;

%include "classifier/svm/LibSVM.i"
%include "regression/svr/LibSVR.h" 
