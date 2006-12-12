%{
 #include "classifier/svm/SVMLin.h" 
%}

%rename(SVMLin) CSVMLin;

%include "lib/common.i"
%include "classifier/svm/SVM.i" 
%include "classifier/svm/SVMLin.h"
