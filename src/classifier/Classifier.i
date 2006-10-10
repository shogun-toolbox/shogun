%module(directors="1") Classifier
%{
 #include "classifier/Classifier.h" 
%}

%include "lib/common.i"

%feature("director") CClassifier;

%include "classifier/Classifier.h" 
