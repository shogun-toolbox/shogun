%module(directors="1") Classifier
%{
 #include "classifier/Classifier.h" 
%}

%include "lib/common.i"

%feature("director") CClassifier;

%rename(Classifer) CClassifier;

%include "classifier/Classifier.h" 

%include "classifier/KNN.i"
%include "classifier/KernelPerceptron.i"
%include "classifier/LDA.i"
%include "classifier/LPM.i"
%include "classifier/LinearClassifier.i"
%include "classifier/Perceptron.i"
%include "classifier/PluginEstimate.i"
